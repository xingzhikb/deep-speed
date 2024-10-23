import json
from PIL import Image
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import datasets
from tqdm import tqdm
import dotenv
from ast import literal_eval
from git_utils.tsv_io import TSVFile
import subprocess
import tempfile
from contextlib import contextmanager
import hashlib
from datasets.utils.filelock import FileLock
import shutil


logger = datasets.logging.get_logger(__name__)

_CITATION = "TBD"

_DESCRIPTION = """\
SA1B, each mask region is annotated with a phrase describing the region.
the phrases are generated by GIT-2 model captioning masked objects on a
white background. 
"""
_HOMEPAGE = "TBD"
_LICENSE = "TBD"

_LATEST_VERSIONS = {
    "mask_region_descriptions": "0.0.1",
}

_BASE_IMAGE_METADATA_FEATURES = {
    "image_id": datasets.Value("int32"),
    "width": datasets.Value("int32"),
    "height": datasets.Value("int32"),
    "task_type": datasets.Value("string"),
}

_BASE_REGION_FEATURES = {
    "region_id": datasets.Value("int64"),
    "image_id": datasets.Value("int32"),
    "phrases": [datasets.Value("string")],
    "x": datasets.Value("int32"),
    "y": datasets.Value("int32"),
    "width": datasets.Value("int32"),
    "height": datasets.Value("int32"),
}


_BASE_MASK_FEATURES = {
    "size": [datasets.Value("int32")],
    "counts": datasets.Value("string"),
}

_BASE_MASK_REGION_FEATURES = {
    "region_id": datasets.Value("int64"),
    "image_id": datasets.Value("int32"),
    "phrases": [datasets.Value("string")],
    "x": datasets.Value("int32"),
    "y": datasets.Value("int32"),
    "width": datasets.Value("int32"),
    "height": datasets.Value("int32"),
    "mask": _BASE_MASK_FEATURES,
    # "area": datasets.Value("int32"),
    # "phrase_conf": datasets.Value("float32"),
}


_ANNOTATION_FEATURES = {
    "region_descriptions": {"regions": [_BASE_REGION_FEATURES]},
    "mask_region_descriptions": {"regions": [_BASE_MASK_REGION_FEATURES]},
}


class SA1BCapConfig(datasets.BuilderConfig):
    """BuilderConfig for SA1BCap."""

    def __init__(
        self,
        name: str,
        version: Optional[str] = None,
        with_image: bool = True,
        with_mask: bool = True,
        # 0
        sa1b_tar_url: Optional[str] = None,
        sa1b_tar_template: Optional[str] = None,
        # 1
        sa1b_annot_tsv_url: Optional[str] = None,
        sa1b_annot_template: Optional[str] = None,
        # 2
        sa1b_cap_tsv_url: Optional[str] = None,
        sa1b_cap_template: Optional[str] = None,
        # 3
        sa1b_filter_tsv_url: Optional[str] = None,
        sa1b_filter_template: Optional[str] = None,
        # 4
        sa1b_file_range: Optional[List[int]] = None,
        # 5
        training_args: Optional[Any] = None,
        # 6
        task_type: str = "caption",
        **kwargs,
    ):
        """BuilderConfig for SA1BCap.
        there should be **no dynamic** computation in __init__.
        The Config is first init in the DatasetBuilder constructor,
        then the attr here are to be modified in `load_dataset`.

        Args:
            name_version: name and version of the dataset.
            description: description of the dataset.
            image_dir: directory containing the images.
            annotation_dir: directory containing the annotations.
            **kwargs: keyword arguments forwarded to super.
        """
        _version = _LATEST_VERSIONS[name] if version is None else version
        # NOTE: f"{name}_v{_version}" is the param for `load_dataset`
        _name = f"{name}_v{_version}"
        super().__init__(version=datasets.Version(_version), name=_name, **kwargs)

        self._name_without_version = name

        # NOTE: the following attr can be overwritten by `load_dataset`
        self.with_image = with_image
        self.with_mask = with_mask

        self.sa1b_tar_url = sa1b_tar_url
        self.sa1b_tar_template = sa1b_tar_template

        self.sa1b_annot_tsv_url = sa1b_annot_tsv_url
        self.sa1b_annot_template = sa1b_annot_template

        self.sa1b_cap_tsv_url = sa1b_cap_tsv_url
        self.sa1b_cap_template = sa1b_cap_template

        self.sa1b_filter_tsv_url = sa1b_filter_tsv_url
        self.sa1b_filter_template = sa1b_filter_template

        self.sa1b_file_range = sa1b_file_range

        # NOTE: To determine whether it is main process or not
        self.training_args = training_args

        self.task_type = task_type

    @property
    def features(self):
        annoation_type = "mask_region_descriptions" if self.with_mask else "region_descriptions"
        logger.info(f"Using annotation type: {annoation_type} due to with_mask={self.with_mask}")
        return datasets.Features(
            {
                **({"image": datasets.Image()} if self.with_image else {}),
                **_BASE_IMAGE_METADATA_FEATURES,
                **_ANNOTATION_FEATURES[annoation_type],
            }
        )


class SA1BCap(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")
    BUILDER_CONFIG_CLASS = SA1BCapConfig
    BUILDER_CONFIGS = [*[SA1BCapConfig(name="mask_region_descriptions", version=version) for version in ["0.0.1"]]]
    DEFAULT_CONFIG_NAME = "region_descriptions_v0.0.1"
    config: SA1BCapConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            version=self.config.version,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        sa1b_tar_url = self.config.sa1b_tar_url
        sa1b_annot_tsv_url = self.config.sa1b_annot_tsv_url
        sa1b_cap_tsv_url = self.config.sa1b_cap_tsv_url
        sa1b_filter_tsv_url = self.config.sa1b_filter_tsv_url

        sa1b_tar_template = self.config.sa1b_tar_template
        sa1b_annot_template = self.config.sa1b_annot_template
        sa1b_cap_template = self.config.sa1b_cap_template
        sa1b_filter_template = self.config.sa1b_filter_template

        sa1b_file_range = self.config.sa1b_file_range

        if sa1b_tar_url is None:
            raise ValueError("sa1b_tar_url is None")
        if sa1b_annot_tsv_url is None:
            raise ValueError("sa1b_annot_tsv_url is None")
        if sa1b_cap_tsv_url is None:
            raise ValueError("sa1b_cap_tsv_url is None")
        if sa1b_file_range is None:
            raise ValueError("sa1b_file_range is None. We need the exact file range to load the dataset.")

        try:
            sa1b_file_range = literal_eval(sa1b_file_range)
        except ValueError as e:
            sa1b_file_range = eval(sa1b_file_range)
        except Exception as e:
            logger.error(f"Failed to literal_eval sa1b_file_range: {e}")
            raise ValueError(f"Failed to literal_eval sa1b_file_range: {e}")

        _DL_URLS = {}

        # NOTE(xiaoke): load sas_key from .env
        logger.info(f"Try to load sas_key from .env file: {dotenv.load_dotenv('.env')}.")

        sa1b_tar_url_sas_key = os.getenv("SA1B_TAR_URL_SAS_KEY", None)
        if sa1b_tar_url_sas_key is None or os.path.exists(sa1b_tar_url):
            sa1b_tar_url_sas_key = ""
        _DL_URLS["sa1b_tar_urls"] = self._build_sa1b_urls(
            sa1b_tar_url, sa1b_tar_template, sa1b_file_range, sa1b_tar_url_sas_key
        )

        sa1b_annot_tsv_url_sas_key = os.getenv("SA1B_ANNOT_TSV_URL_SAS_KEY", None)
        if sa1b_annot_tsv_url_sas_key is None or os.path.exists(sa1b_annot_tsv_url):
            sa1b_annot_tsv_url_sas_key = ""
        _DL_URLS["sa1b_annot_tsv_urls"] = self._build_sa1b_urls(
            sa1b_annot_tsv_url, sa1b_annot_template, sa1b_file_range, sa1b_annot_tsv_url_sas_key
        )

        sa1b_cap_tsv_url_sas_key = os.getenv("SA1B_CAP_TSV_URL_SAS_KEY", None)
        if sa1b_cap_tsv_url_sas_key is None or os.path.exists(sa1b_cap_tsv_url):
            sa1b_cap_tsv_url_sas_key = ""
        _DL_URLS["sa1b_cap_tsv_urls"] = self._build_sa1b_urls(
            sa1b_cap_tsv_url, sa1b_cap_template, sa1b_file_range, sa1b_cap_tsv_url_sas_key
        )

        if sa1b_filter_tsv_url is None:
            logger.info(f"sa1b_filter_tsv_url is None, not filtering dataset.")
        else:
            sa1b_filter_tsv_url_sas_key = os.getenv("SA1B_FILTER_TSV_URL_SAS_KEY", None)
            if sa1b_filter_tsv_url_sas_key is None or os.path.exists(sa1b_filter_tsv_url):
                sa1b_filter_tsv_url_sas_key = ""
            _DL_URLS["sa1b_filter_tsv_urls"] = self._build_sa1b_urls(
                sa1b_filter_tsv_url, sa1b_filter_template, sa1b_file_range, sa1b_filter_tsv_url_sas_key
            )

        if dl_manager.is_streaming is False:
            raise ValueError("dl_manager.is_streaming is False. We need to stream the dataset. Because it is too big.")

        file_urls = dl_manager.download(_DL_URLS)
        num_tars = len(file_urls["sa1b_tar_urls"])
        self._num_tars = num_tars
        list_of_file_urls = []
        for num_tar in range(num_tars):
            list_of_file_urls.append(
                {
                    "sa1b_tar_url": file_urls["sa1b_tar_urls"][num_tar],
                    "sa1b_annot_tsv_url": file_urls["sa1b_annot_tsv_urls"][num_tar],
                    "sa1b_cap_tsv_url": file_urls["sa1b_cap_tsv_urls"][num_tar],
                    "sa1b_filter_tsv_url": file_urls["sa1b_filter_tsv_urls"][num_tar]
                    if "sa1b_filter_tsv_urls" in file_urls
                    else None,
                    "tar_idx": num_tar,
                }
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "list_of_file_urls": list_of_file_urls,  # NOTE: It would be sharded as https://huggingface.co/docs/datasets/dataset_script#sharding, which would be much faster for downloading.
                    "iter_archive_func": dl_manager.iter_archive,
                },
            ),
        ]

    def _build_sa1b_urls(self, url, template, _range, sas_key):
        url_template = os.path.join(url, template)
        return [f"{url_template.format(i)}{sas_key}" for i in _range]

    def _generate_examples(self, list_of_file_urls, iter_archive_func):
        num_tars = len(list_of_file_urls)
        for i, one_file_urls in enumerate(list_of_file_urls):
            logger.info(f"Processing tar {one_file_urls['tar_idx']}/{self._num_tars}")
            tar_data_iter = self._process_one_tar(iter_archive_func, **one_file_urls)
            for image_id, data in tar_data_iter:
                yield image_id, data

    def _get_tsv_file(self, tsv_url):
        return TSVFile(tsv_url, open_func=open)

    def _process_one_tar(
        self,
        iter_archive_func,
        sa1b_tar_url,
        sa1b_annot_tsv_url,
        sa1b_cap_tsv_url,
        sa1b_filter_tsv_url=None,
        tar_idx=-1,
    ):
        # The `open` function of Python is extened with streaming loading from the Internet by `xopen` in `datasets.download.streaming_download_manager`.
        # After that, `xopen` is futher patched into `open` by `datasets.streaming`.

        sa1b_annot_tsv = self._get_tsv_file(sa1b_annot_tsv_url)

        sa1b_cap_tsv = self._get_tsv_file(sa1b_cap_tsv_url)

        sa1b_filter_tsv = None
        if sa1b_filter_tsv_url is not None:
            sa1b_filter_tsv = self._get_tsv_file(sa1b_filter_tsv_url)

        mapping_image_id_region_id_to_annot = self.build_mapping_image_id_region_id_to_annot(
            sa1b_annot_tsv, sa1b_cap_tsv, desc_prefix=f"[tar_idx={tar_idx}/{self._num_tars}]"
        )
        mapping_image_id_to_annots = self.build_mapping_image_id_to_annots(
            mapping_image_id_region_id_to_annot, desc_prefix=f"[tar_idx={tar_idx}/{self._num_tars}]"
        )
        del mapping_image_id_region_id_to_annot

        # NOTE: filter dataset if any:
        with TempFileForAzcopy(sa1b_tar_url) as _sa1b_tar_url:
            for name, buffer in iter_archive_func(_sa1b_tar_url):
                if name.endswith(".json"):
                    continue
                yield self._process_one_sample(name, buffer, mapping_image_id_to_annots)

    def _process_one_sample(self, name, buffer, mapping_image_id_to_annots):
        # name = './sa_%d.jpg"
        name = os.path.basename(name)
        image_id = int(name.split(".")[0].split("_")[-1])

        if self.config.with_image:
            # NOTE: check here see how hugging face datasets handle image
            # https://github.com/huggingface/datasets/blob/8b9649b3cfb49342e44873ce7e29e0c75eaf3efa/src/datasets/features/image.py#L130
            image = Image.open(buffer)
            image.load()
            if image.mode != "RGB":
                image = image.convert("RGB")
                logger.warning(f"convert {image_id} from {image.mode} to RGB")
            image_dict = dict(
                image=image,
                image_id=image_id,
                width=image.width,
                height=image.height,
            )
        else:
            image_dict = dict(
                image_id=image_id,
                width=-1,
                height=-1,
            )
        # convert to RGB is time consuming, from 5 it/s to 1it/s
        # image = image.convert("RGB")

        regions = mapping_image_id_to_annots[image_id]

        return image_id, dict(
            **image_dict,
            regions=regions,
            task_type=self.config.task_type,
        )

    def build_mapping_image_id_region_id_to_annot(self, annot_tsv, cap_tsv, desc_prefix=""):
        if len(annot_tsv) != len(cap_tsv):
            raise ValueError(
                f"len(annot_tsv) != len(cap_tsv): {len(annot_tsv)} != {len(cap_tsv)}. "
                f"Please check the data integrity for {annot_tsv} and {cap_tsv}."
            )

        # NOTE: Build index for fast retrieval of annoation.
        # This is compromised design as the tar file is extracted to image_id.json and image_id.jpg
        # NOTE: size: 965765982 bytes, 921.5 MB
        image_id_region_id_to_annot: Dict[int, Dict[int, List]] = defaultdict(dict)
        for cnt, (annot, cap) in enumerate(
            tqdm(
                zip(annot_tsv, cap_tsv),
                desc=f"{desc_prefix} building image_id_region_id_to_annot.",
                total=len(annot_tsv),
            )
        ):
            if annot[0] != cap[0]:
                raise ValueError(f"Cnt: {cnt}: annot[0] != cap[0], {annot[0]} != {cap[0]}, in {annot} != {cap}")

            # NOTE: identifier format is image_id-region_cnt-region_id
            image_id, region_cnt, region_id = list(map(int, cap[0].split("-")))

            annot_obj = json.loads(annot[1])  # Dict[str, Any], i.e. SA1B format
            # TODO: maybe update to other caption format
            cap_obj = json.loads(cap[1])  # NOTE: List[Dict[str, Any]], i.e. "caption" and "conf" from GIT2

            image_id_region_id_to_annot[image_id][region_id] = annot_obj
            image_id_region_id_to_annot[image_id][region_id]["captions"] = cap_obj

        return image_id_region_id_to_annot

    def build_mapping_image_id_to_annots(self, mapping_image_id_region_id_to_annot, desc_prefix):
        mapping_image_id_to_annots = {}
        for image_id, region_id_to_annot in tqdm(
            mapping_image_id_region_id_to_annot.items(),
            desc=f"{desc_prefix} building image_id_to_annots...",
            total=len(mapping_image_id_region_id_to_annot),
        ):
            annots = []
            for annot in region_id_to_annot.values():
                # _BASE_MASK_REGION_FEATURES
                region_id = annot["id"]
                image_id: int
                # TODO: maybe update to other caption format
                phrases = [caption["caption"] for caption in annot["captions"]]
                x, y, width, height = annot["bbox"]
                mask = annot["segmentation"]

                # Unused by model, but useful for filtering
                # phrase_conf = raw_annot["conf"]
                # area = raw_annot["area"]

                transformed_annot = dict(
                    region_id=region_id,
                    image_id=image_id,
                    phrases=phrases,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    # area=area,
                    # phrase_conf=phrase_conf,
                )
                if self.config.with_mask:
                    transformed_annot["mask"] = mask
                annots.append(transformed_annot)

            mapping_image_id_to_annots[image_id] = annots
        return mapping_image_id_to_annots


class TempFileForAzcopy:
    def __init__(self, file_url):
        self.file_url = file_url
        self.temp_dir = self._get_temp_dir(file_url)
        self.temp_file = None
        self.lock_path = None

    def _get_lock_file_name(self, fname):
        path = urlparse(fname).path
        name = os.path.basename(path)
        return os.path.join(self.temp_dir, name), os.path.join(self.temp_dir, name + ".lock")

    def _get_temp_dir(self, fname):
        with tempfile.NamedTemporaryFile() as fp:
            base_temp_dir = os.path.dirname(fp.name)
        hash_str = hashlib.md5(fname.encode()).hexdigest()
        return os.path.join(base_temp_dir, "sa1b_cap-" + hash_str)

    def _is_file_open(self, file_path):
        return (
            subprocess.run(
                ["lsof", file_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )

    def _remove_unopened_file(self, file_path):
        if self.temp_dir not in file_path:
            return

        logger.info("Try to remove file {}.".format(file_path))

        if self._is_file_open(file_path):
            logger.info(f"{file_path} is still open.")
        else:
            logger.info(f"{file_path} is all closed. So we remove it.")

            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Successfully remove file {file_path}.")

            lock_file = file_path + ".lock"
            if os.path.exists(lock_file):
                os.remove(lock_file)
                logger.info(f"Successfully remove lock file {lock_file}.")

        if os.path.exists(self.temp_dir):
            if os.listdir(self.temp_dir) == 0:
                logger.info(f"{self.temp_dir} is not empty. So we do not remove it.")
            else:
                logger.info(f"Successfully remove temp dir {self.temp_dir} for {self.file_url}")
                shutil.rmtree(self.temp_dir, ignore_errors=True)

    def __enter__(self):
        has_azcopy = subprocess.run(["azcopy"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        has_azcopy = has_azcopy == 0
        file_url = self.file_url

        if "://" not in file_url:
            logger.debug("file_url is directory path.")
            return file_url
        if not has_azcopy:
            logger.warning("azcopy is not installed, skip using azcopy to prepare azure url.")
            return file_url

        if "blob.core.windows.net" not in file_url:
            logger.warning(f"file_url is not azure blob url, skip using azcopy to prepare azure url: {file_url}")
            return file_url

        temp_file, lock_path = self._get_lock_file_name(file_url)
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)
        with FileLock(lock_path):
            try:
                result = subprocess.run(
                    ["azcopy", "cp", file_url, temp_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if result.returncode != 0:
                    raise ConnectionError(f"azcopy failed with return code {result.returncode}")
                logger.info(f"Successfully azcopy {file_url} to {temp_file}.")
                self.temp_file = temp_file
                self.lock_path = lock_path
                return temp_file

            except Exception as e:
                logger.error(f"azcopy failed with exception {e}. Use regular xopen instead which can be slow.")
                if os.path.isfile(temp_file):
                    os.remove(temp_file)
                if os.path.isfile(lock_path):
                    os.remove(lock_path)
                return file_url

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_unopened_file(self.temp_file)

    def __del__(self):
        self._remove_unopened_file(self.temp_file)
