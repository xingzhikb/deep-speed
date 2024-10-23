import json
import os
import datasets

logger = datasets.logging.get_logger(__name__)

_BASE_IMAGE_METADATA_FEATURES = {
    "image_id": datasets.Value("int32"),
    "width": datasets.Value("int32"),
    "height": datasets.Value("int32"),
    "file_name": datasets.Value("string"),
    "task_type": datasets.Value("string"),    
}

_BASE_REGION_FEATURES = {
    "region_id": datasets.Value("int32"),
    "image_id": datasets.Value("int32"),
    "phrases": [datasets.Value("string")],
    "x": datasets.Value("int32"),
    "y": datasets.Value("int32"),
    "width": datasets.Value("int32"),
    "height": datasets.Value("int32"),
}

_ANNOTATION_FEATURES = {
    "regions": [_BASE_REGION_FEATURES],
}

class CustomDatasetConfig(datasets.BuilderConfig):
    def __init__(
        self,
        name,
        splits,
        data_dir: str = None,        
        with_image: bool = True,
        task_type: str = "recognition",        
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.splits = splits
        self.with_image = with_image
        self.task_type = task_type
        self.data_dir = data_dir        
    @property
    def features(self):
        return datasets.Features(
            {
                **({"image": datasets.Image()} if self.with_image else {}),
                **_BASE_IMAGE_METADATA_FEATURES,
                **_ANNOTATION_FEATURES,
            }
        )

class CustomDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIG_CLASS = CustomDatasetConfig
    BUILDER_CONFIGS = [
        CustomDatasetConfig(name="custom", splits=["train", "validation"]),
    ]
    DEFAULT_CONFIG_NAME = "custom"
    config: CustomDatasetConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = self.config.data_dir
        if data_dir is None:
            raise ValueError(
                "This script is supposed to work with a local dataset. The argument `data_dir` in `load_dataset()` is required."
            )

        splits = []
        for split in self.config.splits:
            if split == "train":
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "json_path": os.path.join(data_dir, "train.json"),
                    },
                )
            elif split in ["val", "valid", "validation", "dev"]:
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "json_path": os.path.join(data_dir, "test.json"),  # Using test.json for validation
                    },
                )
            else:
                continue

            splits.append(dataset)

        return splits

    def _generate_examples(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        for idx, image in enumerate(data["images"]):
            image_id = image["image_id"]
            image_metadata = {
                "file_name": image["file_name"],
                "height": image["height"],
                "width": image["width"],
                "image_id": image["image_id"],
            }

            annotations = [
                ann for ann in data["annotations"] if ann["image_id"] == image_id
            ]

            regions = [{
                "region_id": ann["region_id"],
                "image_id": ann["image_id"],
                "phrases": ann["phrases"],
                "x": ann["x"],
                "y": ann["y"],
                "width": ann["width"],
                "height": ann["height"],
            } for ann in annotations]

            image_dict = {"image": os.path.join(self.config.data_dir, image["file_name"])} if self.config.with_image else {}

            yield idx, {**image_dict, **image_metadata, "regions": regions, "task_type": self.config.task_type}