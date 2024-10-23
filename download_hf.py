
from huggingface_hub import login

# 输入你的 Hugging Face Token
huggingface_token = "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"  # 在此处输入你的 Hugging Face 访问令牌

# 登录 Hugging Face 账号
login(token=huggingface_token)

from huggingface_hub import hf_hub_download

# Specify the repository ID, the filename, and the directory to download to
repo_id = "UCSC-VLAA/MedTrinity-25M-FULL"  # Replace with your Hugging Face repository ID
filename = "dataset_shard_67.tar.zst"  # Replace with the specific file you want to download
local_dir = "/workspace"  # Replace with your desired local directory

# Download the file
file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

print(f"File downloaded to: {file_path}")
