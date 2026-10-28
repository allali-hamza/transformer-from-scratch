from transformers import AutoTokenizer
from huggingface_hub import HfApi
from pathlib import Path
import torch
import json


project_dir = Path(__file__).resolve().parent.parent

checkpoint_file = project_dir / "checkpoints" / "checkpoint_epoch_6.pt"
output_dir = project_dir / "model_export"

repo_id = "hamza-allali/linux-command-transformer"

output_dir.mkdir(parents=True, exist_ok=True)


checkpoint = torch.load(
    checkpoint_file,
    map_location="cpu",
    weights_only=True
)

torch.save(
    checkpoint["model_state_dict"],
    output_dir / "model.pt"
)


tokenizer = AutoTokenizer.from_pretrained("gpt2")

special_tokens = {
    "bos_token": "<|bos|>",
    "pad_token": "<|pad|>"
}

tokenizer.add_special_tokens(special_tokens)

tokenizer.save_pretrained(output_dir / "tokenizer")


config = {
    "architecture": "Transformer Encoder-Decoder",
    "src_vocab_size": len(tokenizer),
    "tgt_vocab_size": len(tokenizer),
    "src_seq_len": 64,
    "tgt_seq_len": 96,
    "d_model": 64,
    "N": 2,
    "h": 4,
    "dropout": 0,
    "d_ff": 128,
    "tokenizer": "gpt2",
    "best_epoch": checkpoint["epoch"],
    "validation_loss": checkpoint["val_loss"]
}

with open(output_dir / "config.json", "w") as f:
    json.dump(config, f, indent=4)


api = HfApi()

api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True
)

api.upload_folder(
    folder_path=str(output_dir),
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload trained Transformer checkpoint and tokenizer"
)

print(f"Model uploaded to https://huggingface.co/{repo_id}")