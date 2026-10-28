import sys
import json
from pathlib import Path
import torch

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from src.model import build_transformer, generate_command


def load_model():

    model_dir = Path(snapshot_download(
        repo_id="hamza-allali/linux-command-transformer"
    ))

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")

    transformer = build_transformer(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        src_seq_len=config["src_seq_len"],
        tgt_seq_len=config["tgt_seq_len"],
        d_model=config["d_model"],
        N=config["N"],
        h=config["h"],
        dropout=config["dropout"],
        d_ff=config["d_ff"]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transformer = transformer.to(device)

    state_dict = torch.load(
        model_dir / "model.pt",
        map_location=device,
        weights_only=True
    )

    transformer.load_state_dict(state_dict)
    transformer.eval()

    return transformer, tokenizer, config


if __name__ == "__main__":

    description = " ".join(sys.argv[1:])

    if not description:
        raise SystemExit("Please provide a description.")

    transformer, tokenizer, config = load_model()

    with torch.inference_mode():
        command = generate_command(
            transformer=transformer,
            tokenizer=tokenizer,
            description=description,
            max_len=config["tgt_seq_len"]
        )

    print(command)
