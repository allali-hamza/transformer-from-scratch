from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from src.linux_dataset import LinuxCommandsDataset, collate_fn
from src.mask import create_mask
from src.model import build_transformer



project_dir = Path(__file__).resolve().parent.parent
train_data_file = project_dir / "data" / "processed" / "train.json"
df = pd.read_json(train_data_file)

sample_df = df.sample(n=10, random_state=42)

checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

special_tokens = {
    "bos_token": "<|bos|>",
    "pad_token": "<|pad|>"
}

tokenizer.add_special_tokens(special_tokens)



dataset = LinuxCommandsDataset(
    df=sample_df,
    tokenizer=tokenizer
)

data_loader = DataLoader(
    dataset=dataset,
    batch_size=2,
    collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    
)

batch_iter = iter(data_loader)
source, decoder_input, labels = next(batch_iter)


transformer = build_transformer(
    src_vocab_size=len(tokenizer),
    tgt_vocab_size=len(tokenizer),
    src_seq_len=64,
    tgt_seq_len=96,
    d_model=64,
    N=2,
    h=4, 
    dropout=0,
    d_ff=128
)

encoder_mask, decoder_mask = create_mask(
    source=source,
    decoder_input=decoder_input,
    pad_token_id=tokenizer.pad_token_id
)

encoded_input = transformer.encode(source, encoder_mask)
decoded_input = transformer.decode(decoder_input, encoded_input, encoder_mask, decoder_mask)
projected_output = transformer.project(decoded_input)


print(encoded_input.shape)
print(decoded_input.shape)
print(projected_output.shape)
