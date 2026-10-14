from transformers import AutoTokenizer
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

project_dir = Path(__file__).resolve().parent.parent


train_data_file = project_dir / "data" / "processed" / "train.json"

df = pd.read_json(train_data_file)

x_train = df["input"]
y_train = df["output"]


source_sample = x_train[0]
target_sample = y_train[0]


checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


special_tokens = {
    "bos_token": "<|bos|>",
    "pad_token": "<|pad|>"
}

tokenizer.add_special_tokens(special_tokens)




tokenized_source = tokenizer.tokenize(source_sample)
tokenized_target = tokenizer.tokenize(target_sample)

ids_source = tokenizer.convert_tokens_to_ids(tokenized_source)
ids_target = tokenizer.convert_tokens_to_ids(tokenized_target)



source_ids = ids_source + [tokenizer.eos_token_id]
decoder_input_ids = [tokenizer.bos_token_id] + ids_target
labels = ids_target + [tokenizer.eos_token_id]



class LinuxCommandsDataset(Dataset):
    def __init__(self, df, tokenizer):

        self.df = df
        self.tokenizer = tokenizer
        self.source = self.df["input"]
        self.target = self.df["output"]

        tokenized_source = self.tokenizer(
            self.source.tolist(),
            add_special_tokens=False
        )["input_ids"]

        self.tokenized_source = [
            tokens + [tokenizer.eos_token_id] 
            for tokens in tokenized_source
        ]


        tokenized_target = self.tokenizer(
            self.target.tolist(),
            add_special_tokens=False
        )["input_ids"]

        self.tokenized_target = [
            [tokenizer.bos_token_id] + tokens
            for tokens in tokenized_target
        ]

        self.labels = [
            tokens + [tokenizer.eos_token_id]
            for tokens in tokenized_target
        ]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        source_ids = torch.tensor(
            self.tokenized_source[idx],
            dtype=torch.long
        )
        decoder_input_ids = torch.tensor(
            self.tokenized_target[idx],
            dtype=torch.long
        )
        labels = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return source_ids, decoder_input_ids, labels


df = pd.read_json(train_data_file)

my_data = LinuxCommandsDataset(df, tokenizer)

print(my_data[0])

"""

BOS: beginning of sequence
EOS: end of sequence
PAD: padding token

"""




