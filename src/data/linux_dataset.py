from transformers import AutoTokenizer
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

project_dir = Path(__file__).resolve().parent.parent


train_data_file = project_dir / "data" / "processed" / "train.json"

df = pd.read_json(train_data_file)

x_train = df["input"]
y_train = df["output"]


source_sample = x_train[0]
target_sample = y_train[0]






# tokenized_source = tokenizer.tokenize(source_sample)
# tokenized_target = tokenizer.tokenize(target_sample)

# ids_source = tokenizer.convert_tokens_to_ids(tokenized_source)
# ids_target = tokenizer.convert_tokens_to_ids(tokenized_target)



# source_ids = ids_source + [tokenizer.eos_token_id]
# decoder_input_ids = [tokenizer.bos_token_id] + ids_target
# labels = ids_target + [tokenizer.eos_token_id]



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
            tokens + [self.tokenizer.eos_token_id] 
            for tokens in tokenized_source
        ]


        tokenized_target = self.tokenizer(
            self.target.tolist(),
            add_special_tokens=False
        )["input_ids"]

        self.tokenized_target = [
            [self.tokenizer.bos_token_id] + tokens
            for tokens in tokenized_target
        ]

        self.labels = [
            tokens + [self.tokenizer.eos_token_id]
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

def collate_fn(batch, pad_token_id):
    sources = [element[0] for element in batch]
    decoder_inputs = [element[1] for element in batch]
    labels = [element[2] for element in batch]

    padded_source = pad_sequence(
        sequences=sources,
        batch_first=True,
        padding_value=pad_token_id
    )

    padded_decoder_inputs = pad_sequence(
        sequences=decoder_inputs,
        batch_first=True,
        padding_value=pad_token_id
    )

    padded_labels = pad_sequence(
        sequences=labels,
        batch_first=True,
        padding_value = -100
    )

    return padded_source, padded_decoder_inputs, padded_labels




"""

BOS: beginning of sequence
EOS: end of sequence
PAD: padding token

"""




