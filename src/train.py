from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import torch

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# encoder_mask, decoder_mask = create_mask(
#     source=source,
#     decoder_input=decoder_input,
#     pad_token_id=tokenizer.pad_token_id
# )

# encoded_input = transformer.encode(source, encoder_mask)
# decoded_input = transformer.decode(decoder_input, encoded_input, encoder_mask, decoder_mask)
# projected_output = transformer.project(decoded_input)


# print(encoded_input.shape)
# print(decoded_input.shape)
# print(projected_output.shape)

transformer.train()

criterion = nn.CrossEntropyLoss(ignore_index=-100)

optimizer = AdamW(params=transformer.parameters(), lr=0.001)

train_loses = []
num_epochs = 50

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, example in enumerate(data_loader):

        optimizer.zero_grad()

        source, decoder_input, labels = example
        encoder_mask, decoder_mask = create_mask(
            source=source,
            decoder_input=decoder_input,
            pad_token_id=tokenizer.pad_token_id
        )

        # if torch.cuda.is_available():
            

        encoded_input = transformer.encode(source, encoder_mask)
        decoded_input = transformer.decode(decoder_input, encoded_input, encoder_mask, decoder_mask)
        projected_output = transformer.project(decoded_input)

        y_predicted = projected_output.reshape(-1, len(tokenizer))
        y = labels.reshape(-1)

        loss = criterion(y_predicted, y)
        # train_loses.append(loss.item())

        loss.backward()
        clip_grad_norm_(transformer.parameters(), max_norm=1)
        optimizer.step()

        running_loss += loss.item()
    
    print(f"[Epoch {epoch + 1}] Loss: {running_loss / (i + 1):.4f}")








# y_predicted = projected_output.reshape(-1, len(tokenizer))
# y = labels.reshape(-1)


# transformer.train()

# encoded_input = transformer.encode(source, encoder_mask)
# decoded_input = transformer.decode(decoder_input, encoded_input, encoder_mask, decoder_mask)

# y_predicted = projected_output.reshape(-1, len(tokenizer))
# y = labels.reshape(-1)

# loss_before = criterion(y_predicted, y)
# print("Loss before:", loss_before.item())


# optimizer.zero_grad()
# loss = criterion(y_predicted, y)
# loss.backward()
# clip_grad_norm_(transformer.parameters(), max_norm=1)
# optimizer.step()

# transformer.eval()

# with torch.no_grad():

#     encoded_input = transformer.encode(
#         source,
#         encoder_mask
#     )

#     decoded_input = transformer.decode(
#         decoder_input,
#         encoded_input,
#         encoder_mask,
#         decoder_mask
#     )

#     projected_output = transformer.project(decoded_input)

#     y_predicted = projected_output.reshape(-1, len(tokenizer))

#     loss_after = criterion(y_predicted, y)


# print("Loss after:", loss_after.item())