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
from src.generate import generate_command


project_dir = Path(__file__).resolve().parent.parent
train_data_file = project_dir / "data" / "processed" / "train.json"
val_data_file = project_dir / "data" / "processed" / "val.json"

checkpoint_dir = project_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

train_data = pd.read_json(train_data_file)
val_data = pd.read_json(val_data_file)

# sample_df = df.sample(n=500, random_state=42)

# another_sample = df.drop(
#     index=sample_df.index
# ).sample(n=20, random_state=1)


checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

special_tokens = {
    "bos_token": "<|bos|>",
    "pad_token": "<|pad|>"
}

tokenizer.add_special_tokens(special_tokens)



train_dataset = LinuxCommandsDataset(
    df=train_data,
    tokenizer=tokenizer
)

val_dataset = LinuxCommandsDataset(
    df=val_data,
    tokenizer=tokenizer
)

data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    
)

val_data_loader = DataLoader(
    dataset=val_dataset,
    batch_size=2,
    shuffle=False,
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

device = ("cuda" if torch.cuda.is_available() else "cpu")

transformer = transformer.to(device)


criterion = nn.CrossEntropyLoss(ignore_index=-100)

optimizer = AdamW(params=transformer.parameters(), lr=0.001)

num_epochs = 10

train_losses = []
val_losses = []


for epoch in range(num_epochs):

    transformer.train()

    running_loss = 0.0
    val_loss = 0.0

    for i, example in enumerate(data_loader):

        optimizer.zero_grad()

        source, decoder_input, labels = example

        source, decoder_input, labels = source.to(device), decoder_input.to(device), labels.to(device)  

        encoder_mask, decoder_mask = create_mask(
            source=source,
            decoder_input=decoder_input,
            pad_token_id=tokenizer.pad_token_id
        )

            

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


    transformer.eval()

    for i, example in enumerate(val_data_loader):
        with torch.no_grad():
            source, decoder_input, labels = example
            source, decoder_input, labels = source.to(device), decoder_input.to(device), labels.to(device)

            encoder_mask, decoder_mask = create_mask(
                source=source,
                decoder_input=decoder_input,
                pad_token_id=tokenizer.pad_token_id
            )

            encoded_input = transformer.encode(source, encoder_mask)
            decoded_input = transformer.decode(decoder_input, encoded_input, encoder_mask, decoder_mask)
            projected_output = transformer.project(decoded_input)
    
            y_predicted = projected_output.reshape(-1, len(tokenizer))
            y = labels.reshape(-1)
    
            loss = criterion(y_predicted, y)

            val_loss += loss.item()
        
    train_loss = running_loss / len(data_loader)
    validation_loss = val_loss / len(val_data_loader)

    print(
        f"Epoch [{epoch + 1:02d}/{num_epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {validation_loss:.4f}"
    )


    train_losses.append(train_loss)
    val_losses.append(validation_loss)

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": transformer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": validation_loss
    }

    torch.save(
        checkpoint,
        checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
    )

# correct = 0

# for _, row in another_sample.iterrows():

#     description = row["input"]
#     expected = row["output"]

#     generated = generate_command(
#         transformer=transformer,
#         tokenizer=tokenizer,
#         description=description,
#         max_len=96
#     )

#     is_correct = generated == expected

#     if is_correct:
#         correct += 1

#     print(f"\nDescription : {description}")
#     print(f"Expected: {expected}")
#     print(f"Generated: {generated}")
#     print(f"Result: {'Ok' if is_correct else 'NOT OK'}")

# print(f"\nExact Match : {correct}/{len(another_sample)}")