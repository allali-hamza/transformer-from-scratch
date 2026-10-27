from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path
import torch

from src.model import build_transformer
from src.generate import generate_command


project_dir = Path(__file__).resolve().parent.parent

test_data_file = project_dir / "data" / "processed" / "test.json"
checkpoint_file = project_dir / "checkpoints" / "checkpoint_epoch_6.pt"

test_data = pd.read_json(test_data_file)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

special_tokens = {
    "bos_token": "<|bos|>",
    "pad_token": "<|pad|>"
}

tokenizer.add_special_tokens(special_tokens)


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

device = "cuda" if torch.cuda.is_available() else "cpu"

transformer = transformer.to(device)

checkpoint = torch.load(
    checkpoint_file,
    map_location=device,
    weights_only=True
)

transformer.load_state_dict(checkpoint["model_state_dict"])
transformer.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"Validation Loss: {checkpoint['val_loss']:.4f}")


correct_exact = 0
correct_command = 0
correct_first_argument = 0
correct_command_and_argument = 0

total = len(test_data)
total_with_argument = 0

results = []

with torch.inference_mode():

    for i, row in enumerate(test_data.itertuples(index=False), start=1):

        description = row.input
        expected = row.output

        generated = generate_command(
            transformer=transformer,
            tokenizer=tokenizer,
            description=description,
            max_len=96
        )

        expected = expected.strip()
        generated = generated.strip()

        expected_tokens = expected.split()
        generated_tokens = generated.split()

        exact_match = expected == generated

        command_match = (
            len(expected_tokens) >= 1
            and len(generated_tokens) >= 1
            and expected_tokens[0] == generated_tokens[0]
        )

        first_argument_match = False
        command_and_argument_match = False

        if len(expected_tokens) >= 2:

            total_with_argument += 1

            first_argument_match = (
                len(generated_tokens) >= 2
                and expected_tokens[1] == generated_tokens[1]
            )

            command_and_argument_match = (
                command_match and first_argument_match
            )

            if first_argument_match:
                correct_first_argument += 1

            if command_and_argument_match:
                correct_command_and_argument += 1

        if exact_match:
            correct_exact += 1

        if command_match:
            correct_command += 1

        results.append({
            "description": description,
            "expected": expected,
            "generated": generated,
            "exact_match": exact_match,
            "command_match": command_match,
            "first_argument_match": first_argument_match if len(expected_tokens) >= 2 else None,
            "command_and_argument_match": command_and_argument_match if len(expected_tokens) >= 2 else None
        })

        if i % 100 == 0 or i == total:
            print(f"Evaluated {i}/{total} examples")


exact_match_accuracy = correct_exact / total if total else 0
command_accuracy = correct_command / total if total else 0

first_argument_accuracy = (
    correct_first_argument / total_with_argument
    if total_with_argument else 0
)

command_and_argument_accuracy = (
    correct_command_and_argument / total_with_argument
    if total_with_argument else 0
)


print("\nEvaluation Results")

print(f"Epoch: {checkpoint['epoch']}")
print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
print(f"Total Examples: {total}")

print(f"Exact Match Accuracy: {exact_match_accuracy:.2%}")
print(f"Command Accuracy: {command_accuracy:.2%}")
print(f"First Argument Accuracy: {first_argument_accuracy:.2%}")
print(f"Command + Argument Accuracy: {command_and_argument_accuracy:.2%}")


results_df = pd.DataFrame(results)

results_dir = project_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

results_file = results_dir / "evaluation_results.csv"

results_df.to_csv(results_file, index=False)

print(f"\nPredictions saved to: {results_file}")