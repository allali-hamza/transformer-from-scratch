from .masks import create_src_mask, create_tgt_mask
import torch

def generate_command(transformer, tokenizer, description, max_len):

    device = next(transformer.parameters()).device

    transformer.eval()

    decoder_input = torch.tensor([tokenizer.bos_token_id], dtype=torch.long, device=device)
    decoder_input = decoder_input.reshape(1, -1)

    with torch.no_grad():
        
        tokenized_description = tokenizer(description)["input_ids"]
        tokenized_description = tokenized_description + [tokenizer.eos_token_id]

        tokenized_description = torch.tensor(tokenized_description, dtype=torch.long, device=device)
        source = tokenized_description.reshape(1, -1)

        source_mask = create_src_mask(
            source=source,
            pad_token_id=tokenizer.pad_token_id
        ).to(device)

        encoder_output = transformer.encode(source, source_mask)

        for _ in range(max_len):

            decoder_combined_mask = create_tgt_mask(
                decoder_input=decoder_input,
                pad_token_id=tokenizer.pad_token_id
            ).to(device)

            decoder_output = transformer.decode(decoder_input, encoder_output, source_mask, decoder_combined_mask)

            projected_output = transformer.project(decoder_output)

            logits_for_generated_token = projected_output[0][-1]

            predicted_token_id = torch.argmax(logits_for_generated_token, dim=-1).item()

            next_token = torch.tensor(
                [[predicted_token_id]],
                dtype=torch.long,
                device=device
            )

            decoder_input = torch.cat((decoder_input, next_token), dim=-1).to(device)

            
            if predicted_token_id == tokenizer.eos_token_id:
                break

        command = tokenizer.decode(decoder_input[0][1:].tolist(),skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return command

    
