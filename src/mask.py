import torch

def create_mask(source, decoder_input, pad_token_id):
    
    source_batch_size, source_sequence_length = source.shape
    decoder_input_batch_size, decoder_input_sequence_length = decoder_input.shape

    source_mask = source != pad_token_id
    source_mask = source_mask.reshape(source_batch_size, 1, 1, source_sequence_length)

    causel_mask = torch.tril(
        input=torch.ones(decoder_input_sequence_length, decoder_input_sequence_length, dtype=torch.bool, device=decoder_input.device)
    )
    decoder_mask = decoder_input != pad_token_id
    decoder_mask = decoder_mask.reshape(decoder_input_batch_size, 1, 1, decoder_input_sequence_length)
    combined_mask = causel_mask & decoder_mask


    return source_mask, combined_mask
