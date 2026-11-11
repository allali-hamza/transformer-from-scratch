import torch

from src.model.transformer import InputEmbeddings


def test_input_embeddings_apply_sqrt_d_model_scaling():
    embeddings = InputEmbeddings(d_model=4, vocab_size=3)
    with torch.no_grad():
        embeddings.embedding.weight.fill_(1.0)

    output = embeddings(torch.tensor([[0, 2]]))

    torch.testing.assert_close(output, torch.full((1, 2, 4), 2.0))
