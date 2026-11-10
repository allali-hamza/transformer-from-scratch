import torch

from src.model.transformer import PositionalEncoding


def test_positional_encoding_adds_position_values_without_gradients():
    encoding = PositionalEncoding(d_model=4, max_seq_len=5, dropout=0.0)

    output = encoding(torch.zeros(1, 3, 4))

    torch.testing.assert_close(output, encoding.pe[:, :3, :])
    assert not encoding.pe.requires_grad
    assert not torch.equal(output[:, 0, :], output[:, 1, :])
