import torch
from torch import nn

from src.model.transformer import Decoder, DecoderBlock, FeedForwardBlock, MultiHeadAttentionBlock


def test_decoder_runs_its_blocks_and_preserves_shape():
    decoder = Decoder(nn.ModuleList([
        DecoderBlock(
            MultiHeadAttentionBlock(4, 2, 0.0),
            MultiHeadAttentionBlock(4, 2, 0.0),
            FeedForwardBlock(4, 8, 0.0),
            dropout=0.0,
        )
    ]))
    target = torch.randn(2, 2, 4)

    output = decoder(
        target,
        torch.randn(2, 3, 4),
        torch.ones(2, 1, 1, 3, dtype=torch.bool),
        torch.tril(torch.ones(1, 1, 2, 2, dtype=torch.bool)),
    )

    assert output.shape == target.shape
