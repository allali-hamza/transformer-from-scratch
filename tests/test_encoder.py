import torch
from torch import nn

from src.model.transformer import Encoder, EncoderBlock, FeedForwardBlock, MultiHeadAttentionBlock


def test_encoder_runs_its_blocks_and_preserves_shape():
    encoder = Encoder(nn.ModuleList([
        EncoderBlock(
            MultiHeadAttentionBlock(4, 2, 0.0),
            FeedForwardBlock(4, 8, 0.0),
            dropout=0.0,
        )
    ]))
    x = torch.randn(2, 3, 4)

    output = encoder(x, torch.ones(2, 1, 1, 3, dtype=torch.bool))

    assert output.shape == x.shape
