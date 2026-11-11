import torch

from src.model.transformer import FeedForwardBlock


def test_feed_forward_block_applies_relu_between_linear_layers():
    block = FeedForwardBlock(d_model=2, d_ff=2, dropout=0.0)
    with torch.no_grad():
        block.linear1.weight.copy_(torch.eye(2))
        block.linear1.bias.zero_()
        block.linear2.weight.copy_(torch.eye(2))
        block.linear2.bias.zero_()

    output = block(torch.tensor([[[-3.0, 2.0]]]))

    torch.testing.assert_close(output, torch.tensor([[[0.0, 2.0]]]))
