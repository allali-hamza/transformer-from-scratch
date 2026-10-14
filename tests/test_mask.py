import torch
from src.mask import create_mask


def test_create_mask():

    pad_id = 0

    source = torch.tensor([
        [11, 12,  2,  0, 0],
        [21, 22, 23, 24, 2]
    ])

    decoder_input = torch.tensor([
        [1, 31, 32,  0],
        [1, 41, 42, 43]
    ])

    src_mask, tgt_mask = create_mask(
        source,
        decoder_input,
        pad_id
    )

    assert src_mask.shape == (2, 1, 1, 5)
    assert tgt_mask.shape == (2, 1, 4, 4)

    assert src_mask.dtype == torch.bool
    assert tgt_mask.dtype == torch.bool

    expected_src = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1]
    ], dtype=torch.bool).reshape(2, 1, 1, 5)

    assert torch.equal(src_mask, expected_src)

    expected_tgt = torch.tensor([
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0]
        ],
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1]
        ]
    ], dtype=torch.bool).reshape(2, 1, 4, 4)

    assert torch.equal(tgt_mask, expected_tgt)

