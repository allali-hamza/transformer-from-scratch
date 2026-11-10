import torch

from src.model.masks import create_mask, create_src_mask, create_tgt_mask


def test_create_src_mask_hides_padding_tokens():
    source = torch.tensor([[4, 0, 8], [0, 2, 3]])

    mask = create_src_mask(source, pad_token_id=0)

    expected = torch.tensor([[[[True, False, True]]], [[[False, True, True]]]])
    assert torch.equal(mask, expected)


def test_create_tgt_mask_combines_padding_and_causal_masks():
    target = torch.tensor([[5, 0, 8]])

    mask = create_tgt_mask(target, pad_token_id=0)

    expected = torch.tensor(
        [[[[True, False, False], [True, False, False], [True, False, True]]]]
    )
    assert torch.equal(mask, expected)


def test_create_mask_returns_source_and_target_masks():
    source = torch.tensor([[1, 0]])
    target = torch.tensor([[2, 3]])

    source_mask, target_mask = create_mask(source, target, pad_token_id=0)

    assert torch.equal(source_mask, torch.tensor([[[[True, False]]]]))
    assert torch.equal(target_mask, torch.tensor([[[[True, False], [True, True]]]]))
