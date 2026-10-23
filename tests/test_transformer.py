
import pytest
import torch

from src.model import build_transformer


@pytest.fixture
def transformer():
    torch.manual_seed(42)

    model = build_transformer(
        50259, 50259, 96, 96, 64, 2, 4, 0, 128
    )
    model.eval()

    return model


@pytest.fixture
def inputs():
    torch.manual_seed(42)

    source = torch.randint(0, 50259, (2, 7))
    target = torch.randint(0, 50259, (2, 5))

    source_mask = torch.ones(2, 1, 1, 7)
    target_mask = torch.tril(torch.ones(1, 1, 5, 5))

    return source, target, source_mask, target_mask


def test_encoder_output_shape(transformer, inputs):
    source, _, source_mask, _ = inputs

    with torch.no_grad():
        encoded_source = transformer.encode(
            x=source,
            mask=source_mask
        )

    assert encoded_source.shape == (2, 7, 64)


def test_decoder_output_shape(transformer, inputs):
    source, target, source_mask, target_mask = inputs

    with torch.no_grad():
        encoded_source = transformer.encode(
            x=source,
            mask=source_mask
        )

        decoded_source = transformer.decode(
            x=target,
            encoder_output=encoded_source,
            source_mask=source_mask,
            target_mask=target_mask
        )

    assert decoded_source.shape == (2, 5, 64)


def test_projection_output_shape(transformer, inputs):
    source, target, source_mask, target_mask = inputs

    with torch.no_grad():
        encoded_source = transformer.encode(
            x=source,
            mask=source_mask
        )

        decoded_source = transformer.decode(
            x=target,
            encoder_output=encoded_source,
            source_mask=source_mask,
            target_mask=target_mask
        )

        projected_output = transformer.project(
            x=decoded_source
        )

    assert projected_output.shape == (2, 5, 50259)


def test_decoder_causal_mask(transformer, inputs):
    source, target, source_mask, target_mask = inputs

    modified_target = target.clone()
    modified_target[:, -1] = (target[:, -1] + 1) % 50259

    with torch.no_grad():
        encoded_source = transformer.encode(
            x=source,
            mask=source_mask
        )

        original_output = transformer.decode(
            x=target,
            encoder_output=encoded_source,
            source_mask=source_mask,
            target_mask=target_mask
        )

        modified_output = transformer.decode(
            x=modified_target,
            encoder_output=encoded_source,
            source_mask=source_mask,
            target_mask=target_mask
        )

    torch.testing.assert_close(
        original_output[:, :-1, :],
        modified_output[:, :-1, :],
        rtol=1e-5,
        atol=1e-6
    )

    assert not torch.allclose(
        original_output[:, -1, :],
        modified_output[:, -1, :]
    )
