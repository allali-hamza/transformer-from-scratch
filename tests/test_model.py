from src.model import build_transformer
import torch


transformer = build_transformer(
    50259, 50259, 96, 96, 64, 2, 4, 0, 128
)

source = torch.randint(
    low=0,
    high=50259,
    size=[2, 7]
)



causel_mask = torch.tril(torch.ones([5, 5]))


# print(causel_mask)

# print(source)
# print(target)

mask_source = torch.ones(2, 1, 1, 7)


print(source.shape)
print(mask_source.shape)

print(source)
print(mask_source)

encoded_source = transformer.encode(
    x=source,
    mask=mask_source
)

print(encoded_source.shape)


print("*" * 50)

target = torch.randint(0, 50259, [2, 5])

print(target.shape)

decoded_source = transformer.decode(
    x=target,
    encoder_output=encoded_source,
    source_mask=mask_source,
    target_mask=causel_mask
)

print(decoded_source.shape)


projected_output = transformer.project(x=decoded_source)

print(projected_output.shape)


transformer.eval()

copy_of_target = torch.clone(target)
copy_of_target[:, -1] = 1

print(target)
print(copy_of_target)

print("copy of target", copy_of_target[0][-1])

other_decoded_source = transformer.decode(
    x=copy_of_target,
    encoder_output=encoded_source,
    source_mask=mask_source,
    target_mask=causel_mask
)

print("*" * 100)

print(decoded_source.shape)
print(other_decoded_source.shape)

print(decoded_source[0][0][:10])
print(other_decoded_source[0][0][:10])

print(decoded_source[0][1][:10])
print(other_decoded_source[0][1][:10])

print(decoded_source[0][4][:10])
print(other_decoded_source[0][4][:10])

