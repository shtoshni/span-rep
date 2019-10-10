"""Different batched non-parametric span representations."""
import torch
from encoder import Encoder
from utils import get_span_mask


def get_diff_repr(encoded_input, start_ids, end_ids):
    """Calculates the difference based span representation: h_j - h_i"""
    batch_size = encoded_input.shape[0]
    span_repr = (encoded_input[torch.arange(batch_size), end_ids, :]
                 - encoded_input[torch.arange(batch_size), start_ids, :])
    return span_repr


def get_diff_sum_repr(encoded_input, start_ids, end_ids):
    """Calculates the difference + sum based span representation: [h_j - h_i; h_j + h_i]"""
    batch_size = encoded_input.shape[0]
    span_repr = torch.cat([
        encoded_input[torch.arange(batch_size), end_ids, :]
        - encoded_input[torch.arange(batch_size), start_ids, :],
        encoded_input[torch.arange(batch_size), end_ids, :]
        + encoded_input[torch.arange(batch_size), start_ids, :]
        ], dim=1)
    return span_repr


def get_avg_repr(encoded_input, start_ids, end_ids):
    span_lengths = (end_ids - start_ids + 1).unsqueeze(1)
    span_masks = get_span_mask(start_ids, end_ids, encoded_input.shape[1])
    span_repr = torch.sum(encoded_input * span_masks, dim=1) / span_lengths.float()
    return span_repr


def get_max_repr(encoded_input, start_ids, end_ids):
    span_masks = get_span_mask(start_ids, end_ids, encoded_input.shape[1])
    # put -inf to irrelevant positions
    tmp_repr = encoded_input * span_masks - 1e10 * (1 - span_masks)
    span_repr = torch.max(tmp_repr, dim=1)[0]
    return span_repr


def get_coherent_repr(encoded_input, start_ids, end_ids):
    batch_size = encoded_input.shape[0]
    p_size = int(encoded_input.shape[2]/4)
    h_start = encoded_input[torch.arange(batch_size), start_ids, :]
    h_end = encoded_input[torch.arange(batch_size), end_ids, :]

    coherence_term = torch.sum(
        h_start[:, 2*p_size:3*p_size] * h_end[:, 3*p_size:], dim=1, keepdim=True)
    return torch.cat(
        [h_start[:, :p_size], h_end[:, p_size:2*p_size], coherence_term], dim=1)


def get_span_repr(encoded_input, start_ids, end_ids, method="avg"):
    """ encoded_input: (B, L_max, H) float tensor.
    start_ids, end_ids: (B, ) long tensor.
    """
    if torch.cuda.is_available():
        start_ids = start_ids.cuda()
        end_ids = end_ids.cuda()
    if method == "avg":
        return get_avg_repr(encoded_input, start_ids, end_ids)
    elif method == "max":
        return get_max_repr(encoded_input, start_ids, end_ids)
    elif method == "diff":
        return get_diff_repr(encoded_input, start_ids, end_ids)
    elif method == "diff_sum":
        return get_diff_sum_repr(encoded_input, start_ids, end_ids)
    elif method == "coherent":
        return get_coherent_repr(encoded_input, start_ids, end_ids)
    else:
        assert ("Method not implemented")


def get_repr_size(hidden_size, method="avg"):
    if method == "avg":
        return hidden_size
    elif method == "max":
        return hidden_size
    elif method == "diff":
        return hidden_size
    elif method == "diff_sum":
        return 2 * hidden_size
    elif method == "coherent":
        return (hidden_size//2 + 1)
    elif method == "attn":
        return hidden_size
    elif method == "coref":
        return 3 * hidden_size
    else:
        assert ("Method not implemented")


if __name__ == '__main__':
    model = Encoder(model='xlnet', model_size='base').cuda()
    tokenized_input, input_lengths = model.tokenize_batch(
        ["What's up", "Greetings, my cat is cute!"]
    )
    encoded_input = model(tokenized_input)

    for method in ["avg", "max", "diff", "diff_sum", "coherent"]:
        start_ids = torch.tensor([0, 0]).long()
        end_ids = input_lengths - 1
        print("Method: %s, Shape: %s" % (
                method, str(
                    get_span_repr(encoded_input, start_ids, end_ids, method=method).shape)
            )
        )
