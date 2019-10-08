"""Different non-parametric span representations."""
import torch
from encoder import Encoder


def get_diff_repr(encoded_input, start_idx, end_idx):
    """Calculates the difference based span representation: h_j - h_i"""
    span_repr = (encoded_input[:, end_idx, :]
                 - encoded_input[:, start_idx, :])
    return span_repr


def get_diff_sum_repr(encoded_input, start_idx, end_idx):
    """Calculates the difference + sum based span representation: [h_j - h_i; h_j + h_i]"""
    span_repr = torch.cat([encoded_input[:, end_idx, :] - encoded_input[:, start_idx, :],
                           encoded_input[:, end_idx, :] + encoded_input[:, start_idx, :]], dim=1)
    return span_repr


def get_avg_repr(encoded_input, start_idx, end_idx):
    span_repr = 0
    span_length = (end_idx - start_idx + 1)
    assert(span_length > 0)
    span_repr = torch.sum(
        encoded_input[:, start_idx:(end_idx + 1), :], dim=1)/span_length
    return span_repr


def get_max_repr(encoded_input, start_idx, end_idx):
    return torch.max(encoded_input[:, start_idx:(end_idx + 1), :], dim=1)[0]


def get_coherent_repr(encoded_input, start_idx, end_idx):
    p_size = int(encoded_input.shape[2]/4)
    h_start = encoded_input[:, start_idx, :]
    h_end = encoded_input[:, end_idx, :]

    coherence_term = torch.sum(
        h_start[:, 2*p_size:3*p_size] * h_end[:, 3*p_size:], dim=1, keepdim=True)
    return torch.cat(
        [h_start[:, :p_size], h_end[:, p_size:2*p_size], coherence_term], dim=1)


def get_span_repr(encoded_input, start_idx, end_idx, method="avg"):
    if method == "avg":
        return get_avg_repr(encoded_input, start_idx, end_idx)
    elif method == "max":
        return get_max_repr(encoded_input, start_idx, end_idx)
    elif method == "diff":
        return get_diff_repr(encoded_input, start_idx, end_idx)
    elif method == "diff_sum":
        return get_diff_sum_repr(encoded_input, start_idx, end_idx)
    elif method == "coherent":
        return get_coherent_repr(encoded_input, start_idx, end_idx)
    else:
        assert ("Method not implemented")


if __name__ == '__main__':
    model = Encoder(model='xlnet', model_size='base').cuda()
    tokenized_input = model.tokenize_sentence("What's up")
    encoded_input = model(tokenized_input)

    for method in ["avg", "max", "diff", "diff_sum", "coherent"]:
        print("Method: %s, Shape: %s" % (
            method, str(get_span_repr(encoded_input, 0, tokenized_input.shape[1] - 1, method=method).shape)))
