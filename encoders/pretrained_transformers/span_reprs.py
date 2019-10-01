"""Different non-parametric span representations."""
import torch


def get_diff_repr(encoded_input, start_idx, end_idx):
    """Does the difference based span representation: h_j - h_i"""
    span_repr = (encoded_input[:, end_idx, :]
                 - encoded_input[:, start_idx, :])
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
