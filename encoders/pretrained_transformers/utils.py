import torch


def get_sequence_mask(sequence_len):
    """Returns Sequence Mask.
    sequence_len: Tensor of size (B,) with entries indicating length of seq.
    """
    batch_size = sequence_len.size()[0]
    max_len = torch.max(sequence_len)
    return (torch.arange(max_len).expand(batch_size, max_len).cuda()
            < sequence_len.unsqueeze(1))
