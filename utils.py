import torch

def generate_lm_mask(seq_len, fill = 1.):
    """Generate mask for language modeling
    Examples
    ---------
    generate_mask(max_len = 4, fill = 1.)
    tensor([[0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]])
    """
    mask = torch.full((seq_len, seq_len), fill).triu()
    mask = mask.fill_diagonal_(0.)

    return mask