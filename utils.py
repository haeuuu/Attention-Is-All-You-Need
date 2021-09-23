import torch

def generate_lm_mask(inputs, fill = 1.):
    """Generate mask for language modeling

    Returns
    -------
    mask : (batch_size, seq_len, seq_len)

    Examples
    ---------
    inputs = torch.LongTensor([[5,2,3],
                                [9,8,3]]) ; token indices

    generate_lm_mask(inputs, fill = 1)
    tensor([[[0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]],

            [[0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]]])
    """
    batch_size, seq_len = inputs.shape[0], inputs.shape[1]

    mask = torch.full((seq_len, seq_len), fill).triu()
    mask = mask.fill_diagonal_(0.)
    mask = mask.repeat(batch_size, 1, 1)

    return mask.bool().to(inputs.device)