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

def generate_pad_mask(inputs, pad_idx):
    """Generate mask for self attention
    
    Returns
    -------
    mask : (batch_size, seq_len, 1)

    Examples
    --------
    inputs = torch.LongTensor([[5,2,3],
                                [9,1,1]]) ; token indices

    generate_pad_mask(inputs, pad_idx = 1)
    tensor([[[0, 0, 0]],
            [[0, 1, 1]]])
    """
    mask = (inputs == pad_idx).unsqueeze(1)

    return mask.bool().to(inputs.device)

def generate_masked_attn_mask(targets, pad_idx, fill = 1):
    """Generate mask for masked self attention in Transformer Decoder

    Returns
    -------
    mask : (batch_size, seq_len, seq_len)

    Examples
    --------
    inputs = torch.LongTensor([[5,2,1],
                                [9,1,1]]) ; token indices

    generate_masked_attn_mask(inputs, pad_idx = 1, fill = 1)
    tensor([[[0, 1, 1],
            [0, 0, 1],
            [0, 0, 1]],

            [[0, 1, 1],
            [0, 1, 1],
            [0, 1, 1]]])
    """
    pad_mask = generate_pad_mask(targets, pad_idx)
    lm_mask = generate_lm_mask(targets, fill)

    mask = torch.logical_or(lm_mask, pad_mask)

    return mask.bool()


if __name__ == '__main__':

    inputs = torch.LongTensor([[5,2,3],[9,1,1]])

    print('mask for language model\n',generate_lm_mask(inputs, fill = 1))
    print('mask for self attention\n', generate_pad_mask(inputs, pad_idx = 1))
    print('mask for masked self attention\n', generate_masked_attn_mask(inputs, pad_idx = 1, fill = 1))