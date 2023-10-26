
"""
Evaluate an end-to-end compression model on an image dataset.
"""

import time

import torch.nn.functional as F

import torch

import torch.nn as nn

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)



@torch.no_grad()
def inference(model, x, patch):
    x = x.unsqueeze(0)
    h, w = x.size(2), x.size(3)
    p = patch  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = 0
    padding_right = new_w - w - padding_left
    padding_top = 0
    padding_bottom = new_h - h - padding_top
    pad = nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)
    x_padded = pad(x)

    _, _, height, width = x_padded.size()
    start = time.time()
    out_enc = model.compress(x_padded)

    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = torch.nn.functional.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )



    num_pixels = x.size(0) * x.size(2) * x.size(3)

    bpp = 0
    for s in out_enc["strings"]:
        for j in s:
            if isinstance(j, list):
                for i in j:
                    if isinstance(i, list):
                        for k in i:
                            bpp += len(k)
                    else:
                        bpp += len(i)
            else:
                bpp += len(j)
                """
                这里得到的是图像的字节大小
                """

    bits = bpp * 8
    bpp *= 8.0 / num_pixels

    z_bpp = len(out_enc["strings"][1][0])* 8.0 / num_pixels
    y_bpp = bpp - z_bpp



    return out_dec["x_hat"], bits
