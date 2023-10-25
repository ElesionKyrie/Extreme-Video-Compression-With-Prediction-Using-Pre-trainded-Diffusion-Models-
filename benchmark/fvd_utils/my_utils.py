
import sys
import os
module_dir = os.path.abspath('fvd_utils')
sys.path.append(module_dir)

from calculate_fvd import *


def calculate_fvd(videos1, videos2, device="cuda"):
    
    # videos [batch_size, timestamps, channel, h, w]
    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    assert videos1.shape == videos2.shape

    i3d = load_i3d_pretrained(device=device)

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    feats1 = get_fvd_feats(videos1, i3d=i3d, device=device)
    feats2 = get_fvd_feats(videos2, i3d=i3d, device=device)

    fvd_results = frechet_distance(feats1, feats2)

    return fvd_results