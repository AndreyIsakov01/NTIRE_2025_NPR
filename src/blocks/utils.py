import numpy as np


def create_black_level_mask(image_shape, black_level):
    if not isinstance(black_level, list) or len(black_level) != 4:
        raise ValueError(
            "black_level must be a list of 4 values corresponding to the 2x2 Bayer pattern."
        )

    black_level_mask = np.zeros(image_shape)
    if len(set(black_level)) == 1:
        black_level_mask += black_level[0]
    else:
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0] :: step2, idx[1] :: step2] = black_level[i]
    return black_level_mask.astype(np.float32)
