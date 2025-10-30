# row anchors are pre-defined row positions used to sample lane locations.
# Notes on coordinate conventions used across the codebase:
# - The non-DALI PyTorch dataloader expects anchors in the resized training
#   image coordinate space (height=288, width=800). For example,
#   `culane_row_anchor` below contains row indices in [0..287] (288px space),
#   and `culane_col_anchor` contains column positions in [0..800].
# - The DALI pipeline and evaluation helpers use normalized anchors in [0,1]
#   (e.g. `cfg.row_anchor = np.linspace(0.0, 1.0, num_row)`) which are later
#   multiplied by the original image height/width (590/1640) when producing
#   final coordinates for the CULane evaluation tool.
# Keep these conventions in mind if you convert anchors between spaces.

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]
# culane_row_anchor — sample 18 row positions evenly across the resized
# training height (0..287). We use integers to match the rest of the code.
import numpy as np
culane_row_anchor = list(map(int, np.linspace(0, 287, 18)))
culane_col_anchor = [0.,  20.,  40.,  60.,  80., 100., 120., 140., 160., 180., 200.,
                    220., 240., 260., 280., 300., 320., 340., 360., 380., 400., 420.,
                    440., 460., 480., 500., 520., 540., 560., 580., 600., 620., 640.,
                    660., 680., 700., 720., 740., 760., 780., 800.]
