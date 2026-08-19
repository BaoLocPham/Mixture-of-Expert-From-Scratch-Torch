"""Expected outputs for check_rope.py - values only, no method.

Produced by LLM/common.py (`rope_tables`, `apply_rope`) and by four specific
WRONG implementations, so the grader can name the mistake instead of just
saying "wrong". Reading these tells you nothing about how to produce them.

Fixture: d_head = 8, max_seq = 16, base = 10000, and

    q = [0.6, -0.2, 0.9, 0.4, -0.7, 0.1, 0.3, -0.5]   rotated at position m = 3

Regenerate whenever LLM/common.py changes; never edit by hand.
"""

import torch

# --- stage 1: the tables ---------------------------------------------------
COS_ROW1 = torch.tensor([0.540302, 0.995004, 0.99995, 1.0])
SIN_ROW1 = torch.tensor([0.841471, 0.099833, 0.01, 0.001])
COS_LAST = torch.tensor([-0.759688, 0.070737, 0.988771, 0.999888])

# theta = base**(-i/d_h) over i = 0..d_h/2-1  (the factor of 2 dropped)
COS_ROW1_NO2 = torch.tensor([0.540302, 0.950415, 0.995004, 0.9995])
# the ladder reversed: the last pair turning fastest instead of the first
COS_ROW1_FLIPPED = torch.tensor([1.0, 0.99995, 0.995004, 0.540302])

# --- stage 3: the split-half (GPT-NeoX / HuggingFace) convention -----------
ROT_HALF_Q = torch.tensor([0.7, -0.1, -0.3, 0.5, 0.6, -0.2, 0.9, 0.4])

# --- stage 2: the rotation -------------------------------------------------
Q_ROT = torch.tensor([-0.565772, 0.28267, 0.741595, 0.648103, -0.702685, 0.078958, 0.301499, -0.499098])
# rotated the wrong way round (the transpose of E7's matrix)
Q_ROT_CONJ = torch.tensor([-0.62222, 0.113327, 0.978011, 0.116166, -0.696685, 0.120952, 0.298499, -0.500898])
# right rotation, put back with cat instead of interleaved
Q_ROT_CAT = torch.tensor([-0.565772, 0.741595, -0.702685, 0.301499, 0.28267, 0.648103, 0.078958, -0.499098])
# the split-half (GPT-NeoX) pairing, applied to interleaved channels
Q_ROT_HALF = torch.tensor([-0.495212, -0.220619, 0.890596, 0.401498, 0.777667, 0.03643, 0.326861, -0.498798])
