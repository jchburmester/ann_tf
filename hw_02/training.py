# -*- coding: utf-8 -*-
"""
Python script for training the MLP
created on 03.11.2021 by jchburmester

"""

import numpy as np
import itertools

# creating inputs
inputs = np.array(list(itertools.product([False, True], repeat=2)))

# creating labels for logical gates
# logical 'and'
label_and = np.array([False, False, False, True])

# logical 'or'
label_or = np.array([False, True, True, True])

# logical 'not and'
label_not_and = np.array([True, True, True, False])

# logical 'not or'
label_not_or = np.array([True, False, False, False])

# logical 'xor'
label_xor = np.array([False, True, True, False])
