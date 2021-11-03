# -*- coding: utf-8 -*-
"""
Python script for training the MLP
created on 03.11.2021 by jchburmester

"""

import numpy as np
import itertools

# creating inputs
inputs = np.array(list(itertools.product([False, True], repeat=2)))
