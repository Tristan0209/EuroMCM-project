import copy
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

from q1_adapter import definitions

definitions(Path(__file__).resolve().parent / 'support/network.py',
            ['GainEnsemble'], globals())
