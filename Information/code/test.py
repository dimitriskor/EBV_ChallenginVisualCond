from utils.Events import Events
from utils.Frames import Frames
import numpy as np
import utils.entropy as entropy
import matplotlib.pyplot as plt
from utils import utils

optim_distr_mean, optim_distr_std = utils.optimal_distance('DSEC')
# print(optim_distr_mean)
utils.plot_distr_distance('DSEC', utils.optimal_distance('DSEC'))
