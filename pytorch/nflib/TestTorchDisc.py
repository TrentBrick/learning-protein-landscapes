from TorchDiscCode import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import torch
import itertools

nh = 24
vocab_size = 150
temperature = 0.1
layer = MADE(vocab_size, [nh, nh, nh], vocab_size*2, num_masks=1, natural_ordering=False)
model = DiscreteAutoregressiveFlow( layer, temperature, vocab_size )

# previously modulo has not been working here. 
mod_test = model.forward(torch.tensor(oh[:64,:,:]).float())