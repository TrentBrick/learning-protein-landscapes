''' Running full protein length long MCMC simulation! '''

import numpy as np
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch
import datetime

import time
import pickle
import os

#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot

import seaborn as sns
import matplotlib.pylab as plt
from scipy.special import softmax
import json

from EVCouplingsGen import *
from evcouplings.couplings import CouplingsModel
from EVCouplingsStuff.seq_sele import *

from metropolis import MetropolisHastings
from utils import *

from nflib.MADE import *
from nflib.flows import *
from nflib.spline_flows import NSF_AR, NSF_CL
import itertools
import time
import pickle

start_time = time.time()

protein_length =0 # FULL LENGTH SEQUENCE. 
is_discrete = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading in EVCouplings model 
focus_seqs = read_fa('EVCouplingsStuff/DYR_ECOLI_1_b0.5.a2m_trimmed.fa')
evc_model = CouplingsModel('EVCouplingsStuff/DYR.model')

# extracting the model parameters used to determine the evolutionary hamiltonian
h = evc_model.h_i
J = evc_model.J_ij

if protein_length > 0:
    h = h[0:protein_length, :]
    J = J[0:protein_length, 0:protein_length, :,:]

# processing and plotting the natural sequences: 
# first by converting amino acids into integers and also onehots. 
enc_seqs=[]
oh = []
AA=h.shape[1] # number of amino acids
for seq in focus_seqs['seq']:
    enc_seq = np.asarray(encode_aa(seq, evc_model.alphabet_map))
    if protein_length > 0: 
        enc_seq = enc_seq[:protein_length]
    enc_seqs.append(enc_seq) 
    oh.append(onehot(enc_seq,AA)) # this could be made much more efficient with tensorflow operations. 
enc_seqs = np.asarray(enc_seqs)
oh=np.asarray(oh) # of shape: [batch x L x AA]
N = oh.shape[0] # batch size
L = oh.shape[1] # length of the protein

print('number and dimensions of the natural sequences', oh.shape)

# loading in the environment class, used to score the evolutionary hamiltonians
gen_model = EVCouplingsGenerator(L, AA, h, J, device, is_discrete, gaussian_cov_noise = 1.0)


nsteps = 30000000 # this will be 150 million. samples. 
stride= 10
sampler = MetropolisHastings(gen_model, noise=5.0, 
                     stride=stride, mapper=None, 
                     is_discrete=True, AA_num=AA)
#mapper=HardMaxMapper() but now I have discrete actions so dont need. 
sample_x = sampler.run(nsteps)

# save the output! 
pickle.dump(sample_x, open('Full_Len_MCMC.pickle', rb))

print('======== total time to run: ', time.time() - start_time)