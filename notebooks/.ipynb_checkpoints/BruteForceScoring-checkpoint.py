#!/usr/bin/env python
# coding: utf-8

# In[4]:


cd ../pytorch/


# In[5]:


protein_length =3
calc_Neff = True


# ## Loading in model and setting protein length

# In[6]:


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


# In[7]:



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


# In[8]:


nat_energies = hamiltonians(oh, J, h)
plt.figure()

print('Plotting a hist of all the natural sequences energies:')
plt.hist(nat_energies, bins=100)
plt.show()


# ## Brute Force Sequence Generation!!!

# In[85]:


# getting all possible mutations, saving them as flattened onehots. 
import itertools

def rapid_oh(seq):
    z = np.zeros((len(seq),20))
    z[np.arange(len(seq)), seq] = 1 
    return z.reshape(-1)

all_muts = np.zeros((AA**protein_length, protein_length*AA), dtype=np.uint8)
counter=0
for num in itertools.product(list(range(AA)), repeat=protein_length):
    all_muts[counter,:] = rapid_oh(np.array(num))
    counter+=1
    if counter % 10000000==0:
        print(counter)


# In[86]:


print('number of sequences added', counter)


# In[87]:


'''import pickle
import gzip
pickle.dump(all_muts, gzip.open('all_len6_muts.pickle.gz', 'wb'), protocol=4)'''


# In[90]:


import time 
start_time = time.time()
batch_size = 4000000 # 14 seconds and 4 Gb for 4 million sequences. 
start = np.arange(0, len(all_muts), batch_size)
end = start+batch_size
all_mut_energies = np.zeros((all_muts.shape[0]))
for s, e in zip(start, end):
    all_mut_energies[s:e] = hamiltonians(all_muts[s:e,:], J, h) 
    
print('number of seconds passed', time.time() - start_time)


# ## Getting brute force metrics:

# In[160]:


import copy

def single_ham(seq, mat):
    return 1 - ((seq == mat).sum(-1) / len(seq))

def score_diversity_metric(seqs, energies, theta, E_min, calc_Neff=False, color_clusts=True):
    # if color_clusts it slows things down, needs to do theta for whole set each time. 
    if calc_Neff:
        gen_w, gen_neighbors = msa_weights(seqs, theta=theta, pseudocount=0)
        Neff = gen_w.sum()
    else: 
        Neff = 0.0
    
    sort_ind = np.argsort(-energies ) # sort descending
    e_sorted = energies[sort_ind]
    x_sorted = seqs[sort_ind]
    
    unique_peaks = []
    unique_seqs = []
    if color_clusts:
        seq_clusters = np.zeros(seqs.shape[0])
    else: 
        seq_clusters = None
    x_greedy = copy.copy(x_sorted)
    e_greedy = copy.copy(e_sorted)
    while e_greedy[0] >= E_min: # way this works is that we remove everything nearby and keep taking off the top of the remaining 
        # elements from the sorted list. 
        #print('seqs in egreedy', e_greedy.shape[0])
        unique_peaks.append(e_greedy[0])
        unique_seqs.append(x_greedy[0])
        
        if color_clusts: 
            hams = single_ham(x_greedy[0], seqs) # checks against everything
            seq_clusters[hams<theta] = len(unique_peaks) # unique id for everything removed here
        
        hams = single_ham(x_greedy[0], x_greedy) #NB this will eliminate itself.
        theta_mask = hams>=theta
        x_greedy = x_greedy[theta_mask] # keeps only the sequences that are far away.  
        e_greedy = e_greedy[theta_mask]
        
        if len(e_greedy) == 0: # in case everything is removed. 
            break
        
    return Neff, unique_seqs, unique_peaks, seq_clusters


# In[161]:


E_min = np.min(nat_energies)


# In[162]:


# masking out what is below score threshold to speed up all later processing
print('before masking',all_mut_energies.shape )

e_mask = all_mut_energies >= E_min
all_muts = all_muts[e_mask]
all_mut_energies = all_mut_energies[e_mask]

# need to convert all muts to argmax first
argmax_all_muts = all_muts.reshape(all_muts.shape[0], -1 , AA).argmax(-1).reshape(all_muts.shape[0], -1)
print(argmax_all_muts.shape)

print('after masking',all_mut_energies.shape )


# In[163]:


# brute forced metrics calc
start_time = time.time()

for theta in [0.5, 0.9, 0.8]:

    Neff, unique_seqs, unique_peaks, color_clusts = score_diversity_metric(argmax_all_muts, 
                                                             all_mut_energies, theta, 
                                                             E_min, calc_Neff=calc_Neff)

    print('theta value:', theta)
    print('Neff:', Neff, 'unique peak num', len(unique_peaks))
    print('unique peak sums', sum(unique_peaks))
    print('================')
    print('================')
print('number of seconds passed', time.time() - start_time)


# ## Trying UMAP plot

# In[151]:


# onehot the peak seqs
unique_seqs_oh = onehot(np.asarray(unique_seqs), AA)
unique_seqs_oh= unique_seqs_oh.reshape((unique_seqs_oh.shape[0], -1))
unique_seqs_oh.shape


# In[152]:


to_embed = np.vstack([all_muts, unique_seqs_oh])


# In[153]:


import umap
import seaborn as sns
# sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
start_time = time.time()
reducer = umap.UMAP()
embedding = reducer.fit_transform(to_embed)
print('number of seconds passed', time.time() - start_time)


# In[174]:


cmap = matplotlib.cm.get_cmap(name='viridis')


# In[185]:


num_peaks = len(unique_seqs)
plt.figure(figsize=(12,10))
plt.scatter(embedding[0:-num_peaks, 0], embedding[0:-num_peaks, 1], s=1, alpha=0.8, c=color_clusts, label='brute seqs')
for i in range(num_peaks):
    plt.scatter(embedding[-num_peaks+i, 0], embedding[-num_peaks+i, 1], c=cmap( (i+1)/num_peaks), s=80, alpha=1.0, marker='s', label='cluster '+str(i+1))
plt.colorbar()
plt.legend()
plt.gcf().savefig('BruteForceUMAPwPeaks.png', dpi=250)


# ## Need to make a UMAP with the natural sequences too

# In[111]:


to_embed = np.vstack([all_muts, unique_seqs_oh])
to_embed = np.vstack([to_embed, oh.reshape(oh.shape[0], -1)])
end_of_brute = all_muts.shape[0]
end_of_peaks = end_of_brute+len(unique_seqs)
to_embed.shape


# In[112]:


start_time = time.time()
reducer = umap.UMAP()
embedding = reducer.fit_transform(to_embed)
print('number of seconds passed', time.time() - start_time)


# In[116]:


plt.figure(figsize=(10,10))
plt.scatter(embedding[end_of_peaks:, 0], embedding[end_of_peaks:, 1], color='green', s=1, alpha=0.3, label='nat seqs')
plt.scatter(embedding[0:end_of_brute, 0], embedding[0:end_of_brute, 1], color='blue', s=1, alpha=0.5, label='brute seqs above natural min')
plt.scatter(embedding[end_of_brute:end_of_peaks, 0], embedding[end_of_brute:end_of_peaks, 1], color='orange', s=30, alpha=1.0, label='peaks')
plt.legend()
plt.gcf().savefig('BruteForceUMAPwPeaks_n_NatSeqs.png', dpi=250)


# In[ ]:




