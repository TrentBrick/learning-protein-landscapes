

import copy
from evcouplings import align
import numpy as np

def single_ham(seq, mat):
    return 1 - ((seq == mat).sum(-1) / len(seq))

def msa_weights(oh, theta=0.8, pseudocount=0):
    #i = pairwise_identity(oh)
    #neighbors = msa_neighbors(oh, theta, pseudocount)
    neighbors = align.alignment.num_cluster_members(oh, theta)
    #(i > theta).sum(0)/2
    w = 1 / (neighbors + pseudocount)
    return(w, neighbors)

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