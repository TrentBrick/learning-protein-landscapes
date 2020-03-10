#Author: Nathan Rollins
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
_PI = torch.Tensor([math.pi])

#############################################
# 0. tools for creating gaussian mixtures
def create_gauss(point, cov):
    '''returns parameters according to specified Gaussian.
    Dimension of the gaussian is the number of AA features used.'''
    if len(cov.shape) == 1:
        cov = torch.diag(cov) # no covariation
    n = point.shape[-1] # this is the number of AA features being used, the dimension of the guassian
    Z = ((_PI*2)**(-n/2))*(torch.det(cov)**(-0.5)) # everything before the exponent, normalizing constant
    return point, torch.inverse(cov), Z

def create_gmix(points, covs, props):
    '''takes centers [g x n] & covs [g x n x n] & proportions [g],
    yields centers [g x n] & sigmas [g x n x n] & normalizing terms [g]'''
    g, n = points.shape
    centers, sigs, Zs = torch.zeros((g,n)), torch.zeros((g, n, n)), torch.zeros(g)
    for i in range(len(points)):
        centers[i, :], sigs[i, :, :], Zs[i] = create_gauss(points[i,:], covs[i,:,:])
    Zs = Zs*props # weighting the constants by the proportion. 
    centers = center_and_scale(centers)
    return centers, sigs, Zs

#############################################
# 1. create normalized amino acid properties
def center_and_scale(x):
    '''adjusts and scales values to [-1,1]'''
    #TODO: try and have a CDF for the different properties rather than a norm scaling. 
    # but can I a
    x_n = x - x.min()
    x_n = x_n / x_n.max()
    x_n = x_n*2 - 1
    return x_n

# 2. create feature encoder and decoder
def aa_to_seq_encoder(aa_prop, features, aa_order='ACDEFGHIKLNMPQRSTVWY'):
    '''the decoder works by creating a dictionary that products w one-hots, given seq [L x a]'''
    aa_features = aa_prop.set_index('aa').loc[list(aa_order), features].values
    # normalizing. Should write additional functions to normalize by!
    for c in range(aa_features.shape[1]):
        aa_features[:,c] = center_and_scale(aa_features[:,c])
    feat_map = torch.Tensor(aa_features)
    print('shape of feature map', feat_map.shape)
    return lambda seqs: (feat_map.T@seqs.transpose(1,2)).permute(0,2,1)

def decode_feats_to_seq(x, Cs, sigs, Zs):
    ''' 
    Returns: [batch_size x sequence length x pdf over all AAs]
    x is a list of continous values we want to map to a log pdf over the AAs. 
    log(p_g) at list of points [L x n], given gaussian mixture
    w/ centers [g x n] & covs [g x n x n] & proportions (normalizing constants) [1 x g]'''
    #print(x.shape, Cs.shape)
    d = (x.unsqueeze(2) - Cs.unsqueeze(0).unsqueeze(0))# broadcast [b x L x _ x n] vs [_ x _ x g x n] 
    dE = torch.einsum('blgn,gnm->blgm', d, sigs)
    dEd = torch.einsum('blgn,blgn->blg', d, dE)
    logP = torch.log(Zs) - 0.5*dEd
    return torch.exp( logP - torch.logsumexp(logP, dim=2).unsqueeze(2) ) # normalizes the probabilities of each. 

#############################################   
# Wrap it all together -- initialize this during model initialization- not inside 'forward'
def initialize_continous_aa(AA_FILE, seqlen, noise=0.1, aa_order='ACDEFGHIKLNMPQRSTVWY'):
    '''creates encoder and decoder functions'''
    aaprop = pd.read_csv(AA_FILE)
    feats = ['Hydrophobicity', 'Mass', 'pI']
    n = len(feats)
    encoder = aa_to_seq_encoder(aaprop, feats, aa_order)
    aa_feat_matrix = encoder(torch.eye(20).unsqueeze(0)).squeeze()
    centers, sigs, Zs = create_gmix(
        aa_feat_matrix,
        covs = torch.Tensor([noise*np.eye(n)]*20),
        props=torch.ones(20)/20 )
    #Cs = torch.stack([centers]*seqlen, dim=0) # [L x g x n] # why are we repeating the centers here??? 
    decoder = lambda x: decode_feats_to_seq(x, centers, sigs, Zs)
    return encoder, decoder, len(feats)

#############################################
if __name__ == '__main__':
    # test it out!
    fig, ax = plt.subplots(2, 4)
    fig.set_size_inches(15, 7)
    ax = ax.flatten()
    s = torch.tensor( [np.eye(20)] *5).float() #torch.Tensor(np.eye(20)).unsqueeze(0)
    print('seqs input', s.shape)
    ax[0].imshow(s[2], cmap='Blues')
    for i, noise in enumerate([100, 50, 5, 1, 0.1, 0.01, 0.001][::-1]):
        encode, decode = initialize_aa('../pytorch/EVCouplingsStuff/amino_acid_properties_full.csv',
            seqlen=20, noise=noise, aa_order='WFYPMILVAGCSTQNDEHRK')
        e_s = encode(s)
        print('encoded shape', e_s.shape)
        res = decode(e_s).exp()[2]
        print(res.shape)
        #print(res)
        ax[i+1].imshow(res, cmap='Blues')
        ax[i+1].set_title('gaussian feature noise = '+str(noise))
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle('amino acid decoding probabilities (when centered on each amino acid)', fontsize=20)