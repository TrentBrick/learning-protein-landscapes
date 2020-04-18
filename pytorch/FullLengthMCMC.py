''' Running full protein length long MCMC simulation! '''

import numpy as np
import time
import pickle
import os

#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot


from EVCouplingsGen import *
from evcouplings.couplings import CouplingsModel
from EVCouplingsStuff.seq_sele import *

from metropolis import MetropolisHastings

if __name__=='__main__':

    hillclimb_time = 1297.168361934026
    nsteps = 85000 #30000000 # this will be 150 million. samples. 
    stride= 1
    nwalkers=64
    print_every = 50

    date_time = str(datetime.now()).replace(' ', '_').replace(':', '_') # ensures there aren't any issues saving this as a file name. 
    experiment_name = 'MCMC_lengthFull_sametime_seqInts_bestSeqStarts'+'_datetime_'+str(date_time)

    os.mkdir('experiments/'+experiment_name)
    experiment_dir = 'experiments/'+ experiment_name+'/'

    start_time = time.time()

    protein_length =154 # FULL LENGTH SEQUENCE. 
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


    # getting best natural sequences: 
    nat_energies = hamiltonians(oh, J, h)
    high_ind = np.argsort(-nat_energies)
    high_seqs = oh[high_ind][:nwalkers]
    high_seqs = high_seqs.reshape(high_seqs.shape[0], -1)

    sampler = MetropolisHastings(gen_model, experiment_dir, x0 = high_seqs,
                        stride=stride, mapper=None, 
                        is_discrete=True, AA_num=AA, 
                        nwalkers=nwalkers, save_trajectory=True, 
                        print_every =print_every)

    #while (time.time() - start_time)/60 < hillclimb_time:

    sampler.run(nsteps)

    print('======== total time to run: ', time.time() - start_time)