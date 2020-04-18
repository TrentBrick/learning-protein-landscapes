import time
import pickle 
import numpy as np
from hillclimb import HillClimbing
import os 
from EVCouplingsGen import *
from evcouplings.couplings import CouplingsModel
from EVCouplingsStuff.seq_sele import *

if __name__=='__main__':

    start_time = time.time()

    params = {'protein_length':6, 'is_discrete':True, 
    'gaussian_cov_noise':None, 'nwalkers':64, 'print_every':50,
    'experiment_name':'protein_lengthFull_FullInt_K_steps', 
    'save_trajectory':True}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    date_time = str(datetime.now()).replace(' ', '_').replace(':', '_') # ensures there aren't any issues saving this as a file name. 
    experiment_name = params['experiment_name']+'_datetime_'+str(date_time)
    
    os.mkdir('experiments/'+experiment_name)
    experiment_dir = 'experiments/'+ experiment_name+'/'

    # Loading in EVCouplings model 
    focus_seqs = read_fa('EVCouplingsStuff/DYR_ECOLI_1_b0.5.a2m_trimmed.fa')
    evc_model = CouplingsModel('EVCouplingsStuff/DYR.model')

    # extracting the model parameters used to determine the evolutionary hamiltonian
    h = evc_model.h_i
    J = evc_model.J_ij

    if params['protein_length'] > 0:
        h = h[0:params['protein_length'], :]
        J = J[0:params['protein_length'], 0:params['protein_length'], :,:]
            
    # processing the natural sequences: 
    # first by converting amino acids into integers and also onehots. 
    enc_seqs=[]
    oh = []
    AA=h.shape[1] # number of amino acids
    for seq in focus_seqs['seq']:
        enc_seq = np.asarray(encode_aa(seq, evc_model.alphabet_map))
        if params['protein_length'] > 0: 
            enc_seq = enc_seq[:params['protein_length']]
        enc_seqs.append(enc_seq) 
        oh.append(onehot(enc_seq,AA)) # this could be made much more efficient with tensorflow operations. 
    enc_seqs = np.asarray(enc_seqs)
    oh=np.asarray(oh) # of shape: [batch x L x AA]
    N = oh.shape[0] # batch size
    L = oh.shape[1] # length of the protein
    
    print('number and dimensions of the natural sequences', oh.shape)

    # loading in the environment class, used to score the evolutionary hamiltonians
    gen_model = EVCouplingsGenerator(L, AA, h, J, device, params['is_discrete'], gaussian_cov_noise = params['gaussian_cov_noise'])

    hill_climber = HillClimbing(gen_model, experiment_dir,
        nwalkers=params['nwalkers'], print_every=params['print_every'],
        save_trajectory=params['save_trajectory'])

    local_maxes = hill_climber.run(1000)
    pickle.dump(local_maxes, open(hill_climber.experiment_dir+'local_maxes_and_energies.pickle', 'wb'))

    print('Total run time in minutes: '+str((time.time()-start_time)/60))



    