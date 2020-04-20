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
from multiprocessing import Process, Queue, cpu_count
from metropolis import MetropolisHastings

def StartMCMC(gen_model, experiment_dir, high_seqs, params):
    # loading in the environment class, used to score the evolutionary hamiltonians

    sampler = MetropolisHastings(gen_model, experiment_dir, x0 = high_seqs,
                        stride=params['stride'], mapper=None, 
                        is_discrete=True,
                        nwalkers=params['nwalkers'], save_trajectory=True, 
                        print_every =params['print_every'])


    sampler.run(params['nsteps'])
    
def main(params):

    start_time = time.time()

    hillclimb_time = 1297.168361934026
    params['stride'] = 1

    date_time = str(datetime.now()).replace(' ', '_').replace(':', '_') # ensures there aren't any issues saving this as a file name. 
    experiment_name = params['exp_base_name']+'_datetime_'+str(date_time)
    experiment_dir = 'hill_experiments/'+experiment_name
    os.mkdir(experiment_dir)
    experiment_dir = experiment_dir+'/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading in EVCouplings model 
    focus_seqs = read_fa('EVCouplingsStuff/DYR_ECOLI_1_b0.5.a2m_trimmed.fa')
    evc_model = CouplingsModel('EVCouplingsStuff/DYR.model')

    # extracting the model parameters used to determine the evolutionary hamiltonian
    h = evc_model.h_i
    J = evc_model.J_ij

    if params['protein_length'] > 0:
        h = h[0:params['protein_length'], :]
        J = J[0:params['protein_length'], 0:params['protein_length'], :,:]

    # processing and plotting the natural sequences: 
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
    gen_model = EVCouplingsGenerator(L, AA, h, J, device, params['is_discrete'], gaussian_cov_noise = 1.0)

    # getting best natural sequences: 
    nat_energies = hamiltonians(oh, J, h)
    high_ind = np.argsort(-nat_energies)
    high_seqs = oh[high_ind][:params['nwalkers']]
    high_seqs = high_seqs.reshape(high_seqs.shape[0], -1)

    assert params['ncores'] != 0, "need to set at least one core!"
    if params['ncores'] == -1:
        params['ncores'] = cpu_count()
    
    # multicore generate new samples
    processes = [Process(target=StartMCMC, args=( gen_model,
                    experiment_dir+'worker_'+str(i)+'_', high_seqs, params )) for i in range(params['ncores'])]

    for p in processes:
        p.start()

    for p in processes:
        # waits for all the processes to have completed before
        # continuing with the code.
        p.join()

    print('all processes are done!, trying to join together all of the files')

    files_to_combine = ['MCMC_trajectories_energies.txt', 
                    'MCMC_trajectories_seqs.txt']
    for f in files_to_combine:
        f_ending = f.split('.')[-1]
        f_start = f.split('.')[0]
        combo_file = experiment_dir+'combined_'+f_start+'.txt'
        with open(combo_file,'w') as write_out:

            for i in range(params['ncores']):
                worker_file = experiment_dir+'worker_'+str(i)+'_'+f

                if f_ending=='pickle':
                    temp = pickle.load(open(worker_file, 'rb'))
                    write_out.write('\n'.join('{} {} {}'.format(tup[0],tup[1], tup[2]) for tup in temp))

                elif f_ending=='txt': 
                    temp = np.loadtxt(worker_file)
                    if f_start.split('_')[-1] == 'seqs':
                        np.savetxt(write_out, temp, fmt='%i')
                    elif f_start.split('_')[-1] == 'energies':
                        np.savetxt(write_out, temp, fmt='%2g')
                else: 
                    raise Exception('File type not identified when trying to combine all worker outputs together!')
            
    print('Total run time in minutes: '+str((time.time()-start_time)/60))

