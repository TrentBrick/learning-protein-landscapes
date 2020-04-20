
import time
import pickle 
import numpy as np
from hill_metropolis import HillMetropolis
import os 
from EVCouplingsGen import *
from evcouplings.couplings import CouplingsModel
from EVCouplingsStuff.seq_sele import *
from multiprocessing import Process, Queue, cpu_count

def StartHillMCMC(gen_model, experiment_dir, high_seqs, params, random_seed):
    # loading in the environment class, used to score the evolutionary hamiltonians

    if params['hill_or_MCMC'] == 'hill':

        hill_climber = HillMetropolis(gen_model, experiment_dir, random_seed, params['hill_or_MCMC'],
            nwalkers=params['nwalkers'], print_every=params['print_every'],
            save_trajectory=params['save_trajectory'])

        local_maxes = hill_climber.run(params['nsteps'])
        pickle.dump(local_maxes, open(hill_climber.experiment_dir+'hill_local_maxes_and_energies.pickle', 'wb'))

    elif params['hill_or_MCMC'] == 'mcmc':
        mcmc = HillMetropolis(gen_model, experiment_dir, random_seed, params['hill_or_MCMC'],
                        x0 = high_seqs,
                        stride=params['stride'], mapper=None, 
                        is_discrete=True,
                        nwalkers=params['nwalkers'], save_trajectory=True, 
                        print_every =params['print_every'])


        mcmc.run(params['nsteps'])


def main(params):

    start_time = time.time()

    params['hill_or_MCMC'] = params['hill_or_MCMC'].lower()
    assert params['hill_or_MCMC']=='hill' or params['hill_or_MCMC']=='mcmc', 'did not enter hill or mcmc as the option!'
    params['stride'] = 1

    #params = {'protein_length':6, 'is_discrete':True, 
    #'gaussian_cov_noise':None, 'nwalkers':64, 'print_every':50,
    #'exp_base_name':'protein_lengthFull_FullInt_K_steps', 
    #'save_trajectory':True, 'ncores':10, 'nsteps':100}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    date_time = str(datetime.now()).replace(' ', '_').replace(':', '_') # ensures there aren't any issues saving this as a file name. 
    experiment_name = params['exp_base_name']+'_datetime_'+str(date_time)
    experiment_dir = 'hill_experiments/'+experiment_name
    os.mkdir(experiment_dir)
    experiment_dir = experiment_dir+'/'

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

    assert params['ncores'] != 0, "need to set at least one core!"
    if params['ncores'] == -1:
        params['ncores'] = cpu_count()

    print('running the hill climbers! Using: ', params['ncores'], 'out of the', cpu_count(), 'available cores.')

    gen_model = EVCouplingsGenerator(L, AA, h, J, device, params['is_discrete'], gaussian_cov_noise = params['gaussian_cov_noise'])
    
    # getting best natural sequences needed for MCMC: 
    nat_energies = hamiltonians(oh, J, h)
    high_ind = np.argsort(-nat_energies)
    high_seqs = oh[high_ind][:params['nwalkers']]
    high_seqs = high_seqs.reshape(high_seqs.shape[0], -1)
    
    # starting up the multicore!

    assert params['ncores'] != 0, "need to set at least one core!"
    if params['ncores'] == -1:
        params['ncores'] = cpu_count()

    # multicore generate new samples
    processes = [Process(target=StartHillMCMC, args=( gen_model,
                    experiment_dir+'worker_'+str(i)+'_',  high_seqs, params, np.random.randint(0,10000,1)[0] )) for i in range(params['ncores'])]

    for p in processes:
        p.start()

    for p in processes:
        # waits for all the processes to have completed before
        # continuing with the code.
        p.join()

    print('all processes are done!, trying to join together all of the files')

    if params['hill_or_MCMC'] == 'hill':
        files_to_combine = ['hill_trajectories_energies.txt', 
                        'hill_trajectories_seqs.txt',
                        'hill_local_maxes_and_energies.pickle']
    elif params['hill_or_MCMC'] == 'mcmc':
        files_to_combine = ['mcmc_trajectories_energies.txt', 
                    'mcmc_trajectories_seqs.txt']

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




    