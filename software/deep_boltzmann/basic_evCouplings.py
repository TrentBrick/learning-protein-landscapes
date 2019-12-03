import numpy as np
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import keras
import tensorflow as tf
import datetime

import time
import pickle

#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot

from PlottingFunctions import *

from networks.invertible import create_NICERNet, create_RealNVPNet, invnet
from sampling import GaussianPriorMCMC
from networks.plot import test_xz_projection
from util import count_transitions
from sampling.analysis import free_energy_bootstrap, mean_finite, std_finite
from sampling import MetropolisGauss

import seaborn as sns
import matplotlib.pylab as plt
from scipy.special import softmax

#from EVCouplingsGen import *
from evc import *

from evcouplings.couplings import CouplingsModel
from EVCouplingsStuff.seq_sele import *

import os
cwd = os.getcwd()
print('current directory', cwd)


def main( epochsML = 200, epochsKL = 200,
lr = 0.001, batchsize_ML = 128, batchsize_KL = 1000,
temperature = 1.0, explore = 1.0, latent_std=1.0, 
KL_weight=1.0, ML_weight=0.1, model_architecture = 'NNNNS', nl_activation ='tanh', 
nl_activation_scale = 'tanh', verbose=True, random_seed=27, 
experiment_base_name='didnt_set_exp_name', 
save_partway_inter=None, KL_only=False, dequantize=True):

    start_time = time.time()

    # taking a percentage of the total KL epochs. 
    if save_partway_inter is not None: 
        save_partway_inter = int(save_partway_inter*epochsKL)

    # Experiment save name!
    tf.random.set_random_seed(random_seed)
    np.random.seed(random_seed)
    date_time = str(datetime.now()).replace(' ', '_')
    # used to save the policy and its outputs. 
    experiment_name = experiment_base_name+"rand_seed-%s_ML_epochs-%s_KL_epochs-%s_learning_rate-%s_activation-%s_model_architecture-%s_ML_weight-%s_KL_weight-%s_explore%s_temperature-%s_s_time-%s" % (
        random_seed, epochsML, epochsKL, 
        lr, nl_activation, model_architecture, ML_weight, KL_weight, 
        explore, temperature, date_time )

    os.mkdir('experiments/'+experiment_name)
    experiment_dir = 'experiments/'+ experiment_name+'/'

    ######
    focus_seqs = read_fa('EVCouplingsStuff/DYR_ECOLI_1_b0.5.a2m_trimmed.fa')
    evc_model = CouplingsModel('EVCouplingsStuff/DYR.model')
    #scores = evc_model.hamiltonians(list(focus_seqs['seq']))
            
    enc_seqs=[]
    for seq in focus_seqs['seq']:
        enc_seqs.append(encode_aa(seq, evc_model.alphabet_map)) 

    enc_seqs = np.asarray(enc_seqs)
    #encode_aa(np.char.upper(ali.matrix[0, :]), a2n)

    oh = []
    N=20 # none of these focus have gaps, else should be 21.
    seq_len = 2 
    print('enc seqs before short', enc_seqs.shape)
    enc_seqs = enc_seqs[: , :seq_len]
    print('enc seqs after short', enc_seqs.shape)
    target_seq = enc_seqs[0]
    AA_num=N
    for seq in enc_seqs:
        oh.append(onehot(seq,N))
    oh=np.asarray(oh)

    print('the size of oh', oh.shape)

    print('calculating weights and identities')
    N = oh.shape[0]
    L = oh.shape[1]
    AA = oh.shape[2]
    #w, neighbors = msa_weights(enc_seqs, theta=0.8, pseudocount=0)
    print(' shape of the one hots is', oh.shape)

    t_oh = oh[0]
    #t_oh_flat =t_oh.flatten().reshape(-1,1)
    t_seq_aa = enc_seqs[0,:]

    h = evc_model.h_i
    J = evc_model.J_ij

    print('before shortening', h.shape, J.shape)

    h = h[0:seq_len, :]
    J = J[0:seq_len, 0:seq_len, :,:]

    print('after shortening', h.shape, J.shape)

    plt.figure()
    print('Plotting a hist of all the natural sequences energies:')
    plt.hist(hamiltonians(enc_seqs, J, h), bins=100) # used to be oh
    #plt.show()
    plt.gcf().savefig(experiment_dir+'HistofNatSeqs.png', dpi=250)
    
    gen_model = EVCouplingsGenerator(L, AA, h, J)

    plt.figure()
    plot_potential(AA_num, target_seq, gen_model.energy, orientation='horizontal', pos1=0, pos2=1)
    #plt.show()
    plt.gcf().savefig(experiment_dir+'EnergyPotentialPlot.png', dpi=250)
    
    '''
    # simulation data
    nsteps = 1000
    # starting positions
    I = np.eye(AA_num) 

    def make_rand_starter():
        rand_starter = []
        for i in range(L):
            rand_starter.append( I[np.random.randint(0,20,1),:] )
        rand_starter = np.asarray(rand_starter).flatten().reshape(1,-1)
        return rand_starter
        
    x0_left = make_rand_starter()

    x0_right = make_rand_starter()

    sampler = MetropolisGauss(gen_model, x0_left, noise=5, 
                            stride=5, mapper=None, is_discrete=True, AA_num=AA_num)
    #mapper=HardMaxMapper() but now I have discrete actions so dont need. 
    sampler.run(nsteps)
    traj_left = sampler.traj.copy()

    sampler.reset(x0_left)
    sampler.run(nsteps)
    traj_left_val = sampler.traj.copy()

    sampler.reset(x0_right)
    sampler.run(nsteps)
    traj_right = sampler.traj.copy()

    sampler.reset(x0_right)
    sampler.run(nsteps)
    traj_right_val = sampler.traj.copy()

    # left is blue
    plot_mcmc(traj_left, traj_right, AA_num, pos=0) # pos is for x0 or x1
    #plt.show()
    plot_mcmc(traj_left, traj_right, AA_num, pos=1)
    #plt.show()

    # because of the discreet space, it is too hard to move? 
    # reward for the blue line,
    # the energy states present in over time
    plt.plot(np.arange(traj_left.shape[0]), gen_model.energy(traj_left), color='blue', label='left')
    plt.plot(np.arange(traj_right.shape[0]), gen_model.energy(traj_right), color='red', label='right')
    plt.ylabel('Energy')
    plt.xlabel('Time / steps')
    plt.legend()
    #plt.show()

    both_traj = [traj_left, traj_right]
    names = ['left', 'right']
    for ind, traj in enumerate(both_traj):
        x0 = vect_to_aa_ind(traj, AA_num=AA_num,pos=0)
        x1 = vect_to_aa_ind(traj, AA_num=AA_num,pos=1)
        plt.scatter(x0,x1, alpha=0.2, label='Trajectory - ' + names[ind] )
        
    plt.xlim([0,20])
    plt.ylim([0,20])
    plt.legend()
    #plt.show()

    x = np.vstack([traj_left, traj_right])
    xval = np.vstack([traj_left_val, traj_right_val])

    #x = traj_left[-50:,:]
    #xval= traj_left[-50:,:]
    '''

    ## Generating data in proportion to the boltzmann dist. 

    evh_vals = []
    seq_ints = []
    for i in range(20):
        for j in range(20):
            seq = np.hstack([i,j])
            seq = seq.reshape(1,-1)
            ham = gen_model.energy(seq)[0]
            #print(ham)
            evh_vals.append(ham)
            seq_ints.append(seq)
    vals = evh_vals
    probs = np.exp(vals) / np.sum(np.exp(vals))
    inds= np.argsort(probs)
    ps = np.sort(probs)
    ev_vals = np.asarray(evh_vals)[inds]
    seq_sort = np.asarray(seq_ints)[inds]
    cum =np.cumsum(ps)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    rand_samples=10000
    rands = np.random.uniform(0,1,rand_samples)

    samp_seqs = []
    for r in rands:
        nearest_prob_ind = find_nearest(cum, r)
        nearest_score = ev_vals[nearest_prob_ind]
        nearest_seq = seq_sort[nearest_prob_ind]
        samp_seqs.append(nearest_seq)

    samp_seqs = np.asarray(samp_seqs).reshape(rand_samples, -1)

    '''if dequantize:

        def make_oh(model):
            fa_chars = ''.join(model.alphabet)
            oh_mat = dict(zip(fa_chars, np.eye(len(fa_chars))))
            return oh_mat

        one_hot_mat = make_oh(evc_model)
        def single_mut_profile(model, target_seq):
            # load model parameters
            x = one_hot(target_seq)
            h,J = model.h_i, model.J_ij.transpose(0, 2, 1, 3)
            # total energy contribution to each site i (hia + sum_j[Jia,jb])
            PW_i = h + np.einsum('likj,kj->li', J, x)/2
            # subtract energy contributions from wt aa
            PW_sm = PW_i.T - np.einsum('li,li->li', PW_i, x).sum(axis=1)
            # softmax the columns
            PW_p = PW_sm + np.einsum('li,li->li', PW_i, x).sum() # add back the wt baseline
            PW_p = np.exp(PW_p + 10**-8)
            PW_p = PW_p / PW_p.sum(axis=0)
            return PW_p, PW_sm'''
    
    scores = gen_model.energy(samp_seqs)
    plt.figure()
    plt.hist(scores, bins=250)
    plt.gcf().savefig(experiment_dir+'TrainingSequencesDist.png', dpi=250)

    oh = []
    #N=20
    for seq in samp_seqs:
        #print(seq.shape)
        oh.append(onehot(seq,AA))
    oh=np.asarray(oh)

    num_train_and_test = 10000 
    # without replacement this ensures that they are different
    # need to have a train test split by sequence identity at some point
    rand_inds = np.random.choice(np.arange(oh.shape[0]), num_train_and_test, replace=False)
    train_set = rand_inds[: (num_train_and_test//2) ]
    test_set = rand_inds[ (num_train_and_test//2): ]
    x = oh[train_set, :,:]
    x = x.reshape(x.shape[0], -1)

    xval = oh[test_set, :,:]
    xval = xval.reshape(xval.shape[0], -1)

    print(x.shape)
    gen_model = EVCouplingsGenerator(L, AA, h, J)

    network = invnet(gen_model.dim, model_architecture, gen_model, nl_layers=5, nl_hidden=200, 
                                nl_activation=nl_activation)#, nl_activation_scale=nl_activation_scale)

    network1 = network.train_ML(x, xval=xval, lr=lr, std=latent_std, epochs=epochsML, batch_size=batchsize_ML, 
                                                verbose=verbose)

    print('done with ML training')
    sample_z, sample_x, energy_z, energy_x, log_w = network.sample(temperature=1.0, nsample=10000)

    plt.figure()
    plt.hist(energy_x, bins=100)
    plt.gcf().savefig(experiment_dir+'PostML_GeneratedEnergies.png', dpi=250)

    plt.figure()
    plt.plot(network1.history['loss'], label='training')
    plt.plot(network1.history['val_loss'], label='validation')
    plt.legend()
    plt.gcf().savefig(experiment_dir+'PostML_LossCurves.png', dpi=250)

    network.save(experiment_dir+'Model_Post_ML_Training.tf')

    if KL_only:
        network2 = network.train_KL(epochs=epochsKL, lr=lr, batch_size=batchsize_KL, temperature=temperature, explore=explore, verbose=1,
                                 is_discrete=True, save_partway_inter=save_partway_inter, experiment_dir=experiment_dir)
    
        plt.figure()
        plt.plot(network1.history['loss'], label='training')
        plt.plot(network1.history['val_loss'], label='validation')
        plt.legend()
        plt.gcf().savefig(experiment_dir+'PostKL_LossCurves.png', dpi=250)
    
    else: 
        network2 = network.train_flexible(x, xval=xval, lr=lr, std=latent_std, epochs=epochsKL, batch_size=batchsize_KL, 
                                                            weight_ML=ML_weight, weight_KL=KL_weight, weight_MC=0.0, weight_W2=0.0,
                                                            weight_RCEnt=0.0,
                                                            temperature=temperature, explore=explore, verbose=verbose,
                                                            is_discrete=True,
                                                            save_partway_inter=save_partway_inter,
                                                            experiment_dir=experiment_dir)

        plt.figure()
        plt.plot(network2[1][:,0])
        plt.gcf().savefig(experiment_dir+'PostKL_Overall_LossCurve.png', dpi=250)

        plt.figure()
        plt.plot(network2[1][:,1])
        plt.gcf().savefig(experiment_dir+'PostKL_J_LossCurve.png', dpi=250)

        plt.figure()
        plt.plot(network2[1][:,2])
        plt.gcf().savefig(experiment_dir+'PostKL_KL_LossCurve.png', dpi=250)

        '''plt.figure()
        plot_convergence(network1, network2, 0, 2)
        #plt.show()
        plt.gcf().savefig(experiment_dir+'PlotConvergence.png', dpi=250)'''
    

    network.save(experiment_dir+'Model_Post_KL_Training.tf')

    pickle.dump(network1, open(experiment_dir+'losses_ML.pickle', 'wb'))
    pickle.dump(network2, open(experiment_dir+'losses_KL.pickle','wb'))

    sample_z, sample_x, energy_z, energy_x, log_w = network.sample(temperature=1.0, nsample=10000)

    plt.figure()
    plt.hist(energy_x, bins=100)
    #plt.show()
    plt.gcf().savefig(experiment_dir+'GeneratedEnergies.png', dpi=250)

    total_time = time.time() - start_time
    print('======== total time for this run in minutes', total_time/60)
    
    