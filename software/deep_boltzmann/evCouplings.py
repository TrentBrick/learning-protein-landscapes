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
import os

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
save_partway_inter=None, KL_only=False, dequantize=True, load_model='None'):

    start_time = time.time()

    # taking a percentage of the total KL epochs. 
    if save_partway_inter is not None: 
        save_partway_inter = int(save_partway_inter*epochsKL)

    # Experiment save name!
    tf.random.set_random_seed(random_seed)
    np.random.seed(random_seed)
    date_time = str(datetime.now()).replace(' ', '_').replace(':', '_')
    # used to save the policy and its outputs. 
    experiment_name = experiment_base_name+"_rand_seed-%s_ML_epochs-%s_KL_epochs-%s_learning_rate-%s_activation-%s_model_architecture-%s_ML_weight-%s_KL_weight-%s_explore%s_temperature-%s_s_time-%s" % (
        random_seed, epochsML, epochsKL, 
        lr, nl_activation, model_architecture, ML_weight, KL_weight, 
        explore, temperature, date_time )

    # make a directory to save all of the outputs in: 
    os.mkdir('experiments/'+experiment_name)
    experiment_dir = 'experiments/'+ experiment_name+'/'

    ######
    focus_seqs = read_fa('EVCouplingsStuff/DYR_ECOLI_1_b0.5.a2m_trimmed.fa')
    evc_model = CouplingsModel('EVCouplingsStuff/DYR.model')
    scores = evc_model.hamiltonians(list(focus_seqs['seq']))
            
    enc_seqs=[]
    for seq in focus_seqs['seq']:
        enc_seqs.append(encode_aa(seq, evc_model.alphabet_map)) 

    enc_seqs = np.asarray(enc_seqs)
    target_seq = enc_seqs[0]#encode_aa(np.char.upper(ali.matrix[0, :]), a2n)

    oh = []
    N=20 # none of these focus have gaps, else should be 21. 
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
    oh = oh.reshape(oh.shape[0], -1)

    h = evc_model.h_i
    t_oh = oh[0]
    t_oh_flat =t_oh.flatten().reshape(-1,1)

    t_seq_aa = focus_seqs.loc[0, 'seq']
    J = evc_model.J_ij

    '''plt.figure()
    print('Plotting a hist of all the natural sequences energies:')
    plt.hist(hamiltonians(enc_seqs, J, h)[:,0])
    #plt.show()
    plt.gcf().savefig(experiment_dir+'HistofNatSeqs.png', dpi=250)'''

    gen_model = EVCouplingsGenerator(L, AA, h, J)

    if dequantize:
        samp_seqs = single_mut_profile(enc_seqs, h, J, AA) # samp seqs are now onehot. 
        samp_seqs = samp_seqs.reshape(samp_seqs.shape[0], -1)
        '''scores = gen_model.energy(samp_seqs) # still takes way too long. 
        plt.figure()
        plt.hist(scores, bins=250)
        plt.gcf().savefig(experiment_dir+'TrainingSequences_argmax_Dist.png', dpi=250)'''

        scores = exp_hamiltonians(samp_seqs, J, h)
        plt.figure()
        plt.hist(scores, bins=250)
        plt.gcf().savefig(experiment_dir+'TrainingSequences_cont_Dist.png', dpi=250)
        plt.close()

        oh = samp_seqs

    '''else: # this takes a very long time to compute!!!
        scores = gen_model.energy(enc_seqs)
        plt.figure()
        plt.hist(scores, bins=250)
        plt.gcf().savefig(experiment_dir+'TrainingSequences_argmax_Dist.png', dpi=250)'''

    num_train_and_test = 5000
    # without replacement this ensures that they are different
    # need to have a train test split by sequence identity at some point
    rand_inds = np.random.choice(np.arange(oh.shape[0]), num_train_and_test, replace=False)
    train_set = rand_inds[: (num_train_and_test//2) ]
    test_set = rand_inds[ (num_train_and_test//2): ]
    x = oh[train_set, :]

    xval = oh[test_set, :]

    network = invnet(gen_model.dim, model_architecture, gen_model, nl_layers=5, nl_hidden=200, 
                                nl_activation=nl_activation)#, nl_activation_scale=nl_activation_scale)

    if epochsML>0:
        network1 = network.train_ML(x, xval=xval, lr=lr, std=latent_std, epochs=epochsML, batch_size=batchsize_ML, 
                                                    verbose=verbose)

        print('done with ML training')
        sample_z, sample_x, energy_z, energy_x, log_w = network.sample(temperature=1.0, nsample=10000)

        plt.figure()
        plt.hist(energy_x, bins=100)
        plt.gcf().savefig(experiment_dir+'PostML_GeneratedEnergies.png', dpi=250)
        plt.close()

        plt.figure()
        plt.plot(network1.history['loss'], label='training')
        plt.plot(network1.history['val_loss'], label='validation')
        plt.legend()
        plt.gcf().savefig(experiment_dir+'PostML_LossCurves.png', dpi=250)
        plt.close()

        network.save(experiment_dir+'Model_Post_ML_Training.tf')
        #pickle.dump(network1, open(experiment_dir+'losses_ML.pickle', 'wb'))

    if KL_only:
        network2 = network.train_KL(epochs=epochsKL, lr=lr, batch_size=batchsize_KL, temperature=temperature, explore=explore, verbose=1,
                                 is_discrete=True, save_partway_inter=save_partway_inter, experiment_dir=experiment_dir)
    
        plt.figure()
        plt.plot(network1.history['loss'], label='training')
        plt.plot(network1.history['val_loss'], label='validation')
        plt.legend()
        plt.gcf().savefig(experiment_dir+'PostKL_LossCurves.png', dpi=250)
        plt.close()

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
        plt.close()

        plt.figure()
        plt.plot(network2[1][:,1])
        plt.gcf().savefig(experiment_dir+'PostKL_J_LossCurve.png', dpi=250)
        plt.close()

        plt.figure()
        plt.plot(network2[1][:,2])
        plt.gcf().savefig(experiment_dir+'PostKL_KL_LossCurve.png', dpi=250)
        plt.close()

        '''plt.figure()
        plot_convergence(network1, network2, 0, 2)
        #plt.show()
        plt.gcf().savefig(experiment_dir+'PlotConvergence.png', dpi=250)'''
   
    network.save(experiment_dir+'Model_Post_KL_Training.tf')

    #pickle.dump(network2, open(experiment_dir+'losses_KL.pickle','wb'))

    sample_z, sample_x, energy_z, energy_x, log_w = network.sample(temperature=1.0, nsample=10000)

    plt.figure()
    plt.hist(energy_x, bins=100)
    #plt.show()
    plt.gcf().savefig(experiment_dir+'GeneratedEnergies.png', dpi=250)
    plt.close()

    total_time = time.time() - start_time
    print('======== total time for this run in minutes', total_time/60)
    
    