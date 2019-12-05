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
import json

from evc import *

from evcouplings.couplings import CouplingsModel
from EVCouplingsStuff.seq_sele import *

import os
cwd = os.getcwd()
print('current directory', cwd)


def main(params):
    
    '''params['MLepochs'] = 200, params['KLepochs'] = 200,
params['lr'] = 0.001, params['MLbatch'] = 128, params['KLbatch'] = 1000,
params['temperature'] = 1.0, params['explore'] = 1.0, params['latent_std']=1.0, 
params['KLweight']=1.0, params['MLweight']=0.1, params['model_architecture'] = 'NNNNS', params['nl_activation'] ='tanh', 
params['nl_activation']_scale = 'tanh', params['verbose']=True, params['random_seed']=27, 
params['experiment_base_name']='didnt_set_exp_name', 
params['save_partway_inter']=None, params['KL_only']=False, params['dequantize']=True, params['load_model']='None'):
'''
    # timing the entire run. 
    start_time = time.time()

    # taking a percentage of the total KL epochs. 
    if params['save_partway_inter'] is not None: 
        params['save_partway_inter'] = int(params['save_partway_inter']*params['KLepochs'])

    # setting the random seeds
    tf.random.set_random_seed(params['random_seed'])
    np.random.seed(params['random_seed'])

    # Creating new directory to save all run outputs in
    date_time = str(datetime.now()).replace(' ', '_').replace(':', '_') # ensures there aren't any issues saving this as a file name. 
    experiment_name = params['experiment_base_name']+"_rand_seed-%s_ML_epochs-%s_KL_epochs-%s_learning_rate-%s_activation-%s_model_architecture-%s_MLweight-%s_KLweight-%s_explore%s_temperature-%s_s_time-%s" % (
        params['random_seed'], params['MLepochs'], params['KLepochs'], 
        params['lr'], params['nl_activation'], params['model_architecture'], params['MLweight'], params['KLweight'], 
        params['explore'], params['temperature'], date_time )
    os.mkdir('experiments/'+experiment_name)
    experiment_dir = 'experiments/'+ experiment_name+'/'

    # write out all of the parameters used into a text file: 
    with open('experiments/'+ 'params_used.txt', 'w') as file:
        file.write(json.dumps(params))

    # Loading in EVCouplings model 
    focus_seqs = read_fa('EVCouplingsStuff/DYR_ECOLI_1_b0.5.a2m_trimmed.fa')
    evc_model = CouplingsModel('EVCouplingsStuff/DYR.model')

    # extracting the model parameters used to determine the evolutionary hamiltonian
    h = evc_model.h_i
    J = evc_model.J_ij

    if params['protein_length'] > 0:
        h = h[0:params['protein_length'], :]
        J = J[0:params['protein_length'], 0:params['protein_length'], :,:]
            
    # converting amino acids into integers and also onehots. 
    enc_seqs=[]
    oh = []
    AA=h.shape[1] # number of amino acids
    for seq in focus_seqs['seq']:
        enc_seq = np.asarray(encode_aa(seq, evc_model.alphabet_map))
        if params['protein_length'] > 0: 
            enc_seq = enc_seq[:params['protein_length']]
        enc_seqs.append(enc_seq) 
        oh.append(onehot(enc_seq,AA))
    enc_seqs = np.asarray(enc_seqs)
    oh=np.asarray(oh)
    N = oh.shape[0] # batch size
    L = oh.shape[1] # length of the protein
    # flattening the one hot
    oh = oh.reshape(oh.shape[0], -1)

    print('the size of oh', oh.shape)
    
    # plotting the distribution of natural sequences. Can be slow to run. 
    '''plt.figure()
    print('Plotting a hist of all the natural sequences energies:')
    plt.hist(hamiltonians(enc_seqs, J, h)[:,0])
    #plt.show()
    plt.gcf().savefig(experiment_dir+'HistofNatSeqs.png', dpi=250)'''

    # loading in the environment class, used to score the evolutionary hamiltonians
    gen_model = EVCouplingsGenerator(L, AA, h, J)

    # set to True by default, finds a probability distribution for the most likely single mutation
    # made to every sequence
    if params['dequantize']:
        samp_seqs = single_mut_profile(enc_seqs, h, J, AA) # samp seqs are now onehot. 
        samp_seqs = samp_seqs.reshape(samp_seqs.shape[0], -1)

        # gets the expectation over the sequence scores and plots them to see what the training data looks like
        scores = exp_hamiltonians(samp_seqs, J, h)
        plt.figure()
        plt.hist(scores, bins=250)
        plt.gcf().savefig(experiment_dir+'TrainingSequences_cont_Dist.png', dpi=250)
        plt.close()
        # setting the onehot to the new params['dequantize']d sequences
        oh = samp_seqs

    # otherwise plot the current sequences from the training data and their distribution. 
    '''else: # this takes a very long time to compute!!!
        scores = gen_model.energy(enc_seqs)
        plt.figure()
        plt.hist(scores, bins=250)
        plt.gcf().savefig(experiment_dir+'TrainingSequences_argmax_Dist.png', dpi=250)'''

    # assert that there is more data than the amount of training data requested: 
    assert N > params['tda'], 'requested using too much training data! Lower --tda <amount of training data>'

    # need to have a train test split by sequence identity at some point
    # currently an even split between training and validation data. 
    rand_inds = np.random.choice(np.arange(N), params['tda'], replace=False)
    train_set = rand_inds[: (params['tda']//2) ]
    test_set = rand_inds[ (params['tda']//2): ]
    x = oh[train_set, :]

    xval = oh[test_set, :]

    network = invnet(gen_model.dim, params['model_architecture'], gen_model, nl_layers=5, nl_hidden=200, 
                                nl_activation=params['nl_activation'])#, params['nl_activation']_scale=params['nl_activation']_scale)

    if params['MLepochs']>0:
        network1 = network.train_ML(x, xval=xval, lr=params['lr'], std=params['latent_std'], epochs=params['MLepochs'], batch_size=params['MLbatch'], 
                                                    verbose=params['verbose'])

        print('done with ML training')
        sample_z, sample_x, energy_z, energy_x, log_w = network.sample(temperature=params['temperature'], nsample=5000)

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

    if params['KL_only']:
        network2 = network.train_KL(epochs=params['KLepochs'], lr=params['lr'], batch_size=params['KLbatch'], temperature=params['temperature'], 
        explore=params['explore'], verbose=params['verbose'],
        is_discrete=True, save_partway_inter=params['save_partway_inter'], experiment_dir=experiment_dir)
    
        plt.figure()
        plt.plot(network1.history['loss'], label='training')
        plt.plot(network1.history['val_loss'], label='validation')
        plt.legend()
        plt.gcf().savefig(experiment_dir+'PostKL_LossCurves.png', dpi=250)
        plt.close()

    else: 
        network2 = network.train_flexible(x, xval=xval, lr=params['lr'], std=params['latent_std'], epochs=params['KLepochs'], batch_size=params['KLbatch'], 
                                                            weight_ML=params['MLweight'], weight_KL=params['KLweight'], weight_MC=0.0, weight_W2=0.0,
                                                            weight_RCEnt=0.0, temperature=params['temperature'], explore=params['explore'], verbose=params['verbose'],
                                                            is_discrete=True, save_partway_inter=params['save_partway_inter'],
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

    sample_z, sample_x, energy_z, energy_x, log_w = network.sample(temperature= params['temperature'], nsample=5000)

    plt.figure()
    plt.hist(energy_x, bins=100)
    #plt.show()
    plt.gcf().savefig(experiment_dir+'GeneratedEnergies.png', dpi=250)
    plt.close()

    total_time = time.time() - start_time
    print('======== total time for this run in minutes', total_time/60)
    with open('experiments/'+ 'time_taken.txt', 'w') as file:
        file.write('Total time taken was: ' + str(total_time))
    
    