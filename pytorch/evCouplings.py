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

from nflib.MADE import *
from nflib.flows import *
from nflib.spline_flows import NSF_AR, NSF_CL
import itertools

import os
cwd = os.getcwd()
print('current directory', cwd)

def main(params):

    # timing the entire run. 
    start_time = time.time()

    # taking a percentage of the total KL epochs. 
    if params['save_partway_inter'] is not None: 
        params['save_partway_inter'] = int(params['save_partway_inter']*params['KLepochs'])

    # setting the random seeds
    torch.manual_seed(params['random_seed'])
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
    with open(experiment_dir+ 'params_used.txt', 'w') as file:
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
        oh.append(onehot(enc_seq,AA)) # this could be made much more efficient with tensorflow operations. 
    enc_seqs = np.asarray(enc_seqs)
    oh=np.asarray(oh)
    N = oh.shape[0] # batch size
    L = oh.shape[1] # length of the protein
    # flattening the one hot
    oh = oh.reshape(oh.shape[0], -1)

    print('the size of oh', oh.shape)
    
    # plotting the distribution of natural sequences. As I already know what they look like I have commented this out. 
    '''plt.figure()
    print('Plotting a hist of all the natural sequences energies:')
    plt.hist(gen_model.energy(enc_seqs))
    #plt.show()
    plt.gcf().savefig(experiment_dir+'HistofNatSeqs.png', dpi=100)'''

    # loading in the environment class, used to score the evolutionary hamiltonians
    gen_model = EVCouplingsGenerator(L, AA, h, J)

    # Save the argmax scores for the training sequences
    plt.figure()
    scores = gen_model.energy(enc_seqs)
    plt.hist(scores, bins=250)
    plt.gcf().savefig(experiment_dir+'TrainingSequences_ArgMax_Hist.png', dpi=100)
    plt.close()

    # set to True by default, finds probability distribution for the most likely single mutation
    # made to every sequence
    if params['dequantize']:
        samp_seqs = single_mut_profile(enc_seqs, h, J, AA) # samp seqs are now onehot. 
        samp_seqs = samp_seqs.reshape(samp_seqs.shape[0], -1)

        # gets the expectation over the sequence scores and plots them to see what the training data looks like
        scores = gen_model.energy(samp_seqs)
        plt.figure()
        plt.hist(scores, bins=250)
        plt.gcf().savefig(experiment_dir+'TrainingSequences_Expectation_Hist.png', dpi=100)
        plt.close()
        # setting the onehot to the new params['dequantize']d sequences
        oh = samp_seqs

    # otherwise plot the current sequences from the training data and their distribution. 
    '''else: # this takes a very long time to compute!!!
        scores = gen_model.energy(enc_seqs)
        plt.figure()
        plt.hist(scores, bins=250)
        plt.gcf().savefig(experiment_dir+'TrainingSequences_argmax_Hist.png', dpi=100)'''

    # assert that there is more data than the amount of training data requested: 
    assert N > params['tda'], 'requested using too much training data! Lower --tda <amount of training data>'

    # need to have a train test split by sequence identity at some point
    # currently an even split between training and validation data. 
    rand_inds = np.random.choice(np.arange(N), params['tda'], replace=False)
    train_set = rand_inds[: (params['tda']//2) ]
    test_set = rand_inds[ (params['tda']//2): ]
    x = oh[train_set, :]

    xval = oh[test_set, :]

    base = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(gen_model.dim), torch.eye(gen_model.dim))

    # RealNVP
    #flows = [AffineHalfFlow(dim=2, parity=i%2) for i in range(9)]

    # NICE
    flows = [AffineHalfFlow(dim=gen_model.dim, parity=i%2, nh=64 ,scale=False) for i in range(4)]
    flows.append(AffineConstantFlow(dim=gen_model.dim, shift=False))

    # SlowMAF (MAF, but without any parameter sharing for each dimension's scale/shift)
    # flows = [SlowMAF(dim=2, parity=i%2) for i in range(4)]

    # MAF (with MADE net, so we get very fast density estimation)
    #flows = [MAF(dim=2, parity=i%2, nh=64) for i in range(4)]

    # Neural splines, coupling
    '''nfs_flow = NSF_CL if True else NSF_AR
    # MAY WANT TO CHANGE THIS HIDDEN_DIM SIZE!
    flows = [nfs_flow(dim=2, K=8, B=3, hidden_dim=16) for _ in range(3)]
    convs = [Invertible1x1Conv(dim=2) for _ in flows]
    norms = [ActNorm(dim=2) for _ in flows]
    flows = list(itertools.chain(*zip(norms, convs, flows)))'''

    network = NormalizingFlowModel(base, flows, gen_model)

    #network = invnet(gen_model.dim, params['model_architecture'], gen_model, nl_layers=3, nl_hidden=100, 
    #                            nl_activation=params['nl_activation'],is_discrete=True)#, params['nl_activation']_scale=params['nl_activation']_scale)

    # TODO: Enable loading in of model, use crypto code to get this working. 
    if params['load_model'] != 'None':			
        network = network.load('experiments/'+params['load_model'], gen_model, is_discrete=True)		

    if params['MLepochs']>0:
        # only ML training. 
        ML_losses = network.train_flexible(x, xval=xval, lr=params['lr'], std=params['latent_std'], epochs=params['MLepochs'], batch_size=params['MLbatch'], 
                                                    verbose=params['verbose'], clipnorm=params['gradient_clip'], weight_KL=0.0)

        ML_losses = ML_losses['total_loss']

        print('done with ML training')
        # TODO: Add in temperature for sampling: temperature=params['temperature']
         
        exp_energy_x, hard_energy_x = network.sample_energy(num_samples=5000, temperature=params['temperature'] )

        plt.figure()
        plt.hist(exp_energy_x, bins=100)
        plt.gcf().savefig(experiment_dir+'Post_ML_Expectation_GeneratedEnergies.png', dpi=100)
        plt.close()

        plt.figure()
        plt.hist(hard_energy_x, bins=100)
        plt.gcf().savefig(experiment_dir+'Post_ML_ArgMax_GeneratedEnergies.png', dpi=100)
        plt.close()

        plt.figure()
        plt.plot(ML_losses, label='training')
        #plt.plot(network1.history['val_loss'], label='validation')
        plt.legend()
        plt.gcf().savefig(experiment_dir+'Post_ML_LossCurves.png', dpi=100)
        plt.close()

        #network.save(experiment_dir+'Model_Post_ML_Training.tf')
        #pickle.dump(network1, open(experiment_dir+'losses_ML.pickle', 'wb'))

    if params['KL_only']:
        KL_losses = network.train_flexible(weight_ML=0.0, epochs=params['KLepochs'], lr=params['lr'], batch_size=params['KLbatch'], temperature=params['temperature'], 
        explore=params['explore'], verbose=params['verbose'],
        save_partway_inter=params['save_partway_inter'], experiment_dir=experiment_dir, clipnorm=params['gradient_clip'])
    
        KL_losses = KL_losses['total_loss']

        plt.figure()
        plt.plot(KL_losses, label='training')
        #plt.plot(KL_losses, label='validation')
        plt.legend()
        plt.gcf().savefig(experiment_dir+'Post_KL_LossCurves.png', dpi=100)
        plt.close()

    else: 
        ML_KL_losses = network.train_flexible(x, xval=xval, lr=params['lr'], std=params['latent_std'], epochs=params['KLepochs'], batch_size=params['KLbatch'], 
                                                            weight_ML=params['MLweight'], weight_KL=params['KLweight'],
                                                            temperature=params['temperature'], explore=params['explore'], verbose=params['verbose'],
                                                            save_partway_inter=params['save_partway_inter'], clipnorm=params['gradient_clip'],
                                                            experiment_dir=experiment_dir, entropy_weight = params['Entropyweight'])

        for loss_to_plot in ['total_loss', 'ld_loss', 'kl_loss', 'ml_loss']:

            print('to plot', loss_to_plot, len(ML_KL_losses[loss_to_plot]))
            plt.figure()
            plt.plot(ML_KL_losses[loss_to_plot])
            plt.gcf().savefig(experiment_dir+'Post_KL_'+loss_to_plot+'_LossCurve.png', dpi=100)
            plt.close()
   
    #network.save(experiment_dir+'Model_Post_KL_Training.tf')

    #pickle.dump(network2, open(experiment_dir+'losses_KL.pickle','wb'))

    exp_energy_x, hard_energy_x = network.sample_energy(num_samples=5000, temperature=params['temperature'])

    plt.figure()
    plt.hist(exp_energy_x, bins=100)
    plt.gcf().savefig(experiment_dir+'Post_KL_Expectation_GeneratedEnergies.png', dpi=100)
    plt.close()

    plt.figure()
    plt.hist(hard_energy_x, bins=100)
    plt.gcf().savefig(experiment_dir+'Post_KL_ArgMax_GeneratedEnergies.png', dpi=100)
    plt.close()

    total_time = time.time() - start_time
    print('======== total time for this run in minutes', total_time/60)
    with open(experiment_dir+ 'time_taken.txt', 'w') as file:
        file.write('Total time taken was: ' + str(total_time))
    
    