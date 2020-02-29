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

from double_well_model import *

from metropolis import MetropolisHastings
from utils import *

from nflib.MADE import *
from nflib.flows import *
from nflib.spline_flows import NSF_AR, NSF_CL
import itertools

import os
cwd = os.getcwd()
print('current directory', cwd)

def main(params):

    # setting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params['device'] = device
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # timing the entire run. 
    start_time = time.time()

    if params['random_seed'] == 0:
        params['random_seed'] = np.random.randint(1,100)

    # setting the random seeds
    torch.manual_seed(params['random_seed'])
    np.random.seed(params['random_seed'])

    # Creating new directory to save all run outputs in
    date_time = 'lolzzz' #str(datetime.now()).replace(' ', '_').replace(':', '_') # ensures there aren't any issues saving this as a file name. 
    experiment_name = params['exp_base_name']+"_rand_seed-%s_ML_epochs-%s_KL_epochs-%s_learning_rate-%s_MLweight-%s_KLweight-%s_explore%s_temperature-%s_s_time-%s" % (
        params['random_seed'], params['MLepochs'], params['KLepochs'], 
        params['lr'], params['MLweight'], params['KLweight'], 
        params['explore'], params['temperature'], date_time )
    os.mkdir('experiments/'+experiment_name)
    experiment_dir = 'experiments/'+ experiment_name+'/'

    # write out all of the parameters used into a text file: 
    with open(experiment_dir+ 'params_used.txt', 'w') as file:
        file.write(json.dumps(params, cls=NumpyEncoder))

    # loading in the environment class, used to score the evolutionary hamiltonians
    gen_model = DoubleWell()

    if params['MCMC'] == True:
        nsteps = 3000
        x0_left = np.array([[-1.8, 0.0]])
        x0_right = np.array([[1.8, 0.0]])
        sampler = MetropolisHastings(gen_model, x0=x0_left, noise=5, 
                             stride=5, mapper=None, 
                             is_discrete=False)
        data1 = sampler.run(nsteps)

        sampler = MetropolisHastings(gen_model, x0=x0_right, noise=5, 
                             stride=5, mapper=None, 
                             is_discrete=False)
        data2 = sampler.run(nsteps)

        data = np.concatenate([data1, data2 ], axis=0)
        print('amount of concat data', data.shape)
        

    print('the size of all data to be used (train and val)', data.shape)

    # make data a torch tensor
    data = torch.from_numpy(data).float().to(device)

    # make train test split
    rand_inds = np.random.choice(np.arange(data.shape[0]), params['tda'], replace=False)
    train_set = rand_inds[: (params['tda']//2) ]
    test_set = rand_inds[ (params['tda']//2): ]
    x = data[train_set, :]
    xval = data[test_set, :]

    print('shape of data used for training', x.shape)

    # plotting the training and Xval dataset energy histograms: 
    for dset, name in zip([x, xval], ['Train', 'XVal']):
        plt.figure()
        scores = gen_model.energy(dset.cpu().detach().numpy())
        plt.hist(scores, bins=100)
        plt.gcf().savefig(experiment_dir+'Expectation_Sequences_'+name+'_Data_Hist.png', dpi=100)
        plt.close()

    # ======= setting up the normalizing flows: 
    # logistic distribution
    # base = TransformedDistribution(Uniform(torch.zeros(gen_model.dim), torch.ones(gen_model.dim)), SigmoidTransform().inv)
    base = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(gen_model.dim), torch.eye(gen_model.dim))

    if params['model_type'] == 'realNVP':
        # RealNVP
        # used to have 9 layers
        flows = [AffineHalfFlow(dim=gen_model.dim, parity=i%2, nh=params['hidden_dim'], block_mask=params['block_mask']) for i in range(params['num_layers'])]

    if params['model_type'] == 'NICE':
        # NICE
        # 4 layers
        flows = [AffineHalfFlow(dim=gen_model.dim, parity=i%2, nh=params['hidden_dim'] ,scale=False, block_mask=params['block_mask']) for i in range(params['num_layers'])]
        flows.append(AffineConstantFlow(dim=gen_model.dim, shift=False))

    if params['model_type'] == 'slowMAF':
        #SlowMAF (MAF, but without any parameter sharing for each dimension's scale/shift)
        #4 layers
        flows = [SlowMAF(dim=gen_model.dim, parity=i%2, nh=params['hidden_dim']) for i in range(params['num_layers'])]

    if params['model_type'] == 'MAF':
        # MAF (with MADE net, so we get very fast density estimation)
        # 4 layers
        flows = [MAF(dim=gen_model.dim, parity=i%2, nh=params['hidden_dim']) for i in range(params['num_layers'])]

    # Neural splines, coupling
    if params['model_type'] == 'neuralSpline':
        nfs_flow = NSF_CL if True else NSF_AR
        # MAY WANT TO CHANGE THIS HIDDEN_DIM SIZE!
        # 3 layers
        flows = [nfs_flow(dim=gen_model.dim, K=8, B=3, hidden_dim=params['hidden_dim']) for _ in range(params['num_layers'])]
        convs = [Invertible1x1Conv(dim=gen_model.dim) for _ in flows]
        # PREVIOUSLY WAS ACTNORM BUT THIS CLEVER INIT DOESNT WORK FOR ONEHOTS
        norms = [AffineConstantFlow(dim=gen_model.dim) for _ in flows]
        flows = list(itertools.chain(*zip(norms, convs, flows)))

    network = NormalizingFlowModel(base, flows, gen_model)
    network.flow.to(device)

    if params['MLepochs']>0:
        # only ML training. 

        ML_losses = network.train_flexible(x, xval=xval, lr=params['lr'], std=params['latent_std'], epochs=params['MLepochs'], batch_size=params['MLbatch'], 
                                                    verbose=params['verbose'], clipnorm=params['gradient_clip'], weight_KL=0.0,
                                                    save_partway_inter=params['save_partway_inter'], experiment_dir=experiment_dir)

        ML_losses = ML_losses['total_loss']

        print('done with ML training')
        # TODO: Add in temperature for sampling: temperature=params['temperature']
         
        exp_energy_x, hard_energy_x = network.sample_energy(num_samples=5000, temperature=params['temperature'] )

        plt.figure()
        plt.hist(exp_energy_x, bins=100)
        plt.gcf().savefig(experiment_dir+'Post_ML_Expectation_GeneratedEnergies.png', dpi=100)
        plt.close()

        plt.figure()
        plt.plot(ML_losses, label='training')
        #plt.plot(network1.history['val_loss'], label='validation')
        plt.legend()
        plt.gcf().savefig(experiment_dir+'Post_ML_LossCurves.png', dpi=100)
        plt.close()

        torch.save(network.flow.state_dict(), experiment_dir+'Model_Post_ML_Training.torch')
        pickle.dump(ML_losses, open(experiment_dir+'ML_only_losses_dict.pickle','wb'))

    if params['KL_only']:
        KL_losses = network.train_flexible(x, weight_ML=0.0, epochs=params['KLepochs'], lr=params['lr'], batch_size=params['KLbatch'], temperature=params['temperature'], 
        explore=params['explore'], verbose=params['verbose'],
        save_partway_inter=params['save_partway_inter'], experiment_dir=experiment_dir, clipnorm=params['gradient_clip'])
    
        KL_losses = KL_losses['total_loss']

        plt.figure()
        plt.plot(KL_losses, label='training')
        #plt.plot(KL_losses, label='validation')
        plt.legend()
        plt.gcf().savefig(experiment_dir+'Post_KL_LossCurves.png', dpi=100)
        plt.close()

        torch.save(network.flow.state_dict(), experiment_dir+'Model_Post_KL_Training.torch')
        pickle.dump(KL_losses, open(experiment_dir+'KL_only_losses_dict.pickle','wb'))


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
   
    pickle.dump(ML_KL_losses, open(experiment_dir+'ML_KL_losses_dict.pickle','wb'))

    exp_energy_x, hard_energy_x = network.sample_energy(num_samples=5000, temperature=params['temperature'])

    plt.figure()
    fig, axes = plot_network(network, weight_cutoff=1e-2)
    plt.gcf().savefig(experiment_dir+'network_plot.png', dpi=100)
    plt.close()

    total_time = time.time() - start_time
    print('======== total time for this run in minutes', total_time/60)
    with open(experiment_dir+ 'time_taken.txt', 'w') as file:
        file.write('Total time taken was: ' + str(total_time))
