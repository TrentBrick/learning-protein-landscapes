import numpy as np
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch
from datetime import datetime

import time
import pickle
import os

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
    date_time = str(datetime.now()).replace(' ', '_').replace(':', '_') # ensures there aren't any issues saving this as a file name. 
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
    well_params = DoubleWell.params_default.copy()
    well_params['dim'] = 2
    gen_model = DoubleWell(params=well_params)

    if params['MCMC'] == True:
        nsteps = 10000
        x0_left = np.array([[-1.8, 0.0]])
        x0_right = np.array([[1.8, 0.0]])
        sampler = MetropolisHastings(gen_model, x0=x0_left, noise=0.1, 
                             stride=10, mapper=None,
                             is_discrete=False)
        data1 = sampler.run(nsteps)

        sampler = MetropolisHastings(gen_model, x0=x0_right, noise=0.1, 
                             stride=5, mapper=None,
                             is_discrete=False)
        data2 = sampler.run(nsteps)

        data = np.concatenate([data1, data2 ], axis=0)
        print('amount of concat data', data.shape)
        

    print('the size of all data to be used (train and val)', data.shape)

    # make data a torch tensor
    data = torch.from_numpy(data).float().to(device)

    # prepare transition state
    x_ts = np.vstack([np.zeros(1000), (1.0/gen_model.params['k']) * np.random.randn(1000)]).T

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
        plt.gcf().savefig(experiment_dir+'Expectation_'+name+'_Data_Hist.png', dpi=100)
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

    print('data', data1.shape)
    # printing out where the samples are from
    plt.figure()
    plt.scatter(data1[:,0], data1[:,1], color='blue')
    plt.scatter(data2[:,0], data2[:,1], color='red')
    plt.gcf().savefig(experiment_dir+'training_data.png', dpi=100)
    plt.close()

    plt.figure()
    plt.hist(data1[:,0], color='blue')
    plt.hist(data2[:,0], color='red')
    plt.gcf().savefig(experiment_dir+'training_data_hist.png', dpi=100)
    plt.close()

    if params['MLepochs']>0:
        # only ML training. 

        ML_losses = network.train_flexible(x, xval=xval, lr=params['lr'], std=params['latent_std'], epochs=params['MLepochs'], batch_size=params['MLbatch'], 
                                                    verbose=params['verbose'], clipnorm=params['gradient_clip'], weight_KL=0.0,
                                                    save_partway_inter=params['save_partway_inter'], experiment_dir=experiment_dir)

        ML_losses = ML_losses['total_loss']

        print('done with ML training')
        # TODO: Add in temperature for sampling: temperature=params['temperature']
         
        #exp_energy_x, hard_energy_x = network.sample_energy(num_samples=5000, temperature=params['temperature'] )

        plt.figure()
        fig, axes = plot_network(network, gen_model, data1, data2, x_ts, weight_cutoff=1e-2)
        fig.savefig(experiment_dir+'ML_only_network_plot.png', dpi=100)
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
        KL_losses = network.train_flexible(x, weight_ML=0.0, weight_entropy = params['Entropyweight'], epochs=params['KLepochs'], lr=params['lr'], batch_size=params['KLbatch'], temperature=params['temperature'], 
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
                                                            experiment_dir=experiment_dir, weight_entropy = params['Entropyweight'])

        for loss_to_plot in ['total_loss', 'ld_loss', 'kl_loss', 'ml_loss']:

            print('to plot', loss_to_plot, len(ML_KL_losses[loss_to_plot]))
            plt.figure()
            plt.plot(ML_KL_losses[loss_to_plot])
            plt.gcf().savefig(experiment_dir+'Post_KL_'+loss_to_plot+'_LossCurve.png', dpi=100)
            plt.close()
   
        pickle.dump(ML_KL_losses, open(experiment_dir+'ML_KL_losses_dict.pickle','wb'))
        
    plt.figure()
    fig, axes = plot_network(network, gen_model, data1, data2, x_ts, weight_cutoff=1e-2)
    fig.savefig(experiment_dir+'MLandKL_network_plot.png', dpi=100)
    plt.close()

    total_time = time.time() - start_time
    print('======== total time for this run in minutes', total_time/60)
    with open(experiment_dir+ 'time_taken.txt', 'w') as file:
        file.write('Total time taken was: ' + str(total_time))

def plot_network(network, gen_model, traj_left, traj_right, x_ts, 
        weight_cutoff=1e-2,):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 3.5))
    plt.subplots_adjust(wspace=0.25)
    # Plot X distribution
    axis = axes[0]
    axis.plot(traj_left[:, 0], traj_left[:, 1], linewidth=0, marker='.', markersize=3, color='blue')
    axis.plot(x_ts[:, 0], x_ts[:, 1], linewidth=0, marker='.', markersize=3, color='orange')
    axis.plot(traj_right[:, 0], traj_right[:, 1], linewidth=0, marker='.', markersize=3, color='red')
    axis.set_xlabel('$x_1$')
    axis.set_xlim(-3, 3)
    axis.set_ylabel('$x_2$', labelpad=-12)
    axis.set_ylim(-4, 4)
    axis.set_yticks([-4, -2, 0, 2, 4])

    # Plot Z distribution
    axis = axes[1]
    with torch.no_grad():
        z_left, _, _ = network.forward(torch.from_numpy(traj_left).float())
        z_ts, _, _ = network.forward( torch.from_numpy(x_ts).float())
        z_right, _, _ = network.forward( torch.from_numpy(traj_right).float())

    axis.plot(z_left[:, 0], z_left[:, 1], linewidth=0, marker='.', markersize=3, color='blue')
    axis.plot(z_ts[:, 0], z_ts[:, 1], linewidth=0, marker='.', markersize=3, color='orange')
    axis.plot(z_right[:, 0], z_right[:, 1], linewidth=0, marker='.', markersize=3, color='red')
    circle = plt.Circle((0, 0), radius=1.0, color='black', alpha=0.4, fill=True)
    axis.add_artist(circle)
    circle = plt.Circle((0, 0), radius=2.0, color='black', alpha=0.25, fill=True)
    axis.add_artist(circle)
    circle = plt.Circle((0, 0), radius=3.0, color='black', alpha=0.1, fill=True)
    axis.add_artist(circle)
    axis.set_xlabel('$z_1$')
    axis.set_xlim(-4, 4)
    axis.set_ylabel('$z_2$', labelpad=-12)
    axis.set_ylim(-4, 4)
    axis.set_yticks([-4, -2, 0, 2, 4])

    # Plot proposal distribution

    # getting samples and histograms. 
    X1, Y1 = test_sample(network, temperature=1.0, plot=False) # bin means and then negative log of empirical x0 frequencies. 
    _, W1 = hist_weights(network)
    axis = axes[2]
    # this is a grid of energies that are plotted as a line. ground truth. 
    _, E = gen_model.plot_dimer_energy(axis=axis, temperature=1.0)
    Y1 = Y1 - Y1.min() + E.min()
    Inan = np.where(W1 < weight_cutoff)
    Y1[Inan] = np.nan
    #Y2 = Y2 - Y2.min() + E.min()
    #axis.plot(X2, Y2, color='#FF6600', linewidth=2, label='ML+KL+RC')
    axis.plot(X1, Y1, color='orange', linewidth=2, label='ML+KL')
    axis.set_xlim(-3, 3)
    axis.set_ylim(-12, 5.5)
    axis.set_yticks([])
    axis.set_xlabel('$x_1$')
    axis.set_ylabel('Energy / kT')
    #plt.legend(ncol=1, loc=9, fontsize=12, frameon=False)
    # Plot reweighted distribution
    RX1, RY1, DR1 = test_sample_rew(network, gen_model, temperature=1.0, plot=False)
    axis = axes[3]
    Ex, E = gen_model.plot_dimer_energy(axis=axis, temperature=1.0)
    RY1 = RY1 - RY1[np.isfinite(RY1)].min() + E.min()
    RY1[Inan] = np.nan
    #RY1[RY1 > -4] = np.nan
    #RY2 = RY2 - RY2[np.isfinite(RY2)].min() + E.min()
    #axis.errorbar(RX2, RY2, DR2, color='#FF6600', linewidth=2, label='ML+KL+RC')
    axis.errorbar(RX1, RY1, DR1, color='orange', linewidth=2, label='ML+KL')
    axis.set_xlim(-3, 3)
    axis.set_ylim(-12, 5.5)
    axis.set_yticks([-12, -10, -8, -6, -4, -2, 0, 2, 4])
    axis.set_xlabel('$x_1$')
    axis.set_ylabel('')
    return fig, axes

def test_sample(network, temperature=1.0, nsample=100000, plot=True):
    if nsample <= 100000:
        sample_x = network.sample(temperature=temperature, num_samples=nsample)
    else:
        sample_x = []
        for i in range(int(nsample/100000)):
            sample_x = network.sample(temperature=temperature, num_samples=nsample)
            sample_x.append(sample_x)
        sample_x = np.vstack(sample_x)
    sample_x = sample_x.detach().numpy()
        
    # xgen = network.Tzx.predict(np.sqrt(temperature) * np.random.randn(100000, 2))
    params = DoubleWell.params_default.copy()
    params['dim'] = 2
    double_well = DoubleWell(params=params)
    plt.figure(figsize=(4, 4))

    h, b = np.histogram(sample_x[:, 0], bins=100)
    # h is the numbers in each bin. 
    bin_means = (b[:-1] + b[1:])/2
    Eh = -np.log(h) / temperature # log of numbers in each. this brings it down from the boltzmann. 
    if plot:
        Ex, E = double_well.plot_dimer_energy(temperature=temperature)
        Eh = Eh - Eh.min() + E.min() # from the lowest real energy E, have the increase in energy on a log scale. 
        plt.plot(bin_means, Eh, color='green', linewidth=2)
    return bin_means, Eh

def hist_weights(network):
    sample_x, log_w = network.sample_log_w(temperature=1.0, num_samples=100000)
    log_w -= log_w.max()
    bins = np.linspace(-2.5, 2.5, 100)
    bin_means = (bins[:-1] + bins[1:]) /2
    sample_x_index = np.digitize(sample_x[:, 0], bins)
    whist = np.zeros(len(bins) + 1)
    for i in range(len(log_w)):
        whist[sample_x_index[i]] += np.exp(log_w[i])
    return bin_means, whist[1:-1]

# reweighting
def test_sample_rew(network, gen_model, temperature=1.0, plot=True):
    sample_x, log_w = network.sample_log_w(temperature=1.0, num_samples=100000)
    log_w -= log_w.max()
    bin_means, Es = free_energy_bootstrap(sample_x[:, 0], bins=100, nbootstrap=100, log_weights=log_w)
    plt.figure(figsize=(4, 4))
    Emean = mean_finite(Es, axis=0)-10.7
    Estd = std_finite(Es, axis=0)
    var = mean_finite(std_finite(Es, axis=0) ** 2)
    if plot:
        gen_model.plot_dimer_energy()
        plt.errorbar(bin_means, Emean, Estd, linewidth=2, color='green')
    # variance
    print('Estimator Standard Error: ', np.sqrt(var))
    return bin_means, Emean, Estd

def mean_finite_(x, min_finite=1):
    """ Computes mean over finite values """
    isfin = np.isfinite(x)
    if np.count_nonzero(isfin) > min_finite:
        return np.mean(x[isfin])
    else:
        return np.nan

def std_finite_(x, min_finite=2):
    """ Computes mean over finite values """
    isfin = np.isfinite(x)
    if np.count_nonzero(isfin) >= min_finite:
        return np.std(x[isfin])
    else:
        return np.nan

def mean_finite(x, axis=None, min_finite=1):
    if axis is None:
        return mean_finite_(x)
    if axis == 0 or axis == 1:
        M = np.zeros((x.shape[axis-1],))
        for i in range(x.shape[axis-1]):
            if axis == 0:
                M[i] = mean_finite_(x[:, i])
            else:
                M[i] = mean_finite_(x[i])
        return M
    else:
        raise NotImplementedError('axis value not implemented:', axis)

def std_finite(x, axis=None, min_finite=2):
    if axis is None:
        return mean_finite_(x)
    if axis == 0 or axis == 1:
        S = np.zeros((x.shape[axis-1],))
        for i in range(x.shape[axis-1]):
            if axis == 0:
                S[i] = std_finite_(x[:, i])
            else:
                S[i] = std_finite_(x[i])
        return S
    else:
        raise NotImplementedError('axis value not implemented:', axis)

def free_energy_bootstrap(D, bins=100, range=None, log_weights=None, bias=None, temperature=1.0,
                          nbootstrap=100, align_bins=None):
    """ Bootstrapped free energy calculation

    If D is a single array, bootstraps by sample. If D is a list of arrays, bootstraps by trajectories

    Parameters
    ----------
    D : array of list of arrays
        Samples in the coordinate in which we compute the free energy
    bins : int
        Number of bins
    range : None or (float, float)
        value range for bins, if not given will be chosen by min and max values of D
    nbootstrap : int
        number of bootstraps
    log_weights : None or arrays matching D
        sample weights
    bias : function
        if not None, the given bias will be removed.
    align_bins : None or indices
        if not None, will shift samples to align at the given bins indices

    Returns
    -------
    bin_means : array((nbins,))
        mean positions of bins
    Es : array((sample, nbins))
        for each bootstrap the free energies of bins.

    """
    if range is None:
        range = (np.min(D), np.max(D))
    bin_edges = None
    Es = []
    by_traj = isinstance(D, list)
    for _ in np.arange(nbootstrap):
        Isel = np.random.choice(len(D), size=len(D), replace=True)
        if by_traj:
            Dsample = np.concatenate([D[i] for i in Isel])
            Wsample = None
            if log_weights is not None:
                log_Wsample = np.concatenate([log_weights[i] for i in Isel])
                Wsample = np.exp(log_Wsample - log_Wsample.max())
            Psample, bin_edges = np.histogram(Dsample, bins=bins, range=range, weights=Wsample, density=True)
        else:
            Dsample = D[Isel]
            Wsample = None
            if log_weights is not None:
                log_Wsample = log_weights[Isel]
                Wsample = np.exp(log_Wsample - log_Wsample.max())
            Psample, bin_edges = np.histogram(Dsample, bins=bins, range=range, weights=Wsample, density=True)
        E = -np.log(Psample)
        if align_bins is not None:
            E -= E[align_bins].mean()
        Es.append(E)
    Es = np.vstack(Es)
    bin_means = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if bias is not None:
        B = bias(bin_means) / temperature
        Es -= B

    return bin_means, Es# / temperature