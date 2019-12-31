import numpy as np
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

def plot_mcmc(traj_left, traj_right, AA_num, pos=0):
    plt.figure(figsize=(9, 4))
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((1, 3), (0, 2))
    
    # weighted average of the softmax of hte points. 
    
    left_x1 = vect_to_aa_ind(traj_left ,AA_num,pos=pos)
    right_x1 = vect_to_aa_ind(traj_right ,AA_num,pos=pos)
        
    
    ax1.plot(left_x1, color='blue', label='left', alpha=0.7)
    ax1.plot(right_x1, color='red', label='right', alpha=0.7)
    #ax1.set_xlim(0, 1000)
    #ax1.set_ylim(-2.5, 2.5)
    ax1.set_xlabel('Time / steps')
    ax1.set_ylabel('$x_'+str(pos)+'$ / a.u.')
    ax2.hist(left_x1, 30, orientation='horizontal', histtype='stepfilled', color='blue', alpha=0.2);
    ax2.hist(left_x1, 30, orientation='horizontal', histtype='step', color='blue', linewidth=2);
    ax2.hist(right_x1, 30, orientation='horizontal', histtype='stepfilled', color='red', alpha=0.2);
    ax2.hist(right_x1, 30, orientation='horizontal', histtype='step', color='red', linewidth=2);
    ax2.set_xticks([])
    ax2.set_yticks([])
    #ax2.set_ylim(-2.5, 2.5)
    ax2.set_xlabel('Probability')
    ax1.legend()
    #plt.savefig(paper_dir + 'figs/double_well/prior_trajs.pdf', bbox_inches='tight')
    
def plot_potential_simple(n, score, cbar=True, orientation='vertical', figsize=(4, 5.5)):
    # 2D potential
    '''xgrid = np.arange(0, n)
    ygrid = np.arange(0, n)
    
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    X = np.vstack([Xgrid.flatten(), Ygrid.flatten()]).T
    #print(X)'''
    I = np.eye(n)
    full_grid = []
    for i in range(n):
        x = I[i,:]
        for j in range(n):
            full_grid.append(np.hstack([x, I[j,:]]))

    X = np.asarray(full_grid)
    E = score(X) #double_well.energy(X)
    E = E.reshape((n, n))
    #E = np.minimum(E, 10.0)
    plt.figure(figsize=figsize)
    ax = sns.heatmap(E.T, square = True)
    
def plot_potential(n, target_seq, score, cbar=True, orientation='vertical', pos1=0, pos2=1, figsize=(4, 5.5)):
    
    print('changing only two amino acid positions: pos1=', pos1,' pos2=', pos2)
    
    # 2D potential
    xgrid = np.arange(0, n)
    ygrid = np.arange(0, n)
    
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    X = np.vstack([Xgrid.flatten(), Ygrid.flatten()]).T
    #print(X)
    #I = np.eye(n)
    full_grid = []
    for i in range(n):
        #x = I[i,:]
        
        for j in range(n):
            #y = I[j,:]
            new_seq = np.copy(target_seq)
            new_seq[pos1] = i
            new_seq[pos2] = j
            full_grid.append(new_seq)

    X = np.asarray(full_grid)
    print(X.shape)
    E = score(X) 
    print(E.shape)
    E = E.reshape((n, n))
    #E = np.minimum(E, 10.0)
    plt.figure(figsize=figsize)
    ax = sns.heatmap(E.T, square = True)
    
    
def test_sample(network, model, AA_num, temperature=1.0, nsample=100000, plot=True):
    if nsample <= 100000:
        sample_z, sample_x, energy_z, energy_x, logw = network.sample(temperature=temperature, nsample=nsample)
    else:
        sample_x = []
        for i in range(int(nsample/100000)):
            _, sample_x_, _, _, _ = network.sample(temperature=temperature, nsample=nsample)
            sample_x.append(sample_x_)
        sample_x = np.vstack(sample_x)
        
    # xgen = network.Tzx.predict(np.sqrt(temperature) * np.random.randn(100000, 2))
    plt.figure(figsize=(4, 4))
    
    sample_x1 = vect_to_aa_ind(sample_x, AA_num,pos=0)
    
    h, b = np.histogram(sample_x1, bins=100)
    bin_means = 0.5*(b[:-1] + b[1:])
    Eh = -np.log(h) / temperature
    if plot:
        Ex, E = model.plot_energy(temperature=temperature)
        Eh = Eh - Eh.min() + E.min()
        plt.plot(bin_means, Eh, color='green', linewidth=2)
    return bin_means, Eh
    
def vect_to_aa_ind(vect, AA_num,pos=0):
    # currently am not feeding in AA_num = 21 anywhere. 
    s = pos*AA_num
    e = s+AA_num
    return np.argmax(vect[:, s:e], axis=1)
# bias towards the centernp.sum(softmax(vect[:, s:e], axis=1) * np.arange(AA_num), axis=1)
    
# reweighting
def test_sample_rew(network, model, AA_num, temperature=1.0, plot=True):
    sample_z, sample_x, energy_z, energy_x, log_w = network.sample(temperature=1.0, nsample=100000)
    log_w -= log_w.max()
    print(log_w)
    sample_x0 = vect_to_aa_ind(sample_x, AA_num, pos=0)
    bin_means, Es = free_energy_bootstrap(sample_x0, bins=100, nbootstrap=100, log_weights=np.asarray(log_w))
    plt.figure(figsize=(4, 4))
    Emean = mean_finite(Es, axis=0)-10.7
    Estd = std_finite(Es, axis=0)
    var = mean_finite(std_finite(Es, axis=0) ** 2)
    if plot:
        model.plot_energy()
        plt.errorbar(bin_means, Emean, Estd, linewidth=2, color='green')
    # variance
    print('Estimator Standard Error: ', np.sqrt(var))
    return bin_means, Emean, Estd
    
def hist_weights(network, AA_num):
    sample_z, sample_x, energy_z, energy_x, log_w = network.sample(temperature=1.0, nsample=100000)
    log_w -= log_w.max()
    bins = np.linspace(-2.5, 2.5, 100)
    bin_means = 0.5 * (bins[:-1] + bins[1:])
    sample_x0 = vect_to_aa_ind(sample_x, AA_num, pos=0)
    sample_x_index = np.digitize(sample_x0, bins)
    whist = np.zeros(len(bins) + 1)
    for i in range(len(log_w)):
        whist[sample_x_index[i]] += np.exp(log_w[i])
    return bin_means, whist[1:-1]
    
    
def plot_all_zs(traj_left, traj_right, x_ts, alpha=0.1):
    for i in range(traj_left.shape[1]):
        z_left = network.transform_xz(traj_left)
        z_ts = network.transform_xz(x_ts)
        z_right = network.transform_xz(traj_right)
        plt.plot(z_left[:, 0], z_left[:, 1], linewidth=0, marker='.', markersize=3, color='blue')
        plt.plot(z_right[:, 0], z_right[:, 1], linewidth=0, marker='.', markersize=3, color='red')
    plt.plot(z_ts[:, 0], z_ts[:, 1], linewidth=0, marker='.', markersize=3, color='orange')
    
    circle = plt.Circle((0, 0), radius=1.0, color='black', alpha=0.4, fill=True)
    plt.add_artist(circle)
    circle = plt.Circle((0, 0), radius=2.0, color='black', alpha=0.25, fill=True)
    plt.add_artist(circle)
    circle = plt.Circle((0, 0), radius=3.0, color='black', alpha=0.1, fill=True)
    plt.add_artist(circle)
    plt.set_xlabel('$z_1$')
    plt.set_xlim(-4, 4)
    plt.set_ylabel('$z_2$', labelpad=-12)
    plt.set_ylim(-4, 4)
    plt.set_yticks([-4, -2, 0, 2, 4]);
    plt.show()
    
def plot_network(network, traj_left, traj_right, x_ts, AA_num, model, weight_cutoff=1e-2):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 3.5))
    plt.subplots_adjust(wspace=0.25)
    # Plot X distribution
    
    
    left_x1 = vect_to_aa_ind(traj_left ,AA_num,pos=0)
    left_x2 = vect_to_aa_ind(traj_left ,AA_num,pos=1)
    #np.argmax(traj_left[:, 0:AA_num], axis=1)
    right_x1 = vect_to_aa_ind(traj_right ,AA_num,pos=0)
    right_x2 = vect_to_aa_ind(traj_right ,AA_num,pos=1)
    
    ts_x1 = vect_to_aa_ind(x_ts ,AA_num,pos=0)
    ts_x2 = vect_to_aa_ind(x_ts ,AA_num,pos=1)
    
    axis = axes[0]
    axis.plot(left_x1, left_x2, linewidth=0, marker='.', markersize=3, color='blue')
    axis.plot(ts_x1, ts_x2, linewidth=0, marker='.', markersize=3, color='orange')
    axis.plot(right_x1, right_x2, linewidth=0, marker='.', markersize=3, color='red')
    axis.set_xlabel('$x_1$')
    #axis.set_xlim(-3, 3)
    axis.set_ylabel('$x_2$', labelpad=-12)
    #axis.set_ylim(-4, 4)
    #axis.set_yticks([-4, -2, 0, 2, 4]);
    # Plot Z distribution
    axis = axes[1]
    z_left = network.transform_xz(traj_left)
    z_ts = network.transform_xz(x_ts)
    z_right = network.transform_xz(traj_right)
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
    axis.set_yticks([-4, -2, 0, 2, 4]);
    # Plot proposal distribution
    X1, Y1 = test_sample(network, model, AA_num, temperature=1.0, plot=False);
    _, W1 = hist_weights(network, AA_num)
    axis = axes[2]
    Ex, E = model.plot_energy(axis=axis, temperature=1.0)
    Y1 = Y1 - Y1.min() + E.min()
    Inan = np.where(W1 < weight_cutoff)
    Y1[Inan] = np.nan
    #Y2 = Y2 - Y2.min() + E.min()
    #axis.plot(X2, Y2, color='#FF6600', linewidth=2, label='ML+KL+RC')
    axis.plot(X1, Y1, color='orange', linewidth=2, label='ML+KL')
    #axis.set_xlim(-3, 3)
    #axis.set_ylim(-12, 5.5)
    axis.set_yticks([]);
    axis.set_xlabel('$x_1$')
    axis.set_ylabel('Energy / kT')
    #plt.legend(ncol=1, loc=9, fontsize=12, frameon=False)
    # Plot reweighted distribution
    RX1, RY1, DR1 = test_sample_rew(network, model, AA_num, temperature=1.0, plot=False);
    axis = axes[3]
    Ex, E = model.plot_energy(axis=axis, temperature=1.0)
    RY1 = RY1 - RY1[np.isfinite(RY1)].min() + E.min()
    RY1[Inan] = np.nan
    #RY1[RY1 > -4] = np.nan
    #RY2 = RY2 - RY2[np.isfinite(RY2)].min() + E.min()
    #axis.errorbar(RX2, RY2, DR2, color='#FF6600', linewidth=2, label='ML+KL+RC')
    axis.errorbar(RX1, RY1, DR1, color='orange', linewidth=2, label='ML+KL')
    #axis.set_xlim(-3, 3)
    #axis.set_ylim(-12, 5.5)
    #axis.set_yticks([-12, -10, -8, -6, -4, -2, 0, 2, 4]);
    axis.set_xlabel('$x_1$')
    axis.set_ylabel('')
    return fig, axes