"""
Implements various flows.
Each flow is invertible so it can be forward()ed and backward()ed.
Notice that backward() is not backward as in backprop but simply inversion.
Each flow also outputs its log det J "regularization"
Reference:
NICE: Non-linear Independent Components Estimation, Dinh et al. 2014
https://arxiv.org/abs/1410.8516
Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
https://arxiv.org/abs/1505.05770
Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803
(Laurent's extension of NICE)
Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934
(IAF)
Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017 
https://arxiv.org/abs/1705.07057
"The advantage of Real NVP compared to MAF and IAF is that it can both generate data and estimate densities with one forward pass only, whereas MAF would need D passes to generate data and IAF would need D passes to estimate densities."
(MAF)
Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal, Jul 2018
https://arxiv.org/abs/1807.03039
"Normalizing Flows for Probabilistic Modeling and Inference"
https://arxiv.org/abs/1912.02762
(review paper)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pickle

import matplotlib
if platform.system() == 'Darwin':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from nflib.nets import LeafParam, MLP, ARMLP

class AffineConstantFlow(nn.Module):
    """ 
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """
    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None
        
    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False
    
    def forward(self, x):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None # for now
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().forward(x)


class AffineHalfFlow(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the 
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """
    def __init__(self, dim, parity, net_class=MLP, nh=24, scale=True, shift=True):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if scale:
            self.s_cond = net_class(self.dim // 2, self.dim // 2, nh)
        if shift:
            self.t_cond = net_class(self.dim // 2, self.dim // 2, nh)
        
    def forward(self, x):
        x0, x1 = x[:,::2], x[:,1::2]
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(x0)
        t = self.t_cond(x0)
        z0 = x0 # untouched half
        z1 = torch.exp(s) * x1 + t # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        z0, z1 = z[:,::2], z[:,1::2]
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(z0)
        t = self.t_cond(z0)
        x0 = z0 # this was the same
        x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class SlowMAF(nn.Module):
    """ 
    Masked Autoregressive Flow, slow version with explicit networks per dim
    """
    def __init__(self, dim, parity, net_class=MLP, nh=24):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleDict()
        self.layers[str(0)] = LeafParam(2)
        for i in range(1, dim):
            self.layers[str(i)] = net_class(i, 2, nh)
        self.order = list(range(dim)) if parity else list(range(dim))[::-1]
        
    def forward(self, x):
        z = torch.zeros_like(x)
        log_det = torch.zeros(x.size(0))
        for i in range(self.dim):
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            z[:, self.order[i]] = x[:, i] * torch.exp(s) + t
            log_det += s
        return z, log_det

    def backward(self, z):
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0))
        for i in range(self.dim):
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            x[:, i] = (z[:, self.order[i]] - t) * torch.exp(-s)
            log_det += -s
        return x, log_det

class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """
    
    def __init__(self, dim, parity, net_class=ARMLP, nh=24):
        # nh is the number of hidden units.
        super().__init__()
        self.dim = dim
        # nin, nout, number hidden
        # parity flips the ordering so the transform is applied to the other side. 
        self.net = net_class(dim, dim*2, nh)
        self.parity = parity

    def forward(self, x):
        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        st = self.net(x)
        s, t = st.split(self.dim, dim=1)
        z = x * torch.exp(s) + t
        # reverse order, so if we stack MAFs correct things happen
        z = z.flip(dims=(1,)) if self.parity else z
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        # we have to decode the x one at a time, sequentially
        x = torch.zeros_like(z) # the first pass does nothing, copying the value. just adds lots of padding.
        log_det = torch.zeros(z.size(0))
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            st = self.net(x.clone()) # clone to avoid in-place op errors if using IAF
            s, t = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det

class IAF(MAF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        reverse the flow, giving an Inverse Autoregressive Flow (IAF) instead, 
        where sampling will be fast but density estimation slow
        """
        self.forward, self.backward = self.backward, self.forward


class Invertible1x1Conv(nn.Module):
    """ 
    As introduced in Glow paper.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P # remains fixed during optimization
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def backward(self, z):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det

# ------------------------------------------------------------------------

class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        #zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            #temp = flow.forward(x)
            #print(temp)
            log_det += ld
            #zs.append(x)
        return x, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        #xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            #xs.append(z)
        return z, log_det

class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """
    
    def __init__(self, prior, flows, energy_model):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)
        self.energy_model = energy_model
    
    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        prior_logprob = self.prior.log_prob(zs).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det
    
    def sample(self, num_samples=10000, temperature=1.0):
        z = np.sqrt(temperature) * self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs
    
    def sample_energy(self, num_samples=5000, temperature=1.0):
        sample_x = self.sample(num_samples=num_samples, temperature=temperature ).detach().numpy()
        exp_energy_x = self.energy_model.energy(sample_x) / temperature
        # want to arg max these sequences. Using numpy commands as .energy is in numpy rather than tensorflow. 
        h_max = np.reshape(sample_x, (num_samples, self.energy_model.L, self.energy_model.AA_num ))
        h_max = np.argmax(h_max, axis=-1)
        h_max = np.reshape(h_max, (num_samples, self.energy_model.L ) )
        # fed into energy as integers where they are then turned into onehots. 
        #print('hard max is', h_max.shape)
        hard_energy_x = self.energy_model.energy(h_max) / temperature 
        return exp_energy_x, hard_energy_x # dont want to return any of the other types of energy right now. 

    def train_flexible(self, x, xval=None, optimizer=None, lr=0.001, epochs=2000, 
                       batch_size=1024, verbose=1, clipnorm=None,
                       high_energy=100, max_energy=1e10, std=1.0, reg_Jxz=0.0,
                       weight_ML=1.0, weight_KL=1.0, entropy_weight = 1.0, 
                       temperature=1.0, explore=1.0,
                       save_partway_inter=None,
                       experiment_dir='DidNotPutInAName'):

        if optimizer is None:
            optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)

        # history tracker
        losses_dict = {'total_loss':[]}
        if weight_ML>0.0:
            losses_dict['ml_loss']=[]
        if weight_KL > 0.0:
            losses_dict['ent_loss'] = []
            losses_dict['ld_loss'] = []
            losses_dict['kl_loss'] = []

        self.flow.train()
        for e in range(epochs): 

            optimizer.zero_grad()

            total_loss = torch.zeros(batch_size,1)

            if weight_ML > 0.0:

                # sample training data: 
                #TODO: Develop a dataloader to make this faster
                rand_inds = np.random.choice(np.arange(len(x)), batch_size)
                data = torch.from_numpy(x[rand_inds]).float()

                # forward and reverse are actually the other way around here. 
                zs, prior_logprob, forward_log_det = self.forward( data )
        
                forward_logprob = prior_logprob + forward_log_det
                loss_ML = weight_ML*self.ML_loss(zs, forward_logprob, std=std).unsqueeze(1) # this should be of dim: batch x 1

                #print('dim of loss_ml', loss_ML.shape)
                #print('loss ml', loss_ML)

                total_loss += loss_ML

            if weight_KL > 0.0:

                # sample Z values
                latents = self.sample(batch_size)
                xs, backward_log_det = self.backward(latents)

                loss_KL, ld_loss, ent_loss = self.KL_loss(xs, backward_log_det, 
                    temperature_factors=temperature, explore=explore, 
                    entropy_weight=entropy_weight)

                loss_KL = loss_KL*weight_KL
                #print('loss_KL', loss_KL.shape)
                #print('pre KL total loss', total_loss.shape)
                total_loss += loss_KL

            #print('total loss before the sum', total_loss)
            #print(total_loss.shape)
            total_loss = total_loss.sum() / batch_size
            total_loss.backward()
            torch.nn.utils.clip_grad_norm(self.flow.parameters(), clipnorm)
            optimizer.step()
            
            losses_dict['total_loss'].append(total_loss.detach().numpy())
            if weight_ML>0.0:
                losses_dict['ml_loss'].append(loss_ML.sum().detach().numpy())
            if weight_KL>0.0:
                losses_dict['kl_loss'].append(loss_KL.sum().detach().numpy())
                losses_dict['ent_loss'].append(ent_loss.detach().numpy())
                losses_dict['ld_loss'].append(ld_loss.detach().numpy())
            
            if e%1==0:
                print('===================')
                print('epoch:',e, 'Total loss:',total_loss.item())

                if weight_KL>0.0 and weight_ML>0.0: # else total loss will have the same info
                    print( "Loss KL:", loss_KL.sum().detach().numpy() )
                    print("Loss Log Det:", ld_loss.detach().numpy())
                    print( "Loss ML:", loss_ML.sum().detach().numpy() )

            #TODO: add in xval dataset

            if save_partway_inter is not None and (e+1)%save_partway_inter==0: 

                #self.save(experiment_dir+'Model_During_'+str(e)+'_KL_Training.tf')
                exp_energy_x, hard_energy_x = self.sample_energy(num_samples=5000, temperature=temperature)

                plt.figure()
                plt.hist(exp_energy_x, bins=100)
                plt.gcf().savefig(experiment_dir+'Expectation_Energies_Epoch_'+str(e)+'_ML_'+str(weight_ML)+'_KL_'+str(weight_KL)+'_.png', dpi=100)
                plt.close()

                plt.figure()
                plt.hist(hard_energy_x, bins=100)
                plt.gcf().savefig(experiment_dir+'ArgMax_Energies_Epoch_'+str(e)+'_ML_'+str(weight_ML)+'_KL_'+str(weight_KL)+'_.png', dpi=100)
                plt.close()

                # also saving out the learning trajectory:
                #pickle.dump(np.array(losses_dict), open(experiment_dir+'During_'+str(e)+'losses_dict.pickle', 'wb')) 

        return losses_dict

    def ML_loss(self, z ,log_det, std=1.0):
        return - (log_det - (0.5 / (std**2)) * torch.sum(z**2, dim=1))

    def entropy_seq(self, x):
        """Takes in a batch of sequences of shape 'batchsize x protein length x # AAs' 
        that have been softmaxed and computes their entropy"""

        # need to allow for numerical stability. nans propagate. 
        unq_ents = x * torch.log(x+0.000000000001) # elementwise multiplication of the probabilities
        pos_ents = torch.sum(unq_ents, dim=2) #position wise entropies calculated
        seq_ents = torch.sum(pos_ents, dim=1, keepdim=True) # sum up the entropy for each sequence
        # should now be a batch of entropy numbers
        return -seq_ents

    def softmaxer(self, inp, batch_size):
        #taking the softmax first. 
        inp = inp.view( (batch_size, -1, self.energy_model.AA_num) )
        inp = F.softmax(inp,dim=-1)
        return inp

    def KL_loss(self, x, log_det, high_energy=10, 
    max_energy=100, temperature_factors=1.0, explore=1.0, entropy_weight=1.0):
        # explore is responsible for how large the log determinant should be. 
        batch_size = x.shape[0]

        x_sm = self.softmaxer(x, batch_size) # returns batchsize x protein length x # AAs
        #reshaping so that its flat again
        x_sm_flat = x_sm.view((batch_size, -1))

        E = self.energy_model.discrete_energy_torch(x_sm_flat)
        #E = self.energy_model.discrete_energy_tf(x, b)
        #print('these are the energy rewards', E.shape)

        # energy clipping for the unstable NVP training. 
        #Ereg = -linlogcut(-E, high_energy, max_energy, tf=True)

        #print('log det', log_det.shape)
        
        ent_loss = (entropy_weight * self.entropy_seq(x_sm)).float() # getting the entropy of the sequences
        #print( "sequence entropies", ent_loss.shape)

        ld_loss = (explore * log_det).float().unsqueeze(1)
        #print('ld_loss', ld_loss.shape)
        loss = - E - ld_loss + ent_loss #tf.zeros((batch_size, 1), dtype=tf.float32)#
        #print('kl loss', loss)
        #print('kl loss shape', loss.shape)
        return loss.float(), -ld_loss.sum(), ent_loss.sum()  #  SHOULD THIS BE SUMMED UP OR DIVIDED BY BATCHSIZE? DOES KERAS TAKE THE MEAN? 
