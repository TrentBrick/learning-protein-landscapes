# Need to make a model for the protein values. It will be 2D before moving 
# up the EV couplings
import numpy as np
import tensorflow as tf
from scipy.special import softmax
class EVCouplingsGenerator(object):

    def __init__(self, protein_length, aa_num, h ,J):
        # set parameters
        self.a2n = dict([(a, n) for n, a in enumerate('ACDEFGHIKLMNPQRSTVWY-')])
        self.AA_num = aa_num
        self.L = protein_length
        self.h = h
        self.J =J
        self.h_tf = tf.expand_dims(tf.reshape(tf.constant(h), [-1]), 1) 
        J_tf = tf.constant(J)
        J_tf = tf.transpose(J, [0,2,1,3])
        J_tf = tf.reshape(J_tf,(protein_length*aa_num,protein_length*aa_num))
        self.J_tf = tf.cast(J_tf, tf.float32)
        self.dim = protein_length*self.AA_num
        
    def oh_to_aa(self, oh):
        if len(oh.shape)>2:
            oh = oh.reshape(oh.shape[0], -1, 20)
        else:
            oh = oh.reshape(-1,20)
        res = (oh@np.arange(self.AA_num).reshape(-1,1)).astype(int)
        return res

    def aa_to_oh(self, labels):
        '''one-hot encode a numpy array'''
        O = np.reshape(np.eye(self.AA_num)[labels], (*labels.shape, self.AA_num))
        return(O)
    
        
    def EVH(self, data, use_tf=True):
        
        if not use_tf:
            evh = hamiltonians(data, self.J, self.h)
        else: 
            #J = tf.Print(self.J_tf, [self.J_tf[2]], 'J matrix shouldnt change!')
            #h = tf.Print(self.h_tf, [self.h_tf], ' H matrix shouldnt change')
            
            evh = tf.squeeze(tf.reduce_sum(data * tf.matmul(data, self.J_tf), axis=-1) /2) + tf.squeeze(tf.matmul(data, self.h_tf))
            #evh =tf.Print(evh, [evh], 'printing the evh again ')
            evh = tf.reduce_sum(evh)

            #same code using the einsum: 
            '''xJ_batch = tf.einsum('nl,lk->nlk', data, self.J_tf)
            xJx_batch = tf.einsum('nl,nlk->nlk', data, tf.transpose(xJ_batch, perm=(0,2,1)))
            evh = tf.reduce_sum(xJx_batch,axis=(-1,-2))/2 
            print(data.shape)
            print(self.h_tf)
            evh = evh + tf.einsum('nl,l->n', data, self.h_tf)# tf.cast(data, tf.float64), tf.cast(tf.squeeze(h_test), tf.float64)) # if h is of dim 1
            #evh = tf.reduce_sum(evh)'''
        return evh
        
    def discrete_energy_tf(self, inp, batch_size):

        # meant to compute discrete values for the sequences sampling from the softmax
        # dont need to do this for EVH because I can compute the expectation across
        # all probabilistic sequences immediately. 
        # can also use GPU to do the calculations because can find via matrix multiplication
        
        #taking the softmax first. 
        inp = tf.reshape(inp , (batch_size, -1, self.AA_num))
        inp = tf.nn.softmax(inp,axis=-1)

        #reshaping so that its flat again
        inp = tf.reshape(inp, (batch_size, -1))

        #pass into EVH
        evh = self.EVH(inp) #/ tf.cast(batch_size, tf.float32)
        return evh

    def energy(self, inp):
        
        # inputs are either a one hot or a probability distribution over sequences. 

        if inp.shape[-1] < (self.AA_num*self.L): # needs to be converted to one hot. 
            #print('being converted to one hot!', inp.shape)
            oh=[]
            for seq in inp: # there should be a way to more efficiently parallelize this
                oh.append(self.aa_to_oh(seq))
            oh=np.asarray(oh)
            inp = oh

        elif inp[0,:].sum() != self.AA_num:
            # need to convert to softmax: 
            inp = inp.reshape(inp.shape[0], -1, self.AA_num)
            inp = softmax(inp, axis=-1)
            inp = inp.reshape(inp.shape[0], -1)

        res = self.EVH(inp, use_tf=False) #np.tile(masker, 2)
        #print('the THIS IS IN THE NORMAL ENERGY CALL. ressss',res)
        return res
    
    def plot_energy(self, axis=None, temperature=1.0):
        """ Plots a histogram of different energies to the standard figure """
        
        plt.plot(np.arange(self.AA_num), energies, linewidth=3, color='black')
        plt.set_xlabel('proteins')
        axis.set_ylabel('energy')
        
        return energies


def hamiltonians(seqs, J, h):
    """
    Calculates the Hamiltonian of the global probability distribution P(A_1, ..., A_L)
    for a given sequence A_1,...,A_L from J_ij and h_i parameters
    Parameters
    ----------
    sequences : np.array
        Sequence matrix for which Hamiltonians will be computed
    J_ij: np.array
        L x L x num_symbols x num_symbols J_ij pair coupling parameter matrix
    h_i: np.array
        L x num_symbols h_i fields parameter matrix
    Returns
    -------
    np.array
        Float matrix of size len(sequences) x 3, where each row corresponds to the
        1) total Hamiltonian of sequence and the 2) J_ij and 3) h_i sub-sums
    """
    if len(seqs.shape) ==3:
        seqs = seqs.reshape(seqs.shape[0], -1) # in case the onehots arent already flat. 
    J = np.moveaxis(J, 1,2)
    J = J.reshape(seqs.shape[-1], seqs.shape[-1])
    h = h.reshape(-1)
    #print(J.shape, h.shape, seqs.shape)
    H = ((seqs.T*(J@seqs.T))/2 ).T.sum(1) + seqs@h
    return H # before it was an array of an array. 