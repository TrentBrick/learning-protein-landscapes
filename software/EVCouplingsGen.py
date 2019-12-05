# Need to make a model for the protein values. It will be 2D before moving 
# up the EV couplings
import numpy as np
import tensorflow as tf
class EVCouplingsGenerator(object):

    def __init__(self, protein_length, aa_num, h ,J):
        # set parameters
        self.a2n = dict([(a, n) for n, a in enumerate('ACDEFGHIKLMNPQRSTVWY-')])
        self.AA_num = aa_num
        self.L = protein_length
        self.h = h
        self.J =J
        self.h_tf = tf.expand_dims(tf.reshape(tf.constant(h), [-1]), 1)  
        #tf.reshape(tf.constant(h), [-1])
        #tf.expand_dims(tf.reshape(tf.constant(h), [-1]), 1)
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
        
    def EVH(self, data, batch_size = 0, old_ham=False):
        
        if old_ham:
            if len(data.shape)==1: # putting brackets around it to pass into the function. 
                data = [data] # dont know if i need this or not!
                evh = np.asarray([hamiltonians(np.asarray(data), self.J, self.h)[0][0]]) # getting the first entry of the first array. 
            else: 
                evh = hamiltonians(data, self.J, self.h)[:,0]
        else: 

            h_val = tf.squeeze(tf.matmul(data, tf.expand_dims(self.h_tf, 1) ))
            h_val = tf.Print(h_val, [h_val], 'h value')

            j_val = tf.squeeze(tf.reduce_sum(data * tf.matmul(data, self.J_tf), axis=-1) /2) 
            j_val = tf.Print(j_val, [j_val], 'J value')
            evh = j_val + h_val
            evh = tf.Print(evh, [evh], 'evh value')
            evh = tf.expand_dims(evh, 1)#tf.reshape(evh, (batch_size, -1) )
            evh = tf.Print(evh, [evh], 'evh value')
            #evh = tf.reduce_sum(evh)
            
            
            '''evh = tf.squeeze(tf.reduce_sum(data * tf.matmul(data, self.J_tf), axis=-1) /2) + tf.squeeze(tf.matmul(data, self.h_tf))
            #evh =tf.Print(evh, [evh], 'printing the evh again ')
            evh = tf.reduce_sum(evh)'''
            
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

        # applying the vectorized EVH loss: 
        h_val = tf.matmul(inp, self.h_tf )
        #h_val = tf.Print(h_val, [h_val], 'h value')

        j_val = tf.expand_dims( tf.reduce_sum(inp * tf.matmul(inp, self.J_tf), axis=-1) /2, 1)
        #j_val = tf.Print(j_val, [j_val], 'J value')
        evh = j_val + h_val
        #evh = tf.Print(evh, [evh], 'evh value')
        return evh

    def energy(self, inp):
        #inputs here can be ints of sequences, onehots, pdfs that need to have a hard max. 
        #print('input to the energy function', inp.shape)
        if inp.shape[1] > self.L: # if it isnt a sequence of ints, then all to one hot.  
            #print('the input to oh is', inp)
            
            if len(inp)>2:
                inp = inp.reshape(inp.shape[0], -1, self.AA_num)
            else:
                inp = inp.reshape(-1,self.AA_num)
            
            inp = np.argmax(inp, axis=-1)
            #print('argmax', inp.shape)
            #print(inp)
            
        res = self.EVH(inp, old_ham=True) #np.tile(masker, 2)
        #print('the THIS IS IN THE NORMAL ENERGY CALL. ressss',res)
        return res
    
    def plot_energy(self, axis=None, temperature=1.0):
        """ Plots a histogram of different energies to the standard figure """
        
        plt.plot(np.arange(self.AA_num), energies, linewidth=3, color='black')
        plt.set_xlabel('proteins')
        axis.set_ylabel('energy')
        

        return energies

def exp_hamiltonians(seqs, J, h):
    if len(seqs.shape) ==3:
        seqs = seqs.reshape(seqs.shape[0], -1) # in case the onehots arent already flat. 
    J = np.moveaxis(J, 1,2)
    J = J.reshape(seqs.shape[-1], seqs.shape[-1])
    h = h.reshape(-1)
    #print(J.shape, h.shape, seqs.shape)
    H = ((seqs.T*(J@seqs.T))/2 ).T.sum(1) + seqs@h
    return H # before it was an array of an array. 


def hamiltonians(sequences, J_ij, h_i):
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
        Float matrix of size len(sequences) x 1
    """
    
    N, L = sequences.shape
    H = np.zeros((N, 3))
    for s in range(N):
        A = sequences[s,:]
        #print('the matrix A', A)
        hi_sum = 0.0
        Jij_sum = 0.0
        for i in range(L):
            #print('the value of A[i]', A[i])
            #print('a i as an int', int(A[i]))
            hi_sum += h_i[i, A[i]]
            for j in range(i + 1, L):
                Jij_sum += J_ij[i, j, A[i], A[j]]

        H[s] = [Jij_sum + hi_sum, Jij_sum, hi_sum]

    return H # before it was an array of an array. 