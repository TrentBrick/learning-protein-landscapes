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

        # modifying the J and h so that they can be used in a vectorized fashion
        J_numpy = np.moveaxis(J, 1,2)
        J_numpy = J_numpy.reshape(self.AA_num*self.L, self.AA_num*self.L)
        h = h.reshape(-1)

        self.h = h # numpy versions of the hamiltonian energy weights. 
        self.J =J_numpy
        self.h_tf = tf.expand_dims(tf.constant(h), 1)  
        J_tf = tf.constant(J)
        J_tf = tf.transpose(J, [0,2,1,3])
        J_tf = tf.reshape(J_tf,(protein_length*aa_num,protein_length*aa_num))
        self.J_tf = tf.cast(J_tf, tf.float32)
        self.dim = protein_length*self.AA_num

    def aa_to_oh(self, labels):
        '''one-hot encode a numpy array'''
        O = np.reshape(np.eye(self.AA_num)[labels], (*labels.shape, self.AA_num))
        return(O)
        
    def discrete_energy_tf(self, inp, batch_size):

        """
        Calculates in tensorflow the hamiltonian energy. 
        First computes the softmax over the possible sequences. 
        Then computes the expected energy over this softmax in a vectorized way. 
        Parameters
        ----------
        sequences : np.array
            Flattened protein sequences output from the neural network batch_size x (protein_length x 20) 
        batch_size: int
            Size of the batch to be able to perform reshaping
        Returns
        -------
        tf.Tensor
            tf.float32 matrix of size batch_size x 1
        """
        
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
        """
        Calculates the Hamiltonian of the global probability distribution P(A_1, ..., A_L)
        for a given sequence A_1,...,A_L from J_ij and h_i parameters.
        Will argmax everything. 
        Parameters
        ----------
        seqs : np.array
            Matrix of sequences of the following possible formats: 
            (i) flattened onehot batch_size x (protein_length*20)
            (ii) non flattened one hot batch_size x protein_length x 20
            (iii) monte carlo or neural network output that needs to be softmaxed
            (iv) integer encoding of the sequences that needs to be made into a one hot
        J: np.array
            L x L x num_symbols x num_symbols J_ij pair coupling parameter matrix
        h: np.array
            L x num_symbols h_i fields parameter matrix
        Returns
        -------
        np.array
            Float matrix of size len(sequences) x 1
        """

        # check what format the input is of. 
        # Convert it to either a softmax or argmax of shape batch_size x (protein_length*20)

        assert len(inp.shape) == 2 or len(inp.shape) == 3, "Strange input dimensions. Needs to have 2 or 3 dimensions."

        # if it is an integer encoding, convert to onehot. 
        if inp.shape[1] == self.L:
            oh=[]
            for seq in inp: # there should be a way to more efficiently parallelize this
                oh.append(self.aa_to_oh(seq))
            inp=np.asarray(oh) # this is of dimension batch_size x protein_length x 20

        # if it is monte carlo, need to softmax it. 
        # Everything else should be in a one hot format by now. 
        # the additions here are to deal with small amounts of numerical instability. 
        if len(inp.shape) == 3 and int(inp[:,0,:].sum()+0.01) != inp.shape[0]: # the first position for all of the sequences should be 1. so their sum equals the batch size. 
            inp = softmax(inp, axis=-1)
        elif len(inp.shape) == 2 and (inp[:,:self.AA_num].sum(axis=1)+0.01).astype(int).sum() != inp.shape[0]:
            inp = inp.reshape(inp.shape[0], self.L, self.AA_num)
            inp = softmax(inp, axis=-1) 
            #print('doing the softmax on this!!!', inp.shape)

        # flatten any unflat inputs: 
        if len(inp.shape) == 3: 
            inp = inp.reshape(inp.shape[0], -1)

        # is the input a onehot or softmax?? 
        assert (inp[:,:self.AA_num].sum(axis=1)+0.01).astype(int).sum() == inp.shape[0], "Either the softmax or onehot conversion has failed for this input: " + str(inp.shape)

        H = ((inp.T*(self.J@inp.T))/2 ).T.sum(1) + inp@self.h

        return H 

def hamiltonians(seqs, J, h):
    """
    Calculates the Hamiltonian of the global probability distribution P(A_1, ..., A_L)
    for a given sequence A_1,...,A_L from J_ij and h_i parameters
    Parameters
    ----------
    seqs : np.array
        Matrix of onehots either: (i) batch_size x protein_length x 20; or (ii) flattened so that batch_size x (protein_length x 20) 
    J: np.array
        L x L x num_symbols x num_symbols J_ij pair coupling parameter matrix
    h: np.array
        L x num_symbols h_i fields parameter matrix
    Returns
    -------
    np.array
        Float matrix of size len(sequences) x 1
    """
    
    if len(seqs.shape) ==3:
        seqs = seqs.reshape(seqs.shape[0], -1) # in case the onehots arent already flat. 
    J = np.moveaxis(J, 1,2)
    J = J.reshape(seqs.shape[-1], seqs.shape[-1])
    h = h.reshape(-1)
    #print(J.shape, h.shape, seqs.shape)
    H = ((seqs.T*(J@seqs.T))/2 ).T.sum(1) + seqs@h

    return H 