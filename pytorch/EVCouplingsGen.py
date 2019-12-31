# Need to make a model for the protein values. It will be 2D before moving 
# up the EV couplings
import numpy as np
import torch
from scipy.special import softmax
from torch.autograd import Variable
class EVCouplingsGenerator(object):

    def __init__(self, protein_length, aa_num, h ,J, device):
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
        self.h_torch = torch.unsqueeze(Variable(torch.from_numpy(h), requires_grad=False), 1).to(device)  
        J_torch = Variable(torch.from_numpy(J_numpy), requires_grad=False)
        #J_tf = tf.transpose(J, [0,2,1,3])
        #J_tf = tf.reshape(J_tf,(protein_length*aa_num,protein_length*aa_num))
        self.J_torch = J_torch.float().to(device)
        self.dim = protein_length*self.AA_num

    def aa_to_oh(self, labels):
        '''one-hot encode a numpy array'''
        O = np.reshape(np.eye(self.AA_num)[labels], (*labels.shape, self.AA_num))
        return(O)
        
    def discrete_energy_torch(self, inp):

        """
        Calculates in tensorflow the hamiltonian energy. 
        Takes in the softmax over the sequences generated from the neural network. 
        Then computes the expected energy over this softmax in a vectorized way. 
        Parameters
        ----------
        sequences : np.array
            Flattened protein sequences output from the neural network that have already been softmaxed batch_size x (protein_length x 20) 
        batch_size: int
            Size of the batch to be able to perform reshaping
        Returns
        -------
        tf.Tensor
            tf.float32 matrix of size batch_size x 1
        """

        #print('make sure no change!!! this is the h', self.h_torch)

        # applying the vectorized EVH loss: 
        h_val = torch.matmul(inp, self.h_torch )
        #h_val = tf.Print(h_val, [h_val], 'h value')

        j_val = torch.unsqueeze( torch.sum(inp * torch.matmul(inp, self.J_torch), dim=-1) /2, 1)
        #j_val = tf.Print(j_val, [j_val], 'J value')
        evh = j_val + h_val
        #evh = tf.Print(evh, [evh], 'evh value')
        return evh

    def energy(self, inp):
        """
        Calculates the Hamiltonian of the global probability distribution P(A_1, ..., A_L)
        for a given sequence A_1,...,A_L from J_ij and h_i parameters.
        Will argmax or softmax everything. 
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
            Float matrix of energy scores of size len(sequences) x 1
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