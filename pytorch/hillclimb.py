import numpy as np
import time
import pickle 
import copy
from numba import njit, prange
from numba import jitclass          # import the decorator
from numba import int32, float32

class HillClimbing(object):

    def __init__(self, model, experiment_dir, x0=None, nwalkers=1, 
        print_every=10, save_trajectory=False):
        """ ''' Uses discrete hillclimbing from a starting position. 
        Implemented to use batch scoring of the PFP and able to run multiple chains.  '''

        Parameters
        ----------
        model : Energy model
            Energy model object, must provide the function energy(x)
        x0 : [array]
            Initial configuration. If none then it will create random starter sequences
        nwalkers : int
            Number of parallel walkers
        mapper : Mapper object
            Object with function map(X), e.g. to remove permutation
            If given will be applied to each accepted configuration

        """
        self.model = model
        self.nwalkers = nwalkers
        self.AA_num = self.model.AA_num
        self.L = self.model.L
        self.experiment_dir = experiment_dir
        self.save_trajectory = save_trajectory
        
        if x0 is None: 
            x0 = self.make_rand_starters(self.nwalkers, self.AA_num, self.L)
            self.x = x0
        else: 
            print('NB!')
            print( "Providing sequences. Setting nwalkers to equal the number of sequences given.")
            self.nwalkers = len(x0) 
            self.x = copy.deepcopy(x0) # ensures we dont edit the original input sequence.
        self.x = self.x.astype(np.uint)
        self.E = self.model.hill_energy(self.x)
        print(x0.shape)
        self.local_maxes =[]
        self.total_num_steps = 0 #useful if want to run more steps across multiple run() calls
        self.print_every = print_every

    @staticmethod
    @njit(fastmath=True)
    def make_rand_starters(batch_size, AA_num, L):
        I = np.eye( AA_num)
        res = np.zeros( (batch_size,  L* AA_num)) 
        for i in range(L):
            start = i* AA_num
            res[:,start:(start+ AA_num)] = I[np.random.randint(0, AA_num,batch_size),:]
        return res
        
    @staticmethod
    #@njit(fastmath=True)
    def _all_mutations(seqs, AA_num, L):
        # performs all mutations possible to a single sequence inputted as a numpy array
        #all_muts =  np.tile(seq, (AA_num*L,1)) #np.ones((AA_num*L,AA_num*L)) * seq
        
        # batch mutations of all. 
        all_muts = np.tile(np.expand_dims(seqs,1), (1, AA_num*L,1))
        #all_muts = np.ones((AA_num*L,AA_num*L)) * np.expand_dim(seqs,1)
        I = np.eye(AA_num)

        for i in range(L):
            start = i*AA_num
            end = start+AA_num
            all_muts[:, start:end, start:end ] = I 

        return all_muts

    def _hill_step(self):
        # propose all possible mutations to each sequence, select the best one. 
        # (nwalker x AA*L x AA*L) each sequence, then all possible mutations of it
        self._pick_best( self._all_mutations(self.x, self.AA_num, self.L) )

    def _pick_best(self, all_muts):
        # score all of the sequences and pick the best one
        # if there are no better sequences we are at a local maxima and this sequence
        # should be recorded before restarting the search!
        E = self.model.hill_energy(all_muts, full_batch=True)
        max_inds = np.argmax(E, -1)
        for i in range(self.nwalkers):
            max_ind = max_inds[i]
            if self.E[i] == E[i, max_ind]:
                self.local_maxes.append( (self.x[i,:], self.E[i], self.total_num_steps) )

                new_rand_seq = self.make_rand_starters(1, self.AA_num, self.L)
                self.E[i] = self.model.hill_energy(new_rand_seq)
                self.x[i,:] = new_rand_seq
            else: 
                self.E[i] = E[i, max_ind]
                self.x[i,:] = all_muts[i, max_ind, :]

    def convert_to_position_ints(self):
        temp = []
        for s in self.x: 
            temp.append(np.where(s==1.0)[0])
        return np.asarray(temp).astype(np.uint)

    def run(self, nsteps=1, verbose=0):

        for i in range(nsteps):

            self._hill_step()
    
            if self.total_num_steps % self.print_every ==0:
                print('Run: ', self.total_num_steps, 'total steps. Have found in total:', len(self.local_maxes), 'local maxes')
                print('Current walker energies are:', self.E)
                print('='*10)

            if self.save_trajectory: 
                # save out the current sequences and energies that that the whole trajectory can later be plotted.
                np.savetxt( open(self.experiment_dir+'hill_climb_trajectories_seqs.txt', 'a'), self.convert_to_position_ints())
                np.savetxt( open(self.experiment_dir+'hill_climb_trajectories_energies.txt', 'a'), np.round_(self.E, decimals=))

            self.total_num_steps += 1
            
        #res = np.asarray(res)
        return self.local_maxes

if __name__=='__main__':

    start_time = time.time()

    params = {'protein_length':6, 'is_discrete':True, 
    'gaussian_cov_noise':None, 'nwalkers':64, 'print_every':50,
    'experiment_name':'protein_length6_50K_steps'}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading in EVCouplings model 
    focus_seqs = read_fa('EVCouplingsStuff/DYR_ECOLI_1_b0.5.a2m_trimmed.fa')
    evc_model = CouplingsModel('EVCouplingsStuff/DYR.model')

    # extracting the model parameters used to determine the evolutionary hamiltonian
    h = evc_model.h_i
    J = evc_model.J_ij

    if params['protein_length'] > 0:
        h = h[0:params['protein_length'], :]
        J = J[0:params['protein_length'], 0:params['protein_length'], :,:]
            
    # processing the natural sequences: 
    # first by converting amino acids into integers and also onehots. 
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
    oh=np.asarray(oh) # of shape: [batch x L x AA]
    N = oh.shape[0] # batch size
    L = oh.shape[1] # length of the protein
    
    print('number and dimensions of the natural sequences', oh.shape)

    # loading in the environment class, used to score the evolutionary hamiltonians
    gen_model = EVCouplingsGenerator(L, AA, h, J, device, params['is_discrete'], gaussian_cov_noise = params['gaussian_cov_noise'])

    hill_climber = HillClimbing(gen_model, params['experiment_name'], save_trajectory=False, 
        nwalkers=params['nwalkers'], print_every=params['print_every'])

    local_maxes = hill_climber.run(50)
    


    