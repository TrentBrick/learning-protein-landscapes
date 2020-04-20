import numpy as np
import time
import pickle 
import copy
from numba import njit, prange
from numba import jitclass          # import the decorator
from numba import int32, float32

class HillClimbing(object):

    def __init__(self, model, experiment_dir, x0=None, nwalkers=1, 
        AA_num=20, print_every=10, save_trajectory=False):
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
        self.AA_num = AA_num
        self.experiment_dir = experiment_dir
        self.save_trajectory = save_trajectory
        # set random seed. 
        np.random.seed(np.random.randint(0,3000,1)[0])
        
        if x0 is None: 
            x0 = self.make_rand_starters(self.nwalkers)
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
        self.energy_output = self.experiment_dir+'MCMC_trajectories_energies.txt'
        self.trajectory_output = self.experiment_dir+'MCMC_trajectories_seqs.txt'

    def make_rand_starters(self, batch_size):
        I = np.eye(self.AA_num)
        res = np.zeros( (batch_size, self.model.L*self.AA_num)) 
        for i in range(self.model.L):
            start = i*self.AA_num
            res[:,start:(start+self.AA_num)] = I[np.random.randint(0,self.AA_num,batch_size),:]
        return res
        
    #@njit(fastmath=True)
    def _hill_step(self):
        # propose all possible mutations to each sequence, select the best one. 
        for i in range(self.nwalkers):
            all_muts = self._all_mutations(self.x[i,:])
            self.x[i,:] = self._pick_best(all_muts, self.x[i,:], self.E[i], i)

    #@njit(fastmath=True)
    def _all_mutations(self, seq):
        # performs all mutations possible to a single sequence inputted as a numpy array
        all_muts = np.tile(seq, (self.AA_num*self.model.L,1))
        I = np.eye(self.AA_num)

        for i in range(self.model.L):
            start = i*self.AA_num
            end = start+self.AA_num
            all_muts[start:end, start:end ] = I 

        return all_muts

    #@njit(fastmath=True)
    def _pick_best(self, all_muts, seq, seq_energy, seq_ind):
        # score all of the sequences and pick the best one
        # if there are no better sequences we are at a local maxima and this sequence
        # should be recorded before restarting the search!
        E = self.model.hill_energy(all_muts)
        ind = np.argmax(E)
        if seq_energy == E[ind]: 
            # at a local max. Store this sequence and its energy!
            self.local_maxes.append( (seq, seq_energy, self.total_num_steps) )
            #reset this walker by returning a single new random sequence. 
            new_rand_seq = self.make_rand_starters(1)
            self.E[seq_ind] = self.model.hill_energy(new_rand_seq)
            return new_rand_seq
        else: 
            # return the highest scoring new sequence
            self.E[seq_ind] = E[ind]
            return all_muts[ind]

    def convert_to_position_ints(self):
        temp = []
        for s in self.x: 
            temp.append(np.where(s==1.0)[0])
        return np.asarray(temp).astype(np.uint)

    def run(self, nsteps=1, verbose=0):

        trajectory_output_file = open(self.trajectory_output, 'a')
        energy_output_file = open(self.energy_output, 'a')

        for i in range(nsteps):

            self._hill_step()
    
            if self.total_num_steps % self.print_every ==0:
                print('Run: ', self.total_num_steps, 'total steps. Have found in total:', len(self.local_maxes), 'local maxes')
                print('Current walker energies are:', self.E)
                print('='*10)

            if self.save_trajectory: 
                np.savetxt( trajectory_output_file, self.convert_to_position_ints(), fmt='%i')#, )
                np.savetxt( energy_output_file, self.E, fmt='%1.2f')
                
            self.total_num_steps += 1
        trajectory_output_file.close()
        energy_output_file.close()
            
        #res = np.asarray(res)
        return self.local_maxes