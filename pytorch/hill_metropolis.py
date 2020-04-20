import numpy as np
import copy 
class HillMetropolis(object):

    def __init__(self, model, experiment_dir, random_seed, hill_or_MCMC, x0=None, temperature=1.0, noise=0.1,
                 burnin=0, stride=1, print_every=10, save_trajectory=True, 
                  nwalkers=1, mapper=None, is_discrete=True):
        """ Metropolis Monte-Carlo Simulation or Greedy Hill Climbing.

        Parameters
        ----------
        model : Energy model
            Energy model object, must provide the function energy(x)
        x0 : [array]
            Initial configuration
        noise : float
            Noise intensity, standard deviation of Gaussian proposal step
        temperatures : float or array
            Temperature. By default (1.0) the energy is interpreted in reduced units.
            When given an array, its length must correspond to nwalkers, then the walkers
            are simulated at different temperatures.
        burnin : int
            Number of burn-in steps that will not be saved
        stride : int
            Every so many steps will be saved
        nwalkers : int
            Number of parallel walkers
        mapper : Mapper object
            Object with function map(X), e.g. to remove permutation.
            If given will be applied to each accepted configuration.

        """
        self.model = model
        self.noise = noise
        self.temperature = temperature
        self.burnin = burnin
        self.stride = stride
        self.nwalkers = nwalkers
        self.is_discrete = is_discrete
        self.AA_num = model.AA_num
        self.print_every = print_every
        self.save_trajectory = save_trajectory
        self.total_num_steps = 0
        self.experiment_dir = experiment_dir
        np.random.seed(random_seed)
        self.hill_or_MCMC = hill_or_MCMC.lower()
        self.energy_output = self.experiment_dir+self.hill_or_MCMC+'_trajectories_energies.txt'
        self.trajectory_output = self.experiment_dir+self.hill_or_MCMC+'_trajectories_seqs.txt'
        
        
        '''if mapper is None:
            class DummyMapper(object):
                def map(self, X):
                    return X
            mapper = DummyMapper()
        self.mapper = mapper'''
        if x0 is None: 
            x0 = self.make_rand_starters(self.nwalkers)
            self.x = x0
        else: 
            print('NB!')
            print( "Providing sequences. Setting nwalkers to equal the number of sequences given.")
            self.nwalkers = len(x0) 
            self.x = copy.deepcopy(x0) # ensures we dont edit the original input sequence.

        self.local_maxes =[]
        # initial configuration
        
        #self.x = self.mapper.map(self.x)
        if self.is_discrete:
            self.E = self.model.hill_energy(self.x)
        else: 
            self.E = self.model.energy(self.x)

    def _proposal_step(self):
        # proposal step
        batch_size = self.x.shape[0]
        if self.is_discrete:
            self.x_prop = np.copy(self.x)
            #print('xprop before mut', self.x_prop)
            #will swap out one of the onehots randomly. 
            rand_inds = np.random.randint(0, (self.x.shape[1]//self.AA_num), batch_size) # random positions for each seq in batch
            starts = rand_inds * self.AA_num
            ends = starts+self.AA_num
            mutations = np.zeros((batch_size, self.AA_num))
            rand_muts = np.random.randint(0, self.AA_num, batch_size)
            mutations[np.arange(batch_size), rand_muts] = 1
            for i in range(batch_size):
                self.x_prop[i, starts[i]:ends[i]] = mutations[i]  

            #self.x_prop = self.mapper.map(self.x_prop)
            #print(self.x_prop.shape)
            self.E_prop = self.model.hill_energy(self.x_prop)
        else:
            self.x_prop = self.x + self.noise*np.random.randn(batch_size, self.x.shape[1])
            #self.x_prop = self.mapper.map(self.x_prop)
            #print(self.x_prop.shape)
            self.E_prop = self.model.energy(self.x_prop)

    def _acceptance_step(self ):
        # acceptance step
        # flip the minus signs if trying to minimize
        acc = -np.log(np.random.rand()) > (self.E - self.E_prop) / self.temperature
        self.x = np.where(acc[:, None], self.x_prop, self.x)
        self.E = np.where(acc, self.E_prop, self.E)

    '''def reset(self, x0):
        # counters
        self.step = 0

        # initial configuration
        self.x = np.tile(x0, (self.nwalkers, 1))
        #self.x = self.mapper.map(self.x)
        if self.is_discrete:
            self.E = self.model.energy(self.x, discrete_override=True)
        else: 
            self.E = self.model.energy(self.x)'''

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

    def _pick_best(self, all_muts, seq, seq_energy, seq_ind):
        # score all of the sequences and pick the best one
        # if there are no better sequences we are at a local maxima and this sequence
        # should be recorded before restarting the search!
        E = self.model.hill_energy(all_muts)
        ind = np.argmax(E)
        if seq_energy == E[ind]: 
            # at a local max. Store this sequence and its energy! Getting just where the ones are for a sparse representation. 
            self.local_maxes.append( (np.where(seq==1.0)[0], seq_energy, self.total_num_steps) )
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
        #res = []

        trajectory_output_file = open(self.trajectory_output, 'a')
        energy_output_file = open(self.energy_output, 'a')

        for i in range(nsteps):

            if self.hill_or_MCMC == 'mcmc':
                self._proposal_step()
                self._acceptance_step()
            elif self.hill_or_MCMC == 'hill':
                self._hill_step()

            if self.total_num_steps % self.print_every ==0:

                print('Run: ', self.total_num_steps, 'total steps.')
                if self.hill_or_MCMC == 'hill':
                    print('Have found in total:', len(self.local_maxes), 'local maxes')
                print('Current walker energies are:', self.E)
                print('='*10)

            if self.save_trajectory: 
                np.savetxt( trajectory_output_file, self.convert_to_position_ints(), fmt='%i')
                np.savetxt( energy_output_file, self.E, fmt='%1.2f')

            self.total_num_steps += 1
        trajectory_output_file.close()
        energy_output_file.close()

        if self.hill_or_MCMC == 'hill':
            return self.local_maxes