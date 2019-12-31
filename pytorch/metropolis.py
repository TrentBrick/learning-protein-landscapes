import numpy as np

class MetropolisHastings(object):

    def __init__(self, model, x0=None, temperature=1.0, noise=0.1,
                 burnin=0, stride=1, nwalkers=1, mapper=None, is_discrete=True, AA_num=20):
        """ Metropolis Monte-Carlo Simulation with Gaussian Proposal Steps

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
        self.AA_num = AA_num
        if mapper is None:
            class DummyMapper(object):
                def map(self, X):
                    return X
            mapper = DummyMapper()
        self.mapper = mapper
        if x0 is None: 
            x0 = self.make_rand_starters()
        self.reset(x0)

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
        else:
            self.x_prop = self.x + self.noise*np.random.randn(batch_size, self.x.shape[1])
        self.x_prop = self.mapper.map(self.x_prop)
        #print(self.x_prop.shape)
        self.E_prop = self.model.energy(self.x_prop)

    def _acceptance_step(self ):
        # acceptance step
        # flip the minus signs if trying to minimize
        acc = -np.log(np.random.rand()) > (self.E - self.E_prop) / self.temperature
        self.x = np.where(acc[:, None], self.x_prop, self.x)
        self.E = np.where(acc, self.E_prop, self.E)

    def reset(self, x0):
        # counters
        self.step = 0
        self.traj_ = []
        self.etraj_ = []

        # initial configuration
        self.x = np.tile(x0, (self.nwalkers, 1))
        self.x = self.mapper.map(self.x)
        self.E = self.model.energy(self.x)

        # save first frame if no burnin
        if self.burnin == 0:
            self.traj_.append(self.x)
            self.etraj_.append(self.E)

    def make_rand_starters(self, num=64):
        I = np.eye(self.AA_num)
        res = []
        for i in range(num):
            res.append(self.make_rand_start(I))
        res = np.asarray(res)
        return res.reshape(res.shape[0], -1)

    def make_rand_start(self, I):
        rand_starter = []
        for i in range(self.model.L):
            rand_starter.append( I[np.random.randint(0,self.AA_num,1),:] )
        rand_starter = np.asarray(rand_starter).flatten().reshape(1,-1)
        return rand_starter

    def run(self, nsteps=1, verbose=0):
        res = []
        for i in range(nsteps):
            self._proposal_step()
            self._acceptance_step()
            self.step += 1
            if verbose > 0 and i % verbose == 0:
                print('Step', i, '/', nsteps)
            if self.step > self.burnin and self.step % self.stride == 0:
                res.append(self.x)
                #self.traj_.append(self.x)
                #self.etraj_.append(self.E)
        res = np.asarray(res)
        return res.reshape(-1, res.shape[-1])

    '''@property
    def trajs(self):
        """ Returns a list of trajectories, one trajectory for each walker """
        T = np.array(self.traj_).astype(np.float32)
        return [T[:, i, :] for i in range(T.shape[1])]

    @property
    def traj(self):
        return self.trajs[0]

    @property
    def etrajs(self):
        """ Returns a list of energy trajectories, one trajectory for each walker """
        E = np.array(self.etraj_)
        return [E[:, i] for i in range(E.shape[1])]

    @property
    def etraj(self):
        return self.etrajs[0]'''