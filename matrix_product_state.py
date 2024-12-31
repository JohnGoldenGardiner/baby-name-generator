import numpy as np

class MatrixProductState:
    def __init__(self, num_tensors, site_dims: list | int, verbose=0):
        if isinstance(site_dims, int):
            self.site_dims = [site_dims]*num_tensors
        elif isinstance(site_dims, list):
            if len(site_dims) != num_tensors:
                raise ValueError(f'Length of `site_dims` must match the number of tensors {num_tensors}')
            self.site_dims = site_dims

        self.verbose = verbose

        self.num_tensors = num_tensors
        self.ranks = [2]*(num_tensors - 1)

        self.cutoff = 1e-3
        self.max_rank = 50

        first_tensor = [np.random.random(size=(1, self.site_dims[0], 2))]
        last_tensor = [np.random.random(size=(2, self.site_dims[-1], 1))]
        interior_tensors = [np.random.random(size=(2, site_dim, 2)) for site_dim in self.site_dims[1:-1]]
        self.tensors = first_tensor + interior_tensors + last_tensor

        self.merged = False # flag used to keep self.merge_neighbors idempotent

        self.dataset_names = set()
        self.data = {}
        self.environments = {}
        self.losses = {}
        
        self.batch_size = 32
        self.learning_rate = 1e-5

        self.canonicalize()


    def set_tensors(self, tensors):
        if len(tensors) != self.num_tensors:
            raise ValueError(f'Number of tensors must be {self.num_tensors}')
        
        left_ranks = []
        right_ranks = []
        site_dims = []
        for tensor in tensors:
            left_rank, site_dim, right_rank = tensor.shape
            left_ranks.append(left_rank)
            right_ranks.append(right_rank)
            site_dims.append(site_dim)

        if left_ranks[0] != 1:
            raise ValueError('First axis of first tensor should have dimension 1')
        if right_ranks[-1] != 1:
            raise ValueError('Last axis of last tensor should have dimension 1')
        if left_ranks[1:] != right_ranks[:-1]:
            raise ValueError('Inconsistent dimensions of input tensors')
        if site_dims != self.site_dims:
            raise ValueError(f'Input site dimensions must be {self.site_dims}')

        self.tensors = tensors.copy()
        self.ranks = left_ranks[1:]

    
    def canonicalize(self):
        """
        Canonicalize so that canonical center is at rightmost tensor.
        """

        # make all tensors but the last into (left-facing) isometries
        self.canonical_center = 0
        for _ in range(self.num_tensors - 1):
            self.merge_neighbors()
            self.split_merged()

        # normalize so mps has norm 1
        self.tensors[-1] /= np.sum(self.tensors[-1]**2)**0.5


    def load_data(self, data, dataset_name='train'):
        self.dataset_names.add(dataset_name)
        self.data[dataset_name] = data

        self.calculate_environments(dataset_name)
        
        loss = self.get_loss(dataset_name)
        self.losses[dataset_name] = [loss]


    def calculate_environments(self, dataset_name='train'):
        """
        Calculate and store environment vectors for the given data.

        Note: for later convenience the k-th environment corresponds to 
        the edge *to the left* of the k-th tensor. (This is so that we can 
        keep trivial environments at the ends to avoid edge cases.) For 
        example, the zeroth environment is a trivial environment and the 
        environment corresponding to the edge between tensors 0 and 1 is 
        environments[1]. Also the num_tensors-th environment is a trivial 
        environment to the right of the last tensor (the num_tensors - 1 
        tensor)
        
        Also note that left environments have shape (data.shape[0], rank) 
        while right environments have shape (rank, data.shape[0]). This 
        is also for convenience. (It avoids some transposes later.)
        """
        data = self.data[dataset_name]

        if self.canonical_center != self.num_tensors - 1:
            self.canonicalize()

        # first environment, a trivial left environment
        environments = [np.ones((data.shape[0], 1))]
        # non-trivial environments
        for i in range(0, self.num_tensors - 1):
            environment = np.einsum('jk,kjm->jm', environments[-1], self.tensors[i][:, data[:, i], :])
            environments.append(environment)
        # an additional (trivial) right environment to simplify edge cases
        environments.append(np.ones((1, data.shape[0])))

        self.environments[dataset_name] = environments
        

    def update_environments(self, center_right=True):
        """
        Calculate new environment for rebuilt edge. When a merged tensor 
        is split, the new edge needs a corresponding environment vector 
        for every data point. If the canonical center is made the right tensor 
        in the split (i.e. center_right=True) the environment represents 
        the result of matrices multiplied from the left to the canonical 
        center. If the new canonical center is the left tensor after the 
        split (in which case center_right=False) the environment represents 
        the result of multiplying matrices from the right leftward toward 
        the canonical center.

        Note: this should be called *after* the canonical center has been 
        moved to its new spot, so center_right=True indicates the canonical 
        center is to the right of the environment vector to be rebuilt and 
        center_right=False indicates that the canonical center is to the 
        left of the environment vector that is to be rebuilt.
        """
        if center_right: # rebuild the environment vector to the left of the canonical center
            for dataset_name in self.dataset_names:
                environments = self.environments[dataset_name]
                data_element = self.data[dataset_name][:, self.canonical_center - 1]
                mat = self.tensors[self.canonical_center - 1][:, data_element, :]
                environment = np.einsum('jk,kjm->jm', environments[self.canonical_center - 1], mat)
                environments[self.canonical_center] = environment

        else: # rebuild environment vector to the right of the canonical center
            for dataset_name in self.dataset_names:
                environments = self.environments[dataset_name]
                data_element = self.data[dataset_name][:, self.canonical_center + 1]
                mat = self.tensors[self.canonical_center + 1][:, data_element, :]
                environment = np.einsum('mjk,kj->mj', mat, environments[self.canonical_center + 2])
                environments[self.canonical_center + 1] = environment


    def merge_neighbors(self, merge_right=True):
        if not self.merged:
            if self.canonical_center == 0 and not merge_right:
                raise ValueError('Canonical center is the leftmost tensor. There is no tensor to the left to merge with')
            if self.canonical_center == self.num_tensors - 1 and merge_right:
                raise ValueError('Canonical center is the rightmost tensor. There is no tensor to the right to merge with')
                                
            if merge_right:
                mat1 = self.tensors[self.canonical_center]
                mat2 = self.tensors[self.canonical_center + 1]
            else:
                mat1 = self.tensors[self.canonical_center - 1]
                mat2 = self.tensors[self.canonical_center]
                self.canonical_center -=1 # keep the convention that the canonical center is the left of the two merged tensors
            self.merged_tensor = np.einsum('ijk,kmn->ijmn', mat1, mat2)

            self.merged = True # add this flag to keep this method idempotent

        

    def split_merged(self, center_right=True):
        if self.merged:

            d1, d2, d3, d4 = self.merged_tensor.shape
            try:
                u, s, v_dag = np.linalg.svd(self.merged_tensor.reshape((d1*d2, d3*d4)))
            except np.linalg.LinAlgError as e:
                if np.isnan(self.merged_tensor).any():
                    raise ValueError('Merged tensor contains NaN')
                else:
                    print(f'SVD likely did not converge.')
                raise np.linalg.LinAlgError(e)

            new_rank = 0
            for sigma in s:
                if new_rank < self.max_rank and sigma/s[0] > self.cutoff:
                    new_rank += 1
                else:
                    break

            s = s[:new_rank]
            u = u[:, :new_rank]
            v_dag = v_dag[:new_rank, :]

            self.ranks[self.canonical_center] = new_rank

            if center_right:
                self.tensors[self.canonical_center] = u.reshape((d1, d2, new_rank))
                self.tensors[self.canonical_center + 1] = (s[:, np.newaxis]*v_dag).reshape((new_rank, d3, d4))
                self.canonical_center += 1

                self.update_environments()

            else:
                self.tensors[self.canonical_center] = (u*s).reshape((d1, d2, new_rank))
                self.tensors[self.canonical_center + 1] = v_dag.reshape((new_rank, d3, d4))

                self.update_environments(center_right=False)

            self.merged = False


    def get_loss(self, dataset_name='train'):
        data = self.data[dataset_name]
        left_environment = self.environments[dataset_name][self.canonical_center]
        right_environment = self.environments[dataset_name][self.canonical_center + 1]
        mat = self.tensors[self.canonical_center][:, data[:, self.canonical_center], :]
        amplitudes = np.einsum('ij,jik,ki->i', left_environment, mat, right_environment)
        loss = -2*np.mean(np.log(np.abs(amplitudes)), axis=0)
        return loss
        

    def get_gradient(self):
        """
        Gradient calculated on a batch of randomly chosen datapoints
        """
        num_data_points = self.data['train'].shape[0]
        batch_inds = np.random.choice(num_data_points, size=self.batch_size, replace=True)
        left_site_val = self.data['train'][batch_inds][:, self.canonical_center]
        right_site_val = self.data['train'][batch_inds][:, self.canonical_center + 1]
        left_environment = self.environments['train'][self.canonical_center][batch_inds]
        right_environment = self.environments['train'][self.canonical_center + 2][:, batch_inds]
        # mat = self.tensors[self.canonical_center][:, site_vals[:, 0], :]
        mat = self.merged_tensor[:, left_site_val, right_site_val, :]
        amplitudes = np.einsum('ij,jik,ki->i', left_environment, mat, right_environment)

        mats = -2*np.einsum('ij,i,ki->jik', left_environment, 1/amplitudes, right_environment)

        gradient = np.zeros_like(self.merged_tensor)
        np.add.at(gradient, (slice(None), left_site_val, right_site_val, slice(None)), mats)

        if np.isnan(gradient).any():
            print('Gradient has NaNs')

        return gradient
        

        

    def update_merged(self):
        self.merged_tensor -= self.learning_rate*self.get_gradient()
        self.merged_tensor /= np.sum(self.merged_tensor**2)**0.5 # keep the merged tensor (and hence the whole MPS) normed to 1

    def training_sweep(self, sweep_right=True):
        """
        Do gradient descent updating on all tensors to the right (or left, if `sweep_right` is False) of the current canonical center
        """
        if sweep_right:
            for i in range(self.canonical_center, self.num_tensors - 1):
                self.merge_neighbors()
                self.update_merged()
                self.split_merged()
        else:
            for _ in range(self.canonical_center, 0, -1):
                self.merge_neighbors(merge_right=False)
                self.update_merged()
                self.split_merged(center_right=False)

        for dataset_name in self.dataset_names:
            loss = self.get_loss(dataset_name=dataset_name)
            self.losses[dataset_name].append(loss)

    def train(self, num_sweeps):
        for i in range(num_sweeps):
            # alternate sweeping direction, starting left
            self.training_sweep(sweep_right=bool(i%2))
            if self.verbose:
                print(f'Finished sweep {i}. Sweep direction: {"right" if i%2 else "left"}')
                print(f'Ranks: {self.ranks}')
                latest_losses = {
                    dataset_name: losses[-1]
                    for dataset_name, losses in self.losses.items()
                }
                print(f'Losses: {latest_losses}')


    def sample(self, num_samples):
        if self.canonical_center != self.num_tensors - 1:
            self.canonicalize()

        samples = np.empty((num_samples, self.num_tensors), dtype=int)

        environment = np.ones((1, num_samples))
        for i in range(self.num_tensors - 1, -1, -1):
            mat = np.einsum('ijk,km->ijm', self.tensors[i], environment)
            probs = np.sum(mat**2, axis=0).T # rows are probability distributions
            for j in range(num_samples):
                samples[j, i] = np.random.choice(self.site_dims[i], p=probs[j])
            norms = probs[np.arange(num_samples), samples[:, i]]**0.5
            environment = mat[:, samples[:, i], np.arange(num_samples)]
            environment /= norms

        return samples




class MPSForest:
    """
    An ensemble of MPS generative models. Sampling is from the equal weighted 
    convex combination of the constituent probability distributions. 
    Operations are done on the models in the ensemble in parallel
    """
    def __init__(self, num_models, num_tensors, site_dims: list | int, verbose=0):
        if isinstance(site_dims, int):
            self.site_dims = [site_dims]*num_tensors
        elif isinstance(site_dims, list):
            if len(site_dims) != num_tensors:
                raise ValueError(f'Length of `site_dims` must match the number of tensors {num_tensors}')
            self.site_dims = site_dims

        self.verbose = verbose

        self.num_models = num_models
        self.num_tensors = num_tensors
        self.ranks = [2]*(num_tensors - 1)

        self.cutoff = 1e-3
        self.max_rank = 50

        # the first dimension runs over the different MPSs in the ensemble
        first_tensor = [np.random.random(size=(num_models, 1, self.site_dims[0], 2))]
        last_tensor = [np.random.random(size=(num_models, 2, self.site_dims[-1], 1))]
        interior_tensors = [np.random.random(size=(num_models, 2, site_dim, 2)) for site_dim in self.site_dims[1:-1]]
        self.tensors = first_tensor + interior_tensors + last_tensor

        self.merged = False # flag used to keep self.merge_neighbors idempotent

        self.dataset_names = set()
        self.data = {}
        self.environments = {}
        self.losses = {}
        
        self.batch_size = 32
        self.learning_rate = 1e-5

        self.canonicalize()

    
    def canonicalize(self):
        """
        Canonicalize so that canonical center is at rightmost tensor.
        """

        # make all tensors but the last into (left-facing) isometries
        self.canonical_center = 0
        for _ in range(self.num_tensors - 1):
            self.merge_neighbors()
            self.split_merged()

        # normalize so MPSs all have norm 1
        self.tensors[-1] /= (np.sum(self.tensors[-1]**2, axis=(1, 2, 3))**0.5).reshape((-1, 1, 1, 1))


    def load_data(self, data, dataset_name='train'):
        """
        We randomly sample from the provided data to get different fixed 
        datasets for each copy of the model
        """
        # TODO: get different fixed datasets for each copy of the model
        self.dataset_names.add(dataset_name)
        self.data[dataset_name] = data

        self.calculate_environments(dataset_name)
        
        loss = self.get_loss(dataset_name)
        self.losses[dataset_name] = [loss]


    def calculate_environments(self, dataset_name='train'):
        """
        Calculate and store environment vectors for the given data.

        Note: for later convenience the k-th environment corresponds to 
        the edge *to the left* of the k-th tensor. (This is so that we can 
        keep trivial environments at the ends to avoid edge cases.) For 
        example, the zeroth environment is a trivial environment and the 
        environment corresponding to the edge between tensors 0 and 1 is 
        environments[1]. Also the num_tensors-th environment is a trivial 
        environment to the right of the last tensor (the num_tensors - 1 
        tensor)
        
        Also note that left environments have shape (data.shape[0], rank) 
        while right environments have shape (rank, data.shape[0]). This 
        is also for convenience. (It avoids some transposes later.)
        """
        data = self.data[dataset_name]

        if self.canonical_center != self.num_tensors - 1:
            self.canonicalize()

        # first environment, a trivial left environment
        environments = [np.ones((self.num_models, data.shape[0], 1))]
        # non-trivial environments
        for i in range(0, self.num_tensors - 1):
            environment = np.einsum('njk,nkjm->njm', environments[-1], self.tensors[i][:, :, data[:, i], :])
            environments.append(environment)
        # an additional (trivial) right environment to simplify edge cases
        environments.append(np.ones((self.num_models, 1, data.shape[0])))

        self.environments[dataset_name] = environments
        

    def update_environments(self, center_right=True):
        """
        Calculate new environment for rebuilt edge. When a merged tensor 
        is split, the new edge needs a corresponding environment vector 
        for every data point. If the canonical center is made the right tensor 
        in the split (i.e. center_right=True) the environment represents 
        the result of matrices multiplied from the left to the canonical 
        center. If the new canonical center is the left tensor after the 
        split (in which case center_right=False) the environment represents 
        the result of multiplying matrices from the right leftward toward 
        the canonical center.

        Note: this should be called *after* the canonical center has been 
        moved to its new spot, so center_right=True indicates the canonical 
        center is to the right of the environment vector to be rebuilt and 
        center_right=False indicates that the canonical center is to the 
        left of the environment vector that is to be rebuilt.
        """
        if center_right: # rebuild the environment vector to the left of the canonical center
            for dataset_name in self.dataset_names:
                environments = self.environments[dataset_name]
                data_element = self.data[dataset_name][:, self.canonical_center - 1]
                mat = self.tensors[self.canonical_center - 1][:, :, data_element, :]
                environment = np.einsum('njk,nkjm->njm', environments[self.canonical_center - 1], mat)
                environments[self.canonical_center] = environment

        else: # rebuild environment vector to the right of the canonical center
            for dataset_name in self.dataset_names:
                environments = self.environments[dataset_name]
                data_element = self.data[dataset_name][:, self.canonical_center + 1]
                mat = self.tensors[self.canonical_center + 1][:, :, data_element, :]
                environment = np.einsum('nmjk,nkj->nmj', mat, environments[self.canonical_center + 2])
                environments[self.canonical_center + 1] = environment


    def merge_neighbors(self, merge_right=True):
        if not self.merged:
            if self.canonical_center == 0 and not merge_right:
                raise ValueError('Canonical center is the leftmost tensor. There is no tensor to the left to merge with')
            if self.canonical_center == self.num_tensors - 1 and merge_right:
                raise ValueError('Canonical center is the rightmost tensor. There is no tensor to the right to merge with')
                                
            if merge_right:
                mat1 = self.tensors[self.canonical_center]
                mat2 = self.tensors[self.canonical_center + 1]
            else:
                mat1 = self.tensors[self.canonical_center - 1]
                mat2 = self.tensors[self.canonical_center]
                self.canonical_center -=1 # keep the convention that the canonical center is the left of the two merged tensors
            self.merged_tensor = np.einsum('nijk,nkml->nijml', mat1, mat2)

            self.merged = True # add this flag to keep this method idempotent

        

    def split_merged(self, center_right=True):
        if self.merged:

            num_models, d1, d2, d3, d4 = self.merged_tensor.shape
            try:
                u, s, v_dag = np.linalg.svd(self.merged_tensor.reshape((num_models, d1*d2, d3*d4)))
            except np.linalg.LinAlgError as e:
                print(e)

            max_rank = 0
            for model_sigmas in s:
                new_rank = 0
                for sigma in model_sigmas:
                    if new_rank < self.max_rank and sigma/model_sigmas[0] > self.cutoff:
                        new_rank += 1
                    else:
                        break
                max_rank = max(max_rank, new_rank)

            s = s[:, :new_rank]
            u = u[:, :, :new_rank]
            v_dag = v_dag[:, :new_rank, :]

            self.ranks[self.canonical_center] = new_rank

            if center_right:
                self.tensors[self.canonical_center] = u.reshape((self.num_models, d1, d2, new_rank))
                self.tensors[self.canonical_center + 1] = (s[:, :, np.newaxis]*v_dag).reshape((self.num_models, new_rank, d3, d4))
                self.canonical_center += 1

                self.update_environments()

            else:
                self.tensors[self.canonical_center] = (u*s[:, np.newaxis, :]).reshape((self.num_models, d1, d2, new_rank))
                self.tensors[self.canonical_center + 1] = v_dag.reshape((self.num_models, new_rank, d3, d4))

                self.update_environments(center_right=False)

            self.merged = False


    def get_loss(self, dataset_name='train'):
        data = self.data[dataset_name]
        left_environment = self.environments[dataset_name][self.canonical_center]
        right_environment = self.environments[dataset_name][self.canonical_center + 1]
        mat = self.tensors[self.canonical_center][:, :, data[:, self.canonical_center], :]
        amplitudes = np.einsum('nij,njik,nki->ni', left_environment, mat, right_environment)
        # loss = -2*np.mean(np.log(np.abs(amplitudes)), axis=1)
        probabilities = np.mean(amplitudes**2, axis=0) # take mixture of probabilities to get new probability
        loss = -np.mean(np.log(np.abs(probabilities)), axis=0)
        return loss
        

    def get_gradient(self):
        """
        Gradient calculated on a batch of randomly chosen datapoints
        """
        num_data_points = self.data['train'].shape[0]
        batch_inds = np.random.choice(num_data_points, size=(self.num_models, self.batch_size), replace=True)
        left_site_val = self.data['train'][batch_inds][:, :, self.canonical_center]
        right_site_val = self.data['train'][batch_inds][:, :, self.canonical_center + 1]
        left_environment = self.environments['train'][self.canonical_center][np.arange(self.num_models)[:, np.newaxis], batch_inds, :] # TODO
        right_environment = self.environments['train'][self.canonical_center + 2][np.arange(self.num_models)[:, np.newaxis], :, batch_inds] # The advanced indexing here switches the ordering of the dimensions
        mat = self.merged_tensor[np.arange(self.num_models)[:, np.newaxis], :, left_site_val, right_site_val, :] # The advanced indexing here switches the ordering of the dimensions
        amplitudes = np.einsum('nij,nijk,nik->ni', left_environment, mat, right_environment)

        mats = -2*np.einsum('nij,ni,nik->nijk', left_environment, 1/amplitudes, right_environment)

        gradient = np.zeros_like(self.merged_tensor)
        np.add.at(gradient, (np.arange(2)[:, np.newaxis], slice(None), left_site_val, right_site_val, slice(None)), mats) # Note the advanced indexing here switches the ordering of the dimensions

        return gradient
        

        

    def update_merged(self):
        self.merged_tensor -= self.learning_rate*self.get_gradient()
        self.merged_tensor /= (np.sum(self.merged_tensor**2, axis=(1, 2, 3, 4))**0.5).reshape((-1, 1, 1, 1, 1)) # keep the merged tensors (and hence each whole MPS) normed to 1

    def training_sweep(self, sweep_right=True):
        """
        Do gradient descent updating on all tensors to the right (or left, if `sweep_right` is False) of the current canonical center
        """
        if sweep_right:
            for i in range(self.canonical_center, self.num_tensors - 1):
                self.merge_neighbors()
                self.update_merged()
                self.split_merged()
        else:
            for _ in range(self.canonical_center, 0, -1):
                self.merge_neighbors(merge_right=False)
                self.update_merged()
                self.split_merged(center_right=False)

        for dataset_name in self.dataset_names:
            loss = self.get_loss(dataset_name=dataset_name)
            self.losses[dataset_name].append(loss)

    def train(self, num_sweeps):
        for i in range(num_sweeps):
            # alternate sweeping direction, starting left
            self.training_sweep(sweep_right=bool(i%2))
            if self.verbose:
                print(f'Finished sweep {i}. Sweep direction: {"right" if i%2 else "left"}')
                print(f'Ranks: {self.ranks}')
                latest_losses = {
                    dataset_name: losses[-1]
                    for dataset_name, losses in self.losses.items()
                }
                print(f'Losses: {latest_losses}')


    def sample(self, num_samples):
        if self.canonical_center != self.num_tensors - 1:
            self.canonicalize()

        samples = np.empty((num_samples, self.num_tensors), dtype=int)

        environment = np.ones((self.num_models, 1, num_samples))
        for i in range(self.num_tensors - 1, -1, -1):
            mat = np.einsum('nijk,nkm->nijm', self.tensors[i], environment)
            probs = np.mean(np.sum(mat**2, axis=1), axis=0).T # take mixture of the probability distributions from the separate models
            for j in range(num_samples):
                samples[j, i] = np.random.choice(self.site_dims[i], p=probs[j])
            norms = probs[np.arange(num_samples), samples[:, i]]**0.5
            environment = mat[:, :, samples[:, i], np.arange(num_samples)]
            environment /= norms

        return samples
