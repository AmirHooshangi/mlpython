from generic import Learner
from numpy import dot,ones,zeros,log,argmax,sum,sqrt,roots,array,diag,eye,pi,outer,newaxis,inf,nan,iscomplex

#from numpy import * #array, dot, ones, zeros, diag, log, pi, eye, hstack, vstack, exp
from numpy.linalg import inv, eigvals
from numpy.random.mtrand import RandomState
import sys, time

# The training set for these models should be an iterator over matrices, where
# each matrix is a sequence and each row is an observation (vector)
# from that sequence.


class LinearDynamicalSystem(Learner):
    """ 
    Linear Dynamical System (LDS)
 
    This is a standard linear dynamical system, trained by EM.

    Options:
    - 'n_epochs'
    - 'latent_size'
    - 'latent_covariance_matrix_regularizer'
    - 'input_covariance_matrix_regularizer'
    - 'latent_transition_matrix_regularizer'
    - 'input_transition_matrix_regularizer'
    - 'seed'

    Required metadata:
    - 'input_size'

    Reference: Pattern Recognition and Machine Learning
               Christopher M. Bishop
               http://research.microsoft.com/en-us/um/people/cmbishop/prml/
               Note: I tried to use the same notation. The only exception is that
                     I refer to the latent covariance matrix \Gamma as E here.
    """
    def __init__(   self,
                    n_epochs= sys.maxint, # Maximum number of iterations on the training set
                    latent_size = 10, # Size of the latent variable
                    latent_covariance_matrix_regularizer = 0,
                    input_covariance_matrix_regularizer = 0,
                    latent_transition_matrix_regularizer = 0,
                    input_transition_matrix_regularizer = 0,
                    seed = 1827,
                    ):
        self.epoch = 0
        self.n_epochs = n_epochs
        self.latent_size = latent_size
        self.seed = seed
        self.rng = RandomState(seed)
        self.latent_covariance_matrix_regularizer = latent_covariance_matrix_regularizer
        self.input_covariance_matrix_regularizer = input_covariance_matrix_regularizer
        self.latent_transition_matrix_regularizer = latent_transition_matrix_regularizer
        self.input_transition_matrix_regularizer = input_transition_matrix_regularizer 

        # Temporary variables, to (try to) avoid massive memory allocation
        self.tmp_mat = array(0)
        self.tmp_lambda = array(0) 
        self.tmp_inv_lambda_update = array(0)

    def multivariate_norm_log_pdf(self,x,mu,cov):
        return -0.5 * (dot(x-mu,dot(inv(cov),x-mu)) + len(x)*log(2*pi) + sum(log(eigvals(cov))))

    def E_step(self,y_set):
        """
        Computes the posterior statistics needed in the M step of
        EM, given a set of observation sequences Y_set. Those are:
          - E[z_n | y]
          - E[z_n z_{n-1}^T | y]
          - E[z_n z_n^T | y] 
        The set of probabilities p(y_t | y_{t-1}, ... , y_1) are also given.
        """

        # Note (HUGO): this function should probably be implemented in C
        #              to make it much faster, since it requires for loops.

        # Setting variables with friendlier name
        d_y = self.input_size
        d_z = self.latent_size
        mu_zero = self.mu_zero
        V_zero = self.V_zero
        A = self.A
        C = self.C
        Sigma = self.Sigma
        E = self.E
        
        z_n_post = []
        z_n_z_n_1_post = []
        z_n_z_n_post = []
        cond_probs = []

        # Temporary variable
        mat_times_C_trans = zeros((d_z,d_y))
        K = zeros((d_z,d_y))
        J = zeros((d_z,d_z))

        for y_t in y_set:
            T = len(y_t)
            mu_kalman_t = zeros((T,d_z))     # Filtering mus
            E_kalman_t = zeros((T,d_z,d_z))  # Filtering Es
            mu_post_t = zeros((T,d_z))
            E_post_t = zeros((T,d_z,d_z))
            P_t = zeros((T-1,d_z,d_z)) 
            z_n_z_n_1_post_t = zeros((T-1,d_z,d_z))
            z_n_z_n_post_t = zeros((T,d_z,d_z))
            cond_probs_t = zeros(T)

            # Forward pass

            # Initialization at n = 0
            A_times_prev_mu = zeros(d_z)
            mat_times_C_trans = dot(V_zero,C.T)
            pred = dot(C,mu_zero)
            cov_pred = dot(C,mat_times_C_trans)+Sigma
            K = dot(mat_times_C_trans,inv(cov_pred))
            
            mu_kalman_t[0,:] = mu_zero + dot(K,y_t[0]-pred)
            E_kalman_t[0,:,:] = dot(eye(d_z)-dot(K,C),V_zero)
            cond_probs_t[0] = self.multivariate_norm_log_pdf(y_t[0],pred,cov_pred)
            
            # from n=1 to T-1
            for y_tn,mu_kalman_tn,E_kalman_tn,prev_mu_kalman_tn,prev_E_kalman_tn,P_tn,n in zip(y_t[1:],mu_kalman_t[1:],E_kalman_t[1:],mu_kalman_t[:-1],E_kalman_t[:-1],P_t,xrange(1,T)):
                P_tn[:] = dot(A,dot(prev_E_kalman_tn,A.T))+E
                A_times_prev_mu[:] = dot(A,prev_mu_kalman_tn)
                mat_times_C_trans[:] = dot(P_tn,C.T)
                pred[:] = dot(C,A_times_prev_mu)
                cov_pred[:] = dot(C,mat_times_C_trans)+Sigma
                K[:] = dot(mat_times_C_trans,inv(cov_pred))
                
                mu_kalman_tn[:] = A_times_prev_mu + dot(K,y_tn-pred)
                E_kalman_tn[:,:] = dot(eye(d_z)-dot(K,C),P_tn)
                cond_probs_t[n] = self.multivariate_norm_log_pdf(y_tn,pred,cov_pred)
            
            mu_post_t[-1,:] = mu_kalman_t[-1,:]
            E_post_t[-1,:,:] = E_kalman_t[-1,:,:]
            z_n_z_n_post_t[-1,:,:] = E_post_t[-1,:,:] + outer(mu_post_t[-1,:],mu_post_t[-1,:])

            # Backward pass
            pred = zeros(d_z)
            cov_pred = zeros((d_z,d_z))
            for mu_post_tn,E_post_tn,next_mu_post_tn,next_E_post_tn,mu_kalman_tn,E_kalman_tn,P_tn,z_n_z_n_1_post_tn, z_n_z_n_post_tn in zip(mu_post_t[:-1][::-1],E_post_t[:-1][::-1],mu_post_t[1:][::-1],E_post_t[1:][::-1],mu_kalman_t[:-1][::-1],E_kalman_t[:-1][::-1],P_t[::-1],z_n_z_n_1_post_t[::-1],z_n_z_n_post_t[:-1][::-1]):
                J[:] = dot(E_kalman_tn,dot(A.T,inv(P_tn)))
                pred[:] = dot(A,mu_kalman_tn)
                mu_post_tn[:] = mu_kalman_tn + dot(J,next_mu_post_tn-pred)
                cov_pred[:] = dot(J,dot(next_E_post_tn-P_tn,J.T))
                E_post_tn[:] = E_kalman_tn + cov_pred
                z_n_z_n_1_post_tn[:] = dot(J,next_E_post_tn) + outer(next_mu_post_tn,mu_post_tn)
                z_n_z_n_post_tn[:] = E_post_tn + outer(mu_post_tn,mu_post_tn)
                
                
            z_n_post += [mu_post_t]
            z_n_z_n_1_post += [z_n_z_n_1_post_t]
            z_n_z_n_post += [z_n_z_n_post_t]
            cond_probs += [cond_probs_t]


        return z_n_post,z_n_z_n_1_post,z_n_z_n_post,cond_probs

    def M_step(self,y_set,z_n_post,z_n_z_n_1_post,z_n_z_n_post):
        """
        Updates parameters using the training set and the precomputed posterior statistics
        """

        # Setting variables with friendlier name
        d_y = self.input_size
        d_z = self.latent_size
        mu_zero = self.mu_zero
        V_zero = self.V_zero
        A = self.A
        C = self.C
        Sigma = self.Sigma
        E = self.E

        # Update transition and emission matrices
        A_new_num = zeros((d_z,d_z))
        A_new_denum = zeros((d_z,d_z))
        for z_n_z_n_1_post_t,z_n_z_n_post_t in zip(z_n_z_n_1_post,z_n_z_n_post):            
            A_new_num += z_n_z_n_1_post_t.sum(0)
            A_new_denum += z_n_z_n_post_t[:-1].sum(0)
        A[:] = dot(A_new_num,inv(A_new_denum+eye(d_z)*self.latent_transition_matrix_regularizer))
        
        
        C_new_num = zeros((d_y,d_z))
        C_new_denum = zeros((d_z,d_z))
        for y_t,z_n_post_t,z_n_z_n_post_t in zip(y_set,z_n_post,z_n_z_n_post):            
            C_new_num += sum(y_t[:,:,newaxis]*z_n_post_t[:,newaxis,:],0)
            C_new_denum += z_n_z_n_post_t.sum(0)
        C[:] = dot(C_new_num, inv(C_new_denum+eye(d_z)*self.input_transition_matrix_regularizer))
        
        # Update covariance matrices
        total_sum = 0
        E[:] = 0+eye(d_z)*self.latent_covariance_matrix_regularizer
        for z_n_z_n_1_post_t,z_n_z_n_post_t in zip(z_n_z_n_1_post,z_n_z_n_post):        
            E += sum(z_n_z_n_post_t[1:],0)
            z_n_z_n_1_A_T = dot(sum(z_n_z_n_1_post_t,0),A.T)
            E -= z_n_z_n_1_A_T.T
            E -= z_n_z_n_1_A_T # There is an error in Bishop's equation: the transpose on A is missing
            E += dot(A,dot(sum(z_n_z_n_post_t[:-1],0),A.T))
            total_sum += len(z_n_z_n_1_post_t)
        
        E /= total_sum
        
        total_sum = 0
        Sigma[:] = 0+eye(d_y)*self.input_covariance_matrix_regularizer
        for y_t,z_n_post_t,z_n_z_n_post_t in zip(y_set,z_n_post,z_n_z_n_post):
            Sigma += sum(y_t[:,:,newaxis]*y_t[:,newaxis,:],0)
            C_z_n_y_n = dot(C,sum(z_n_post_t[:,:,newaxis]*y_t[:,newaxis,:],0))
            Sigma -= C_z_n_y_n
            Sigma -= C_z_n_y_n.T # There is an error in Bishop's equation: the transpose on C is missing
            Sigma += dot(C,dot(sum(z_n_z_n_post_t,0),C.T)) # ... idem
            total_sum += len(z_n_z_n_post_t)
        
        Sigma /= total_sum
        
        mu_zero[:] = 0
        V_zero[:] = 0
        total_sum = 0
        for z_n_post_t,z_n_z_n_post_t in zip(z_n_post,z_n_z_n_post):
            mu_zero[:] += z_n_post_t[0]
            V_zero[:] += z_n_z_n_post_t[0]
            V_zero[:] -= outer(z_n_post_t[0],z_n_post_t[0])
            total_sum += 1
        mu_zero /= total_sum
        V_zero /= total_sum

    def train(self,trainset):
        """
        Trains model with the EM algorithm, for (n_epochs - epoch) iterations. 
        If self.epoch == 0, first initialize the model.
        """

        self.input_size = trainset.metadata['input_size']

        # Initialize model
        if self.epoch == 0:
            self.forget()

        # Training with the EM algorithm
        for it in xrange(self.epoch,self.n_epochs):
            # E step
            z_n_post, z_n_z_n_1_post, z_n_z_n_post, cond_probs = self.E_step(trainset)
            
            total_len = reduce(lambda x,y: x+len(y),cond_probs,0)
            print "NLL: ", -reduce(lambda x,y: x+sum(y),cond_probs,0)/total_len

            # M step
            self.M_step(trainset, z_n_post,z_n_z_n_1_post,z_n_z_n_post)
            
    def forget(self):
        d_y = self.input_size
        d_z = self.latent_size
        rng = RandomState(self.seed)

        self.epoch = 0 # Model will be untrained after initialization
        self.mu_zero = rng.randn(d_z)/d_z
        self.V_zero = diag(ones(d_z))
        self.A = rng.rand(d_z,d_z)/d_z
        self.C = rng.rand(d_y,d_z)/d_z
        self.Sigma = diag(ones(d_y))
        self.E = diag(ones(d_z))

    def use(self,dataset):
        """
        Outputs the log-likelihood of the sequences in dataset
        """
        z_n_post, z_n_z_n_1_post, z_n_z_n_post, cond_probs = self.E_step(dataset)
        outputs = array(map(sum,cond_probs))
        return outputs[:,newaxis]

    def test(self,dataset):
        """
        Outputs the log-likelihood and average NLL (normalized by the length of
        each sequence) of the sequences in dataset
        """
        outputs = self.use(dataset)
        costs = zeros((len(outputs),1))
        # Compute normalized NLLs
        for seq,t in zip(dataset,xrange(len(dataset))):
            costs[t,0] = -outputs[t,0]/len(seq)

        return outputs,costs
                
    #def computeOutputsOnDataset(self, dataspec, dataset):
    #    """
    #    Computes the log-likelihood of the sequences in a dataset
    #    """
    #
    #    C = self.n_classes
    #    M = self.n_components
    #    for elt in dataset:
    #        x = elt.data()
    #        out = {}
    #        if self.use_gain_adaptation:
    #            gain = ones((len(x),M))
    #        for k,c in self.target_string_mapping.iteritems():
    #            if self.use_gain_adaptation:
    #                self.update_gain(x,c,gain)
    #                log_obs = self.observation_log_probabilities(x, c, gain)
    #            else:
    #                log_obs = self.observation_log_probabilities(x, c, None)
    #            #alpha, beta, scaling = self.forward_backward( c, log_obs)
    #            alpha, beta, scaling = msvarhelper.forwardBackward(log_obs,self.log_transition_matrix[c],log(self.prior[c]))
    #            out[k] = -self.get_nll(scaling)
    #            if not self.ignore_class_prior:
    #                out[k] += log( self.class_prior[c] )
    #        yield out
    #
    #def computeAllOutputs(self, dataspec):
    #    """
    #    Compute class scores on the test set and updates dataspec accordingly
    #    """
    #    testset = self.test_inputspec(dataspec)
    #    outputsname = self.testOutputName()
    #    dataspec[outputsname] = DatasetWIterThunk( preproc=None, 
    #                                               dataspec=dataspec, 
    #                                               dataset=testset,
    #                                               func = self.computeOutputsOnDataset )
    #    return dataspec
        



class SparseLinearDynamicalSystem(Learner):
    """ 
    Sparse Linear Dynamical System (SLDS)
 
    This is a linear dynamical system where the latent space representation
    is encouraged to be sparse.

    Options:
    - 'n_epochs'
    - 'latent_size'
    - 'latent_covariance_matrix_regularizer'
    - 'input_covariance_matrix_regularizer'
    - 'latent_transition_matrix_regularizer'
    - 'input_transition_matrix_regularizer'
    - 'gamma_prior'
    - 'seed'

    Required metadata:
    - 'input_size'

    """
    def __init__(   self,
                    n_epochs= sys.maxint, # Maximum number of iterations on the training set
                    latent_size = 10, # Size of the latent variable
                    latent_covariance_matrix_regularizer = 0.,
                    input_covariance_matrix_regularizer = 0.,
                    latent_transition_matrix_regularizer = 0.,
                    input_transition_matrix_regularizer = 0.,
                    gamma_prior = 1.,
                    seed = 1827,
                    ):
        self.epoch = 0
        self.n_epochs = n_epochs
        self.latent_size = latent_size
        self.seed = seed
        self.rng = RandomState(seed)
        self.latent_covariance_matrix_regularizer = latent_covariance_matrix_regularizer
        self.input_covariance_matrix_regularizer = input_covariance_matrix_regularizer
        self.latent_transition_matrix_regularizer = latent_transition_matrix_regularizer
        self.input_transition_matrix_regularizer = input_transition_matrix_regularizer 
        self.gamma_prior = gamma_prior

        # Temporary variables, to (try to) avoid massive memory allocation
        self.tmp_mat = array(0)
        self.tmp_lambda = array(0) 
        self.tmp_inv_lambda_update = array(0)

    def multivariate_norm_log_pdf(self,x,mu,cov):
        return -0.5 * (dot(x-mu,dot(inv(cov),x-mu)) + len(x)*log(2*pi) + sum(log(eigvals(cov))))

    def E_step(self,y_set,gamma_set):
        """
        Computes the posterior statistics needed in the M step of
        EM, given a set of observation sequences Y_set. Those are:
          - E[z_n | y]
          - E[z_n z_{n-1}^T | y]
          - E[z_n z_n^T | y] 
        The E step also outputs the non-parametric, sparsity inducing variances gamma_t
        Finally, the set of probabilities p(y_t | y_{t-1}, ... , y_1, gamma_{t-1}, ... , gamma_1) are also given.
        """

        # Note (HUGO): this function should probably be implemented in C
        #              to make it much faster, since it requires for loops.

        # Setting variables with friendlier name
        d_y = self.input_size
        d_z = self.latent_size
        #mu_zero = self.mu_zero
        #V_zero = self.V_zero
        A = self.A
        C = self.C
        Sigma = self.Sigma
        E = self.E

        finished = False
        while not finished:
            z_n_post = []
            z_n_z_n_1_post = []
            z_n_z_n_post = []
            cond_probs = []

            # Temporary variable
            mat_times_C_trans = zeros((d_z,d_y))
            K = zeros((d_z,d_y))
            J = zeros((d_z,d_z))

            for y_t,gamma_t in zip(y_set,gamma_set):
                T = len(y_t)
                mu_kalman_t = zeros((T,d_z))     # Filtering mus
                E_kalman_t = zeros((T,d_z,d_z))  # Filtering Es
                mu_post_t = zeros((T,d_z))
                E_post_t = zeros((T,d_z,d_z))
                P_t = zeros((T-1,d_z,d_z)) 
                z_n_z_n_1_post_t = zeros((T-1,d_z,d_z))
                z_n_z_n_post_t = zeros((T,d_z,d_z))
                cond_probs_t = zeros(T)
            
                # Forward pass
            
                # Initialization at n = 0
                A_times_prev_mu = zeros(d_z)
                mat_times_C_trans = dot(diag(gamma_t[0]),C.T)
                pred = zeros((d_y))
                cov_pred = dot(C,mat_times_C_trans)+Sigma
                K = dot(mat_times_C_trans,inv(cov_pred))

                mu_kalman_t[0,:] = dot(K,y_t[0]-pred)
                E_kalman_t[0,:,:] = dot(eye(d_z)-dot(K,C),diag(gamma_t[0]))
                cond_probs_t[0] = self.multivariate_norm_log_pdf(y_t[0],pred,cov_pred)
                
                # from n=1 to T-1
                for y_tn,mu_kalman_tn,E_kalman_tn,prev_mu_kalman_tn,prev_E_kalman_tn,P_tn,gamma_tn,n in zip(y_t[1:],mu_kalman_t[1:],E_kalman_t[1:],mu_kalman_t[:-1],E_kalman_t[:-1],P_t,gamma_t[1:],xrange(1,T)):
                    E_gamma = diag(1/(1/E + 1/gamma_tn))
                    A_gamma = A/((ones(d_z)+E/gamma_tn)[:,newaxis])
                    P_tn[:] = dot(A_gamma,dot(prev_E_kalman_tn,A_gamma.T))+E_gamma
                    A_times_prev_mu[:] = dot(A_gamma,prev_mu_kalman_tn)
                    mat_times_C_trans[:] = dot(P_tn,C.T)
                    pred[:] = dot(C,A_times_prev_mu)
                    cov_pred[:] = dot(C,mat_times_C_trans)+Sigma
                    K[:] = dot(mat_times_C_trans,inv(cov_pred))
                    
                    mu_kalman_tn[:] = A_times_prev_mu + dot(K,y_tn-pred)
                    E_kalman_tn[:,:] = dot(eye(d_z)-dot(K,C),P_tn)
                    cond_probs_t[n] = self.multivariate_norm_log_pdf(y_tn,pred,cov_pred)
                
                mu_post_t[-1,:] = mu_kalman_t[-1,:]
                E_post_t[-1,:,:] = E_kalman_t[-1,:,:]
                z_n_z_n_post_t[-1,:,:] = E_post_t[-1,:,:] + outer(mu_post_t[-1,:],mu_post_t[-1,:])
            
                # Backward pass
                pred = zeros(d_z)
                cov_pred = zeros((d_z,d_z))
                for mu_post_tn,E_post_tn,next_mu_post_tn,next_E_post_tn,mu_kalman_tn,E_kalman_tn,P_tn,z_n_z_n_1_post_tn, z_n_z_n_post_tn,gamma_tn in zip(mu_post_t[:-1][::-1],E_post_t[:-1][::-1],mu_post_t[1:][::-1],E_post_t[1:][::-1],mu_kalman_t[:-1][::-1],E_kalman_t[:-1][::-1],P_t[::-1],z_n_z_n_1_post_t[::-1],z_n_z_n_post_t[:-1][::-1],gamma_t[1:][::-1]):
                    A_gamma = A/((ones((d_z))+E/gamma_tn))[:,newaxis]
                    J[:] = dot(E_kalman_tn,dot(A_gamma.T,inv(P_tn)))
                    pred[:] = dot(A_gamma,mu_kalman_tn)
                    mu_post_tn[:] = mu_kalman_tn + dot(J,next_mu_post_tn-pred)
                    cov_pred[:] = dot(J,dot(next_E_post_tn-P_tn,J.T))
                    E_post_tn[:] = E_kalman_tn + cov_pred
                    z_n_z_n_1_post_tn[:] = dot(J,next_E_post_tn) + outer(next_mu_post_tn,mu_post_tn)
                    z_n_z_n_post_tn[:] = E_post_tn + outer(mu_post_tn,mu_post_tn)
                                        
                z_n_post += [mu_post_t]
                z_n_z_n_1_post += [z_n_z_n_1_post_t]
                z_n_z_n_post += [z_n_z_n_post_t]
                cond_probs += [cond_probs_t]
            
            # Compute new gammas
            gamma_mean_diff = 0
            tot_length = 0
            for gamma_t,zz_t in zip(gamma_set,z_n_z_n_post):
                tot_length += len(gamma_t)
                # For n=0, the solution is (see model description):
                #    gamma_i = \frac{d}{2} (-1 + \sqrt{1 + 8*E[z_t z_t]_i/d})
                new_gamma = self.gamma_prior/2 * (sqrt(1+8/self.gamma_prior*diag(zz_t[0]))-1)
                gamma_mean_diff += sum((gamma_t[0]-new_gamma)**2)/d_z
                
                gamma_t[0] = new_gamma
                for gamma_tn,zz_tn,zz_tn_prev in zip(gamma_t[1:],zz_t[1:],zz_t[:-1]):
                    AzzA_prev = diag(dot(A,dot(zz_tn_prev,A.T)))
                    for i in xrange(d_z):
                        E_ii = E[i]
                        G = zz_tn[i,i]
                        H = AzzA_prev[i]
                        a1 = 2/self.gamma_prior
                        a2 = 4/self.gamma_prior*E_ii
                        a3 = 2/self.gamma_prior*E_ii**2 + E_ii + H - G
                        a4 = (E_ii-2*G)*E_ii
                        a5 = -G*E_ii**2
                        r = roots([a1,a2,a3,a4,a5])
                        m = inf
                        am = nan
                        # Find the best positive solution among fix points.
                        # We don't check if 0 is the best solution, since it's
                        # very unlikely to be a minimum (it can only happen if
                        # zz_tn[i,i] is 0).
                        for c in r:
                            if not iscomplex(c) and c >= 0:
                                f = 0.5*(G/c-H/(c+E_ii)-log(E_ii+c)+log(c))+c/self.gamma_prior
                                if f < m:
                                    am = c
                        gamma_mean_diff += (am-gamma_tn[i])**2/d_z
                        gamma_tn[i] = am
            gamma_mean_diff /= tot_length
            if gamma_mean_diff < 0.0001:
                finished = True

        return z_n_post,z_n_z_n_1_post,z_n_z_n_post,gamma_set,cond_probs

    def M_step(self,y_set,z_n_post,z_n_z_n_1_post,z_n_z_n_post,gamma_set):
        """
        Updates parameters using the training set, the precomputed posterior statistics and the gammas
        """

        # Setting variables with friendlier name
        d_y = self.input_size
        d_z = self.latent_size
        #mu_zero = self.mu_zero
        #V_zero = self.V_zero
        A = self.A
        C = self.C
        Sigma = self.Sigma
        E = self.E

        # Update transition and emission matrices
        A_new_num = zeros((d_z))
        A_new_denum = zeros((d_z,d_z))
        for i in xrange(d_z):
            A_new_num[:] = 0
            A_new_denum[:] = 0
            for z_n_z_n_1_post_t,z_n_z_n_post_t,gamma_t in zip(z_n_z_n_1_post,z_n_z_n_post,gamma_set):
                A_new_num += z_n_z_n_1_post_t[:,i,:].sum(0) # Could divide by E[i], but is cancelled by denum
                weights = gamma_t[:,i]/(gamma_t[:,i]+E[i])
                A_new_denum += (z_n_z_n_post_t[:-1]*weights[:,newaxis,newaxis]).sum(0) # Could divide by E[i], but is cancelled by num
            A[i] = dot(A_new_num,inv(A_new_denum+eye(d_z)*self.latent_transition_matrix_regularizer))
                
        C_new_num = zeros((d_y,d_z))
        C_new_denum = zeros((d_z,d_z))
        for y_t,z_n_post_t,z_n_z_n_post_t in zip(y_set,z_n_post,z_n_z_n_post):            
            C_new_num += sum(y_t[:,:,newaxis]*z_n_post_t[:,newaxis,:],0)
            C_new_denum += z_n_z_n_post_t.sum(0)
        C[:] = dot(C_new_num, inv(C_new_denum+eye(d_z)*self.input_transition_matrix_regularizer))
        
        # Update covariance matrices
        # TODO!

        total_sum = 0
        Sigma[:] = 0+eye(d_y)*self.input_covariance_matrix_regularizer
        for y_t,z_n_post_t,z_n_z_n_post_t in zip(y_set,z_n_post,z_n_z_n_post):
            Sigma += sum(y_t[:,:,newaxis]*y_t[:,newaxis,:],0)
            C_z_n_y_n = dot(C,sum(z_n_post_t[:,:,newaxis]*y_t[:,newaxis,:],0))
            Sigma -= C_z_n_y_n
            Sigma -= C_z_n_y_n.T # There is an error in Bishop's equation: the transpose on C is missing
            Sigma += dot(C,dot(sum(z_n_z_n_post_t,0),C.T)) # ... idem
            total_sum += len(z_n_z_n_post_t)
        
        Sigma /= total_sum
        
    def train(self,trainset):
        """
        Trains model with the EM algorithm, for (n_epochs - epoch) iterations. 
        If self.epoch == 0, first initialize the model.
        """

        self.input_size = trainset.metadata['input_size']

        # Initialize model
        if self.epoch == 0:
            self.forget()

        # Initialize the gammas
        if self.epoch == 0:
            self.gamma_set = []
            for seq in trainset:
                self.gamma_set = self.gamma_set + [ones((len(seq),self.latent_size))]

        # Training with the EM algorithm
        for it in xrange(self.epoch,self.n_epochs):
            # E step
            z_n_post, z_n_z_n_1_post, z_n_z_n_post, new_gamma_set, cond_probs = self.E_step(trainset,self.gamma_set)
            self.gamma_set = new_gamma_set
            total_len = reduce(lambda x,y: x+len(y),cond_probs,0)
            print "NLL: ", -reduce(lambda x,y: x+sum(y),cond_probs,0)/total_len

            # M step
            self.M_step(trainset, z_n_post,z_n_z_n_1_post,z_n_z_n_post,self.gamma_set)
            
    def forget(self):
        d_y = self.input_size
        d_z = self.latent_size
        rng = RandomState(self.seed)

        self.epoch = 0 # Model will be untrained after initialization
        #self.mu_zero = rng.randn(d_z)/d_z
        #self.V_zero = diag(ones(d_z))
        self.A = rng.rand(d_z,d_z)/d_z
        self.C = rng.rand(d_y,d_z)/d_z
        self.Sigma = diag(ones(d_y))
        self.E = ones(d_z)

    def use(self,dataset):
        """
        Outputs the log-likelihood of the sequences in dataset
        """
        # Initialize gamma_set
        gamma_set = []
        for seq in dataset:
            gamma_set = gamma_set + [ones((len(seq),self.latent_size))]

        z_n_post, z_n_z_n_1_post, z_n_z_n_post, new_gamma_set, cond_probs = self.E_step(dataset,gamma_set)
        outputs = array(map(sum,cond_probs))
        return outputs[:,newaxis]

    def test(self,dataset):
        """
        Outputs the log-likelihood and average NLL (normalized by the length of
        each sequence) of the sequences in dataset
        """
        outputs = self.use(dataset)
        costs = zeros((len(outputs),1))
        # Compute normalized NLLs
        for seq,t in zip(dataset,xrange(len(dataset))):
            costs[t,0] = -outputs[t,0]/len(seq)

        return outputs,costs
        
