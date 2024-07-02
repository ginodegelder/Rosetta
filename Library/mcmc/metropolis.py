# -*- coding: utf-8 -*-

"""
    Module for the metropolis-based samplers
    @author: Navid Hedjazian
"""
import os
import sys
sys.path.insert(1, os.path.abspath('../../'))
from Inputs import inversion_params

import numpy as np
from datetime import datetime
from numpy.random import default_rng
from mpi4py import MPI
import arviz


from .mcmc_base import MCMCBase

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nb_proc = comm.Get_size()

STATS = {
    "loglikelihood": ["_x_loglike", 1],
    "accept_ratio": ["_general_accept_ratio", 1],
    "prop_S": ["prop_S", "n_vars"],
    "parameter_accept_ratio": ["_accept_ratios", "n_vars"]
}

RESTART = inversion_params['restart']
R_HAT = inversion_params['R-hat']
n_chains = inversion_params['n_chains']
# Number of processors used in a single chain
proc_in_chain = nb_proc // n_chains
# Arrays with ranks of main cores. One main per chain.
main_list = np.arange(0, nb_proc, proc_in_chain)


class Metropolis1dStep(MCMCBase):
    """
    Class Metropolis1dStep.

    Run a Metropolis MCMC algorithm using 1D steps only.

    Description of attributes:

    MCMC parameters:
    ----------------
        # Information for MCMC
        n_vars : integer
            total number of variables to sample.

        n_samples : integer
            number of saved iterations in the chain.

        show_stats : integer
            interval n° of samples to show stats.

        # Functions

        loglikelihood : function
            Log-likelihood function.

        logprior : function
            Prior probability function.

        proposal : function
            Proposal function.

        prop_S : 1d numpy array, shape (n_vars)
            Standard deviation for the proposal function.

    MCMC results:
    -------------
        samples : ndarray shape (n_samples, n_vars)
            The samples in the parameter space.

        save_stats : dictionary
            Container for the stats to save during simulation.

        duration : datetime
            Duration of the last run.

    """

    # How to define prior, proposal and loglikelihood functions
    # Proposal
    # def proposal(x, prop_S)
    #
    # Prior
    # def logprior(x)
    #
    # Loglikelihood
    # def loglikelihood(x)
    #

    def __init__(self, n_vars=None, verbose=0, show_stats=10000):
        super(Metropolis1dStep, self).__init__(n_vars=n_vars,
                                               verbose=verbose,
                                               show_stats=show_stats)

        # MCMC state
        self._current_iter = 1
        self._recent_n_prop = None
        self._recent_n_accept = None
        self._prop_mat = None
        self._accept_mat = None
        self._accept_ratios = None
        self._general_accept_ratio = None
        self._starting_sample = None
        self._init_prop_S = None  # Saves the initial proposal std
        self._x_loglike = None  # Saves the current loglikelihood
        
        # Number and index of chains
        self.n_chains = n_chains
        self.i_chain = rank // proc_in_chain

    # TODO: define a default for the getattr method

    def add_stat(self, name):
        if name in STATS.keys():
            self._save_stats.append(name)
        else:
            print("Invalid stat, chose from:")
            print(STATS.keys())

    def print_stats(self, i_sample):
        """
        Print the stats that are recorded
        """
        print()
        print("---------------------------------")
        print("        -- MCMC stats --         ")
        print("---------------------------------")
        print("At sample number {}".format(self._current_iter))
        for stat in self._save_stats:
            print(stat)
            print(self.stats[stat][i_sample, ...])

    def initialize(self):
        """
        Run all functions required before starting or restarting the MCMC
        algorithm
        1. Re-initialize results arrays and set state to 0
        2. Reset adaptive parameters
        """
        self.reset()
        pass

    def initialize_arrays(self, x0, n, thin, tune, tune_interval,
                          discard_tuning, posterior_predict):
        #self.n_chains = n_chains
        self.n_vars = x0.size
        # Results arrays
        self.n_samples = n
        if discard_tuning:
            self.n_samples -= tune
        self.n_samples = n // thin

        self.samples = np.zeros((self.n_samples, self.n_vars),
                                dtype=self.sample_dtype)

        # Set self.stats as a dictionary of numpy arrays for each stat
        for stat in self._save_stats:
            # get the stats attributes, the second element of the list is the
            # number of variables
            attr = STATS[stat][1]
            if isinstance(attr, str):
                n_params = int(getattr(self, attr))
            else:
                n_params = int(attr)
            self.stats[stat] = np.zeros((self.n_samples, n_params))

        for key in posterior_predict.keys():
            data_shape = posterior_predict[key].shape
            # Multi : does not work with n_chains as it will be concat at the end !!!
            # Here, better let n_chains to 1.
            self.posterior_predictive[key] = np.empty((
                self.n_chains, self.n_samples, *data_shape))

        # Private arrays
        self._current_iter = 1
        self._recent_n_prop = np.zeros(self.n_vars, dtype=np.int32)
        self._recent_n_accept = np.zeros(self.n_vars, dtype=np.int32)
        self._prop_mat = np.zeros((tune_interval, self.n_vars),
                                  dtype=np.bool_)
        self._accept_mat = np.zeros((tune_interval, self.n_vars),
                                    dtype=np.bool_)
        self._accept_ratios = np.zeros(self.n_vars)

    def restart_arrays(self, dataset, x0, n, thin, tune, tune_interval,
                          discard_tuning, posterior_predict):
        self.n_vars = x0.size
        # Results arrays
        self.n_samples = n
        if discard_tuning:
            self.n_samples -= tune
        self.n_samples = n // thin
        # Saved samples form the saved chain.
        saved_samples = dataset.posterior.x[0,:,:].values
        # Initialize the array for the new samples.
        self.samples = np.zeros((self.n_samples, self.n_vars),
                                dtype=self.sample_dtype)
        # Concatenate saved and new samples arrays
        self.samples = np.concatenate((saved_samples, self.samples), axis=0)

        # Saved loglikelihoods
        save_loglike = dataset.log_likelihood.log_likelihood.values[0,:,:]
        # Saved stats
        stats_items = (dataset.sample_stats
                       .to_dict(data='array')['data_vars'].items())
        save_sample_stat = {key: value['data'][0,:,:] 
                            for key, value in stats_items}

        # Set self.stats as a dictionary of numpy arrays for each stat
        for stat in self._save_stats:
            # get the stats attributes, the second element of the list is the
            # number of variables
            attr = STATS[stat][1]
            if isinstance(attr, str):
                n_params = int(getattr(self, attr))
            else:
                n_params = int(attr)
            self.stats[stat] = np.zeros((self.n_samples, n_params))

            # Concat the new stats dict and the saved stats dict
            if stat == 'loglikelihood':
                self.stats[stat] = np.concatenate((save_loglike, 
                                                   self.stats[stat]), axis=0)
            # Concatenate the values of dict_init into data_vars_dict
            else:
                self.stats[stat] = np.concatenate((save_sample_stat[stat], 
                                                   self.stats[stat]), axis=0)

        # Extract the post predict keys and values of saved chain.
        postpred_items = (dataset.posterior_predictive
                          .to_dict(data='array')['data_vars'].items())
        # Keep only data values from the values in items.
        save_posterior_predict = {
            key: value['data'] for key, value in postpred_items
            }
        for key in save_posterior_predict.keys():
            # Take only one sample from the save.
            data_shape = save_posterior_predict[key][0,0,:].shape
            # Create a dict of empty values for the new post predicts.
            self.posterior_predictive[key] = np.empty((
                self.n_chains, self.n_samples, *data_shape))
            # Concatenate the values of save dict and empty dict.
            self.posterior_predictive[key] = np.concatenate((
                save_posterior_predict[key], 
                self.posterior_predictive[key]), axis=1)
            
        # Update n_samples to take saved chain in account. 
        self.n_samples += self.saved_n_samples
        # Update prop_S to saved one. Check for multi chains.
        self.prop_S = dataset.sample_stats.prop_S[0,-1,:]

        # Private arrays
        self._current_iter = self.saved_n_samples + 1
        #recent_n_prop = np.array(np.diff(ds.sample_stats.parameter_accept_ratio[0,-2000:,:].values, axis=0),dtype=np.bool_)
        self._recent_n_prop = np.zeros(self.n_vars, dtype=np.int32)
        self._recent_n_accept = np.zeros(self.n_vars, dtype=np.int32)
        self._prop_mat = np.zeros((tune_interval, self.n_vars),
                                  dtype=np.bool_)
        # accept_mat = np.array(np.diff(dataset.posterior.x[0,-self.tune_interval:,:].values,axis=0),dtype=np.bool_)
        self._accept_mat = np.zeros((tune_interval, self.n_vars),
                                    dtype=np.bool_)
        #self.accept_ratios = dataset.sample_stats.parameter_accept_ratio[0,-1,:]
        self._accept_ratios = np.zeros(self.n_vars)

    def reset(self):
        """
        Function to reset the parameters learned during the sampling, such as
        proposal parameters.
        """
        self.prop_S = self._init_prop_S

    def save_sample(self, x, i):
        """
        Store vector x as the ith sample
        Parameters
        ----------
        x : 1D numpy array
        i: integer
        """
        self.samples[i, :] = x  # Add point to history of accepted models

    def save_posterior_predictive(self, predict, ichain, isample):
        """
        Store data predictions predict as the ith sample
        Parameters
        ----------
        predict : dict
        ichain : integer
        isample : integer
        """
        for key in predict.keys():
            self.posterior_predictive[key][ichain, isample, ...] = predict[key]

    def tune(self):
        """
        Tune the proposal standard deviation according to the previous
        samples acceptance.

        Taken from pymc3 documentation:
            Rate    Variance adaptation
            ----    -------------------
            <0.001        x 0.1
            <0.05         x 0.5
            <0.2          x 0.9
            >0.5          x 1.1
            >0.75         x 2
            >0.95         x 10
        Returns
        -------

        """
        for i in range(self.n_vars):
            if self._accept_ratios[i] < 0.001:
                self.prop_S[i] *= 0.1
                continue
            elif self._accept_ratios[i] < 0.05:
                self.prop_S[i] *= 0.5
                continue
            elif self._accept_ratios[i] < 0.2:
                self.prop_S[i] *= 0.9
                continue
            elif self._accept_ratios[i] > 0.95:
                self.prop_S[i] *= 10.
                continue
            elif self._accept_ratios[i] > 0.75:
                self.prop_S[i] *= 2.
                continue
            elif self._accept_ratios[i] > 0.5:
                self.prop_S[i] *= 1.1
                continue

    def run(self, chains, n, Outs_path, tune=0, tune_interval=1000,
            discard_tuned_samples=False, thin=1):
        """
        Runs the algorithm.

        Parameters
        ----------
        x0 : 1D numpy array
            Starting sample.

        n : int
            Number of iterations.

        tune : int
            Number of samples where proposal tuning is allowed.

        tune_interval : int
            Interval of samples for tuning the proposal.

        discard_tuned_samples : bool
            Do or do not save tuned samples. Default to False.

        thin : int
            Thining of the chain, save the sample every number of iterations.

        """
        start_time = datetime.now()
        #self.tune_interval = tune_interval
        
        # Initializes the saving dictionnaries from one simulation to another.
        dict_save_run = {}
        dict_save_vars = {}

        if type(RESTART) == str:
            if rank == 0:
                dataset = arviz.from_netcdf(
                    './Outs/FigS4d/'+RESTART+"/Dataframes/MCMC_raw.nc"
                    )
                    
                for other_rank in range(1, nb_proc):
                    comm.send(dataset, dest=other_rank, tag=0)
                    
            else:
                dataset = comm.recv(source = 0, tag=0)
            
            # Number of runs for saved chain.
            self.saved_n_samples = len(dataset.posterior.draw.values)
            # Extract the last sample of the saved chain. Index 0 for 1 chain.
            x0 = dataset.posterior.x[0, -1, :].values   
            
            # Initialize numpy random generator
            rng = default_rng()
            print("Start Metropolis1dStep")
            print("----------------------")
            print("number of saved samples")
            print(self.saved_n_samples)

            x = x0
            # Initialize  likelihood and prior from last sample.
            # Extract for 1 chain. Multi : use self.i_chain
            self._x_loglike = (dataset.log_likelihood
                               .log_likelihood[0, -1].values)
            # Extract all posterior predict in save file
            postpred_items = (dataset.posterior_predictive
                              .to_dict(data='array')['data_vars'].items())
            # Keep only the last arrays. Multi : use self.i_chain
            posterior_predict = {key: value['data'][0,-1,:] 
                                 for key, value in postpred_items}
            # Empty the extraction.
            postpred_items = None
            #self._x_loglike, posterior_predict = self.loglikelihood(x)
            # Re initialize arrays from saved chain.
            self.restart_arrays(dataset, x0, n, thin, tune, tune_interval,
                               discard_tuned_samples, posterior_predict)
            
            # Extract the number of accepted models from saved chain
            n_accept = (dataset.sample_stats.accept_ratio[0,-1,:].values * 
                        self.saved_n_samples)
            # Close the dataset
            del dataset
            n_tup = (self.saved_n_samples, self.saved_n_samples+n)

        else:
            # Initialize numpy random generator
            rng = default_rng()
            print("Start Metropolis1dStep")
            print("----------------------")
            print("number of saved samples")
            print(self.n_samples)
            
            # Multi : Each chain has its own x0
            x0 = chains[rank//proc_in_chain]
            # x0 = x[0]
            x = x0
    
            # Initialize likelihood and prior
            # Multi : Don't need to change
            (self._x_loglike, posterior_predict, 
             dict_save_run, dict_save_vars) = self.loglikelihood(
                 x, dict_save_run, dict_save_vars)
            # Initialize predictions
            # Initialize arrays
            # Multi : Don't need to change
            self.initialize_arrays(x0, n, thin, tune, tune_interval,
                               discard_tuned_samples, posterior_predict)
            
            n_accept = 0
            n_tup = (0, n)
        
        # Multi : different for each chains. No need to change
        x_logprior = self.logprior(x)
        if self.verbose > 1:
            print("Starting log-likelihood")
            print(self._x_loglike)
            print()

        # Initialize other parameters
        
        i_sample = 0
        
        if rank == 0: # Multi : if rank in main_list
            # Start MCMC sampling
            # -------------------
            for i in range(*n_tup):
    
                if self.verbose > 1:
                    print("ITER", i)
                    print(" ------------------------------------")
    
                # perturb the proposal point x to xp
                xp = self.proposal(x, self.prop_S)
                modified_parameters_list = np.nonzero(xp - x)
    
                # Roll prop and accept matrix
                self._prop_mat = np.roll(self._prop_mat, 1, axis=0)
                self._prop_mat[0, :] = False
                self._accept_mat = np.roll(self._accept_mat, 1, axis=0)
                self._accept_mat[0, :] = False
    
                # Saves which parameter has been proposed
                for param_i in modified_parameters_list:
                    self._prop_mat[0, param_i] = True
    
                if self.verbose > 1:
                    print("Modified parameter")
                    print(modified_parameters_list)
                    print("Proposal x")
                    print(xp - x)
                    print(xp)
    
                # Get the prior at xp
                xp_logprior = self.logprior(xp)
    
                if xp_logprior == -np.inf:
                    compute_likelihood = False
                    if self.verbose > 1:
                        print("Prior is {}".format(xp_logprior))
                        print("Do not compute log-likelihood")
                else:
                    compute_likelihood = True
                
                # Share the list of parameters and compute_likelihood
                # Multi : for other_rank in range(rank+1, rank+proc_in_chain)
                for other_rank in range(1, nb_proc):
                    comm.send(xp, dest=other_rank, tag = 1)
                    comm.send(compute_likelihood, dest=other_rank, tag = 2)
                    
                if compute_likelihood:
                    #print("Simu number", self._current_iter)
                    xp_loglike, prop_predict, dict_save_run, dict_save_vars = (
                        self.loglikelihood(xp, dict_save_run, dict_save_vars)
                        )
                else:
                    xp_loglike = - np.inf
    
                u = rng.random()
                if self.verbose > 1:
                    print("       LOGLIKE       ")
                    print("---------------------")
                    print(self._x_loglike, xp_loglike)
    
                    print("")
                    print("       PRIOR       ")
                    print("-------------------")
                    print(x_logprior, xp_logprior)
    
                    print("rand u is ", u)
                    print("Accept if LOG(PRIOR) + LOGLIKE diff > log(u)")
                    print("log u ", np.log(u))
                    print("LOGLIKE difference", xp_loglike - self._x_loglike)
                    print("")
    
                # Draw a random number between 0 and 1
                # def accept():
                # Crash : and xp_loglike is not None
                if np.log(u) < xp_logprior + xp_loglike - self._x_loglike - \
                        x_logprior:
                    # case accept
                    np.copyto(x, xp)  # copy xp into x
                    self._x_loglike = xp_loglike  # update likelihood value
                    x_logprior = xp_logprior  # update log prior
                    posterior_predict = prop_predict.copy()
                    n_accept += 1  # Update accepted moves
    
                    if self.verbose > 1:
                        print(" Accept iter {} \n".format(i))
    
                    # Saves which parameter has been accepted
                    for param_i in modified_parameters_list:
                        self._accept_mat[0, param_i] = True
    
                # Update acceptance ratios
                self._recent_n_prop = np.sum(self._prop_mat, axis=0)
                self._recent_n_accept = np.sum(self._accept_mat, axis=0)
                self._accept_ratios = \
                    np.true_divide(self._recent_n_accept, self._recent_n_prop,
                                   where=(self._recent_n_prop != 0))
    
                self._general_accept_ratio = float(n_accept) / float(
                    self._current_iter)
    
                if self._current_iter % tune_interval == 0 and \
                        self._current_iter < tune:
                    self.tune()
                    if self.verbose > 0:
                        print()
                        print("----------------------")
                        print(" Tune MCMC parameters ")
                        print("----------------------")
                        print("New standard deviation of proposal distribution at "
                              "iter {}:".format(self._current_iter))
                        with np.printoptions(precision=2, suppress=True):
                            print(self.prop_S)
                        print()
                        
                t1 = datetime.now()
                print(f"Iteration: {self._current_iter}/{n_tup[1]}. "\
                      f"Mean time for one iteration: "\
                      f"{(t1 - start_time)/(self._current_iter - n_tup[0])}",
                      end='\r')
                
                # Multi : put gelman rubin and chain number. Put it with save.
                if self._current_iter % self.show_stats == 0:
                    self.print_stats(i_sample)
                    if self.verbose > 1:
                        print("Position x: {}".format(x))
                        print("Acceptance rate for each parameter:")
                        print(self._accept_ratios)
                    print()
    
                # Records the proposal point
                if n % thin == 0:
                    i_sample = int(i / thin)
                    # Get chain number. Only 0 for serial case
                    # Multi : self.i_chain 
                    i_chain = 0
                    self.save_sample(x, i_sample)
                    self.save_posterior_predictive(posterior_predict,
                                                   i_chain, i_sample)
                    for stat in self._save_stats:
                        # get the attribute name
                        attr = STATS[stat][0]
                        # save it to dictionnary
                        self.stats[stat][i_sample, ...] = getattr(self, attr)

                # Save the chain in arviz format. Multi : keep one for each and 
                # concat along chains. comm.send/rec the datasets. 
                # f'/MCMC_chain_{self.i_chain}.nc'
                # if rank == 0: concat files in MCMC_raw.nc 
                if (self._current_iter % 1000 == 0 or 
                    self._current_iter == n_tup[1]-1):
                    #Multi
                    #self.write_samples(Outs_path+f'/MCMC_chain_{self.i_chain}.nc', 
                    #                   format='arviz')
                    #dataset = self.get_results(format='arviz')
                    #data_list = comm.allgather(dataset)
                    #if rank == 0:
                    #    concat_array = arviz.concat(data_list, dim='chain')
                    #    concat_array.to_netcdf(Outs_path+'/MCMC_raw.nc', 
                    #                   format='arviz')
                    #    concat_array = None
                    #dataset = None
                    self.write_samples(Outs_path+'/MCMC_raw.nc', 
                                       format='arviz')
    
                # Update iter
                self._current_iter += 1
                # Send the iteration to other ranks. Multi : change other_ranks
                for other_rank in range(1, nb_proc):
                    comm.send(self._current_iter, dest=other_rank, tag = 3)
                
        else:            
             for i in range (n):
                 # Multi : Select the master core for the chain         
                 #chain_main = main_list[self.i_chain]
                     
                 xp = comm.recv(source = 0, tag = 1)
                 compute_likelihood = comm.recv(source = 0, tag = 2)
                 
                 if compute_likelihood:
                     (xp_loglike, prop_predict, 
                      dict_save_run, dict_save_vars) = self.loglikelihood(
                          xp, dict_save_run, dict_save_vars)
                 else:
                     xp_loglike = - np.inf
                     
                 # Synchronize the iteration with rank 0
                 self._current_iter = comm.recv(source=0, tag=3)
                                     
                 
        # Saves the final quantities
        self.duration = datetime.now() - start_time
