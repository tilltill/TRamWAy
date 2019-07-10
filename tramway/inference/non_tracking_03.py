from .base import *
import numpy as np
import pandas as pd
from time import time
from numpy import array, reshape, sum, zeros, ones, arange, dot, max, argmax, log, exp, sqrt, pi
from scipy import optimize as optim
from scipy.special import iv
from scipy.special import gamma as g
from scipy.special import gammaln as lng
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment as kuhn_munkres
from scipy.stats import skellam
from scipy.optimize import minimize
from scipy.integrate import quad, dblquad
import inspect
import multiprocessing as mp
import sys
import timeit
import random
# import non_tracking_extensions
# from tqdm import tqdm
from matplotlib import pylab as plt
import warnings

warnings.simplefilter("error")

setup = {
    'infer': 'non_tracking_03',
    'sampling': 'group',
    'arguments': {'cell_index': {}, 'dt': {}, 'p_off': {}, 's2': {}, 'D_bounds': {}, 'D0': {}, 'method': {}, \
                  'tol': {}, 'times': {}}
}
''' The argument passing is very clumsy and needs to be better integrated with existing code.
'''


def non_tracking_03(cells, dt=0.04, p_off=0., mu_on=0., s2=0.0025, D0=0.2, method='None',
                    tol=1e-3, times=[0.]):
    if method == 'None':
        raise ValueError("method must be given")
    try:
        useLq = method['useLq']
    except KeyError:
        useLq = False
    try:
        q = method['q']
    except KeyError:
        q = 0.5
    try:
        global_inference = method['global_inference']
    except KeyError:
        global_inference = False
    try:
        hex_phimu = method['hex_phimu']
        ring_phimu = method['ring_phimu']
    except KeyError:
        hex_phimu = ""
        ring_phimu = ""
    try:
        correct_p_off = method['correct_p_off']
    except KeyError:
        correct_p_off = False
    try:
        correct_p_off_MC = method['correct_p_off_MC']
    except KeyError:
        correct_p_off_MC = True
    try:
        tessellation = method['tessellation']
    except KeyError:
        tessellation = None
    try:
        dynamic_damping = method['dynamic_damping']
        dynamic_damping_delta = method['dynamic_damping_delta']
        dynamic_damping_gamma_min = method['dynamic_damping_gamma_min']
    except KeyError:
        dynamic_damping = False
        dynamic_damping_delta = None
        dynamic_damping_gamma_min = None
    try:
        MPA_max_distance_cutoff = method['MPA_max_distance_cutoff']
    except KeyError:
        MPA_max_distance_cutoff = np.inf
    inferrer = NonTrackingInferrer(cells=cells,
                                   dt=dt,
                                   tessellation=tessellation,
                                   gamma=method['gamma'],
                                   smoothing_factor=method['smoothing_factor'],
                                   optimizer=method['optimizer'],
                                   tol=method['tol'],
                                   epsilon=method['epsilon'],
                                   maxiter=method['maxiter'],
                                   phantom_particles=method['phantom_particles'],
                                   messages_type=method['messages_type'],
                                   chemPot=method['chemPot'],
                                   chemPot_gamma=method['chemPot_gamma'],
                                   chemPot_mu=method['chemPot_mu'],
                                   scheme=method['scheme'],
                                   method=method['method'],
                                   distribution=method['distribution'],
                                   temperature=method['temperature'],
                                   parallel=method['parallel'],
                                   hij_init_takeLast=method['hij_init_takeLast'],
                                   useLq=useLq,
                                   q=q,
                                   p_off=p_off, mu_on=mu_on,
                                   starting_diffusivities=method['starting_diffusivities'],
                                   starting_drifts=method['starting_drifts'],
                                   inference_mode=method['inference_mode'],
                                   cells_to_infer=method['cells_to_infer'],
                                   neighbourhood_order=method['neighbourhood_order'],
                                   minlnL=method['minlnL'],
                                   dynamic_damping=dynamic_damping,
                                   dynamic_damping_delta=dynamic_damping_delta,
                                   dynamic_damping_gamma_min=dynamic_damping_gamma_min,
                                   global_inference=global_inference,
                                   MPA_max_distance_cutoff=MPA_max_distance_cutoff,
                                   hex_phimu=hex_phimu,
                                   ring_phimu=ring_phimu,
                                   correct_p_off=correct_p_off,
                                   correct_p_off_MC=correct_p_off_MC,
                                   verbose=method['verbose'])
    try:
        inferrer.infer()
    except (SystemExit, KeyboardInterrupt):
        pass
    if method['inference_mode'] == "D":
        final_parameters = pd.DataFrame(inferrer._final_diffusivities, index=list(inferrer._cells.keys()),
                                        columns=['D'])
    elif method['inference_mode'] == "DD":
        final_diffusivities = pd.DataFrame(inferrer._final_diffusivities, index=list(inferrer._cells.keys()),
                                           columns=['D'])
        # final_drifts = pd.DataFrame(inferrer._final_drifts, index=list(inferrer._cells.keys()),
        #                            columns=['drift x', 'drift y'])
        final_parameters = final_diffusivities
        final_parameters['drift x'] = inferrer._final_drifts['dx']
        final_parameters['drift y'] = inferrer._final_drifts['dy']
    else:
        raise ValueError("Invalid value for inference_mode. Choose 'D' or 'DD'.")

    return final_parameters


''' v0.3 does solve Rac1 bug (empty cells)
    v0.3 does support massive parallelization at cluster level (separate SLURM jobs for each cell) using argument `cells_to_infer`
    v0.3 is syntactically closer to an eventual c++ version of the code
    The handling of particle count and drs distance matrices is not very pleasing yet
    drift management is horrendously confusing in the argument passing of the optimizer
'''


class FunctionEvaluation(Exception):
    def __init__(self, message):
        self.message = message


def Lq(x, q=1):
    """
        Implements the $L_q$ function.
    :param x: the argument of `Lq`
    :param q: the parameter `q`
    :return: The value of `Lq` applied to x with parameter `q`
    """
    if q == 1:
        return log(x)
    elif 0 < q < 1:
        return (x ** (1 - q) - 1) / (1 - q)
    else:
        raise ValueError(f"q={q} is out of bounds.")


def giants_causeway_plot(tessellation, Z):
    """
        A function that plots the diffusivity map in 3D
    :param tessellation:
    :param Z:
    :return:
    """
    # TODO


def expq(y, q=1):
    """
        Implements the inverse $L_q$ function
    :param y: The argument
    :param q: The parameter `q`
    :return: The value of the inverse $L_q$ function
    """
    if q == 1:
        return exp(y)
    elif 0 < q < 1:
        p = 1 / (1 - q)  # the conjugate of q, i.e the number such that (1/p + 1/q = 1)
        return (y / p + 1) ** p
    else:
        raise ValueError(f"q={q} is out of bounds.")


def logdotexp(a, b):
    """
        Computes log(dot(exp(a,b))) robustly. Taken from
        https://stackoverflow.com/questions/23630277/numerically-stable-way-to-multiply-log-probability-matrices-in-numpy
    :param a:
    :param b:
    :return:
    """
    max_a, max_b = np.max(a), np.max(b)
    exp_a, exp_b = a - max_a, b - max_b
    np.exp(exp_a, out=exp_a)
    np.exp(exp_b, out=exp_b)
    c = np.dot(exp_a, exp_b)
    np.log(c, out=c)
    c += max_a + max_b
    return c


class NonTrackingInferrer:
    __slots__ = ('_cells', '_dt', '_tessellation', '_p_off', '_mu_on', '_cells_to_infer', '_gamma', '_smoothing_factor',
                 '_tol', '_maxiter', '_temperature', '_chemPot_gamma', '_epsilon', '_chemPot_mu', '_q', '_minlnL',
                 '_neighbourhood_order', '_starting_diffusivities', '_starting_drifts', '_several_runs_maxiter',
                 '_several_runs_tolerance', '_optimizer', '_scheme', '_method', '_parallel', '_phantom_particles',
                 '_chemPot', '_messages_type', '_inference_mode', '_distribution', '_useLq', '_correct_p_off',
                 '_hex_phimu', '_ring_phimu', '_hij_init_takeLast', '_global', '_MPA_max_distance_cutoff',
                 '_cutoff_low_p_non', '_cutoff_low_Pij', '_cutoff_log_threshold', '_sparse_energy_computation',
                 '_sparse_energy_computation_sparsity', '_dynamic_damping', '_dynamic_damping_delta',
                 '_dynamic_damping_gamma_min', '_final_diffusivities', '_final_drifts', '_verbose', '_plot',
                 "_correct_p_off_MC")

    def __init__(self, cells, dt, tessellation=None, gamma=0.8, smoothing_factor=0, optimizer='NM', tol=1e-3,
                 epsilon=1e-8,
                 maxiter=10000, phantom_particles=1, messages_type='CLV', chemPot='None', chemPot_gamma=1, chemPot_mu=1,
                 scheme='1D', method='BP', distribution='gaussian', temperature=1, parallel=1, hij_init_takeLast=False,
                 useLq=False, q=0.5, p_off=0, mu_on=0, starting_diffusivities=[1], starting_drifts=[0, 0],
                 inference_mode='D', cells_to_infer='all', neighbourhood_order=1, minlnL=-100, dynamic_damping=False,
                 dynamic_damping_delta=None, dynamic_damping_gamma_min=None, global_inference=False,
                 MPA_max_distance_cutoff=np.inf, hex_phimu="", ring_phimu="", correct_p_off=False,
                 correct_p_off_MC=True, verbose=1):
        """
        :param cells: An object of type 'Distributed' containing the data tessellated into cells
        :param tessellation:
        :param gamma: The damping factor of the BP. h = gamma * h_old + (1-gamma) * h_new
        :param smoothing_factor: The coefficient used in the smoothing prior
        :param optimizer: The optimizer to use. Only Nelder-Mead 'NM' is supported
        :param tol: The tolerance for the optimizer. How precise an optimum do we require ?
        :param epsilon: The convergence tolerance for the BP. When two successive energies differ less than epsilon, we say that BP has converged
        :param maxiter: The maximal number of iterations to be done in the BP. If it is attained, BP stops and returns the last energy
        :param phantom_particles: (1) Phantom particles are introduced when necessary, or (0) No phantom particles are introduced
        :param messages_type: 'CLV': for classical messages
                              'JB' : for alternative update formulation
        :param chemPot: 'None': Don't use chemPot
                        'Chertkov': Use Chertkov chemPot
                        'Mezard': Use Mezard chemPot
        :param chemPot_gamma: The gamma parameter in the Mezard formulation of the Chemical Potential. Default is 1
        :param chemPot_mu: The mu parameter in the Chertkov formulation of the Chemical Potential. Default is 1
        :param scheme: The dimension-reducing optimization scheme. Can be '1D' or '2D'
        :param method: The marginalization method to use. Can be 'MPA' or 'BP'
        :param distribution: The likelihood-distribution to use. Can be 'gaussian' or 'rayleigh'
        :param temperature: The temperature of the BP (beta in the notes)
        :param parallel: Boolean. If `True`, we use the parallel implementation. If False, we use the sequential implementation
        :param hij_init_takeLast: Boolean. If True, then we initialize h_ij to its last value at the previous BP
        :param dt: The time step between frames
        :param p_off: The probability of a particle disappearing
        :param mu_on: The particle appearance intensity
        :param starting_diffusivities: The starting diffusivities serve only as an input variable. final_diffusivities is immediately initialized to starting_diffusivities
        :param starting_drifts: NOT YET IMPLEMENTED. The starting drifts serve only as an input variable. final_drifts is immediately initialized to starting_drifts.
        :param inference_mode: 'D' : Infer only diffusivity
                               'DD': Infer diffusivity and drift.
        :param final_diffusivities: NOT A PARAMETER The final diffusivities are the most up-to-date. Only to be changed at the end of an estimation
        :param final_drifts: NOT A PARAMETER The final drifts are the most up-to-date. Only to be changed at the end of an estimation
        :param cells_to_infer: An array of indices on which we want to infer
        :param neighbourhood_order: The order of the neighbourhoods for regional inference
        :param minlnL: The cut-off threshold for small probabilities
        :param hex_phimu: In the central hex of the comb use
                            'phi'  : phantom particles, no chemical potential
                            'mu'   : Chemical potential, no phantom particles
                            'phimu': Phantom particles and chemical potential
                            ''     : Neither phantom particles, nor chemical potential
                           Only if chemPot == 'phimu'
        :param ring_phimu: In the outer ring(s) of the comb, use
                            'phi'  : phantom particles, no chemical potential
                            'mu'   : Chemical potential, no phantom particles
                            'phimu': Phantom particles and chemical potential
                            ''     : Neither phantom particles, nor chemical potential
                           Only if chemPot == 'phimu'
        :param dynamic_damping: If `True` adapts the damping parameter dynamically
        :param dynamic_damping_delta: The discount factor for dynamic damping. A float between 0 (no memory) and 1 (perfect memory)
        :param dynamic_damping_gamma_min: The minimum threshold for the damping factor.
        :param global_inference: If `True`, the neighbourhood of each cell is the whole population of cells
        :param MPA_max_distance_cutoff: float, default value `inf`. Gives the maximum admissible distance for translocations.
        :param correct_p_off: (boolean) If `True`, use the p_off correction accounting for particles leaving the region
        :param correct_poff_MC: (boolean) If `True` and if `correct_p_off` is `True`, then poff correction is done via Monte-Carlo simulation and a mean-field approach. Otherwise, the correction is computed by numerical integration for each praticle using numerical integration.
        :param verbose: Level of verbosity.
                0 : mute (not advised)
                1 : introverted (advised)
                2 : extroverted
                3 : Don't stop talking !!
        """
        # Idea to reduce the number of parameters: Put diffusivity and drift together in a pandas dataframe
        # Problem data
        self._cells = cells
        self._tessellation = tessellation
        self._dt = dt
        self._p_off = p_off
        self._mu_on = mu_on
        if cells_to_infer == 'all':
            self._cells_to_infer = list(cells.keys())
        else:
            self._cells_to_infer = cells_to_infer
        # self._particle_count = self.particle_count()

        # Algorithm tuning parameters (numerical values)
        self._gamma = gamma
        self._smoothing_factor = smoothing_factor
        self._tol = tol
        self._maxiter = maxiter
        self._temperature = temperature
        self._chemPot_gamma = chemPot_gamma
        self._epsilon = epsilon
        self._chemPot_mu = chemPot_mu
        self._q = q
        self._minlnL = minlnL
        self._neighbourhood_order = neighbourhood_order
        self._MPA_max_distance_cutoff = MPA_max_distance_cutoff
        if self._MPA_max_distance_cutoff == 0:
            self.vprint(1, f"Warning: The cutoff distance is set to 0.")
        if len(starting_diffusivities) == 1:
            self._starting_diffusivities = starting_diffusivities[0] * np.ones(len(cells.keys()))
        else:
            self._starting_diffusivities = starting_diffusivities

        if len(starting_drifts) == 2:
            self._starting_drifts = np.tile(starting_drifts, (len(cells.keys()), 1))
        else:
            self._starting_drifts = starting_drifts
        self._several_runs_maxiter = 1
        self._several_runs_tolerance = 1

        # Methods to use (discrete values)
        self._optimizer = optimizer
        self._scheme = scheme
        self._method = method
        self._parallel = parallel
        self._phantom_particles = phantom_particles
        self._chemPot = chemPot
        self._messages_type = messages_type
        self._inference_mode = inference_mode
        self._distribution = distribution
        self._useLq = useLq
        self._correct_p_off = correct_p_off
        self._correct_p_off_MC = correct_p_off_MC

        # Phantom particles and chemical potential Hex/Ring combinations
        # possible values: "phi", "mu", "phimu", ""
        self._hex_phimu = hex_phimu
        self._ring_phimu = ring_phimu

        # Speed-up hacks
        self._hij_init_takeLast = hij_init_takeLast
        self._global = global_inference
        self._cutoff_low_p_non = True
        self._cutoff_low_Pij = True
        self._cutoff_log_threshold = -10
        self._sparse_energy_computation = True
        self._sparse_energy_computation_sparsity = 5
        self._dynamic_damping = dynamic_damping
        self._dynamic_damping_delta = dynamic_damping_delta
        self._dynamic_damping_gamma_min = dynamic_damping_gamma_min

        # Others
        self._final_diffusivities = self._starting_diffusivities
        self._final_drifts = self._starting_drifts
        self._verbose = verbose
        self._plot = False

        self.check_parameters()

        self.vprint(3, f"The cell volume is {self._cells[self._cells_to_infer[0]].volume}")

    def check_parameters(self):
        """
            This functions checks incoherent configurations and configurations that are not supported yet
        """
        if self._method == 'MPA':
            assert (self._phantom_particles is True)
            assert (self._chemPot == 'None')
        if self._chemPot == 'None' or self._chemPot == 'PhanPot':
            assert (self._phantom_particles is True)
        if self._chemPot == 'PhanPot':
            self.vprint(1, f"Warning: PhanPot mode is experimental")
        if self._distribution != 'gaussian' and self._distribution != 'rayleigh':
            raise ValueError("distribution not supported")
        if self._messages_type == 'JB':
            assert (self._method == 'BP')
            assert (self._chemPot == 'None')
            assert (self._phantom_particles is True)
            self.vprint(1, f"Warning: JB method is numerically unbstable")
        if self._inference_mode == 'DD':
            assert (self._scheme == '1D')
            assert (self._parallel is False)
        if self._useLq is True:
            assert (0 < self._q < 1)
            self.vprint(1, f"Warning: Lq method is numerically unstable")
        if self._global is True:
            self.vprint(1, f"Warning: global mode is experimental")
            assert (self._method == 'BP')
        if self._cutoff_low_Pij is True:
            assert (self._minlnL < self._cutoff_log_threshold)
        if self._correct_p_off is True:
            assert (self._distribution == 'gaussian')
            assert (self._tessellation is not None)
        if self._dynamic_damping is True:
            assert (self._chemPot == 'None' and (self._messages_type == 'CLV' or self._messages_type == 'Naive'))
        if self._correct_p_off_MC is False:
            assert (self._correct_p_off is True)
        if self._MPA_max_distance_cutoff is not np.inf:
            pass
            # self.vprint(1, f"Warning: The cutoff is not implemented yet.")

    def confirm_parameters(self):
        """
            prints a summary of the parameters of the inference. This can be useful when we store computed values on files. The log file then contains the information about the algorithm parameters.
        """
        print(f"Verbosity is {self._verbose}")
        self.vprint(1,
                    f"Inference is done with methods: \n*\t`method={self._method}` \n*\t`scheme={self._scheme}` \n*\t`optimizer={self._optimizer}` \n*\t`distribution={self._distribution}` \n*\t`parallel={self._parallel}` \n*\t`chemPot={self._chemPot}` \n*\t`messages_type={self._messages_type}` \n*\t`inference_mode={self._inference_mode}` \n*\t`phantom_particles={self._phantom_particles}` \n*\t`useLq={self._useLq}` \n*\t`global_inference={self._global}`")
        self.vprint(1,
                    f"The tuning is: \n*\t`gamma={self._gamma}` \n*\t`smoothing_factor={self._smoothing_factor}` \n*\t`tol={self._tol}` \n*\t`maxiter={self._maxiter}` \n*\t`temperature={self._temperature}` \n*\t`chemPot_gamma={self._chemPot_gamma}` \n*\t`chemPot_mu={self._chemPot_mu}` \n*\t`epsilon={self._epsilon}` \n*\t`minlnL={self._minlnL}` \n*\t`neighbourhood_order={self._neighbourhood_order}` \n*\t`q={self._q}` \n*\t`MPA_max_distance_cutoff={self._MPA_max_distance_cutoff}`")
        self.vprint(1,
                    f"Speed-ups are: \n*\t`dynamic_damping={self._dynamic_damping}` \n*\t`dynamic_damping_delta={self._dynamic_damping_delta}` \n*\t`dynamic_damping_gamma_min={self._dynamic_damping_gamma_min}` \n*\t`cutoff_low_p_non={self._cutoff_low_p_non}` \n*\t`cutoff_low_Pij={self._cutoff_low_Pij}` \n*\t`cutoff_log_threshold={self._cutoff_log_threshold}` \n*\t`sparse_energy_computation={self._sparse_energy_computation}` \n*\t`sparse_energy_computation_sparsity={self._sparse_energy_computation_sparsity}` \n*\t`hij_init_takeLast={self._hij_init_takeLast}`")
        self.vprint(1, f"Inference will be done on cells {self._cells_to_infer}")
        self.vprint(1, f"`p_off is {self._p_off},\tmu_on={self._mu_on}`")

    '''
    # TODO : check if this function is really necessary at this level. It gets overridden by the child
    def particle_count(self):
        """
            Used for __init__
        :return: The matrix of particle numbers in the cells first wrt cells, the wrt times
        """

        # Step 1: Compute the times
        r_total = []
        t_total = []
        S_total = 0
        index = list(self._cells.keys())
        for j in index:
            cell = self._cells[j]
            r = cell.r
            t = cell.t
            S = cell.volume
            r_total.extend(r)
            t_total.extend(t)
            S_total += S
        r_total = np.array(r_total)
        t_total = np.array(t_total)
        # List of times corresponding to frames:
        times = np.arange(min(t_total), max(t_total + self._dt / 100.), self._dt)

        # Step 2: Compute the particle number
        particle_number = np.zeros( (len(index),len(times)) )
        print(f"index={index}")
        for j in index:
            cell = self._cells[j]
            r = cell.r
            t = cell.t
            frames_j = self.rt_to_frames(r, t, times)
            N_j = [len(frame) for frame in frames_j]
            particle_number[j,:] = N_j

        return particle_number
    '''

    def infer(self):
        """
            Does the inference in the self._cells_to_infer and stores the result in self._final_diffusivities
        """
        self.confirm_parameters()
        # PARALLEL VERSION with mp #
        if self._parallel:
            n_cores = mp.cpu_count()
            pool = mp.Pool(n_cores)
            self.vprint(1, f"Using {n_cores} cpus")
            result_objects = [pool.apply_async(self.estimate, args=(i,)) for i in
                              self._cells_to_infer]  # How to handle this with drift ?
            pool.close()
            pool.join()
            results = [r.get() for r in result_objects]
            self._final_diffusivities = pd.DataFrame(results, index=list(self._cells_to_infer), columns=['D'])
            # Check : Is index correct ?

        # SEQUENTIAL VERSION #
        elif self._parallel is False:
            """ In the sequential method for the 1D scheme, the inferred value in cell i is assumed as the estimated
                diffusivity of cell i when inferring cells j>i. In the parallel method, these optimizations are
                independent.
            """
            for i in self._cells_to_infer:
                if self._inference_mode == 'DD':
                    self._final_diffusivities[i], self._final_drifts[i, :] = self.estimate(i)
                elif self._inference_mode == 'D':
                    self._final_diffusivities[i] = self.estimate(i)
                self.vprint(2, f"Current parameters with tol = {self._tol}")
                self.vprint(2, f"diffusivities={self._final_diffusivities}")
                if self._inference_mode == 'DD':
                    self.vprint(2, f"drifts={self._final_drifts}")
            self._final_diffusivities = pd.DataFrame(self._final_diffusivities, index=list(self._cells.keys()),
                                                     columns=['D'])
            self._final_drifts = pd.DataFrame(self._final_drifts, index=list(self._cells.keys()),
                                              columns=['dx', 'dy'])
            # Check : Is index correct ?
            if self._inference_mode == 'D' and self._several_runs_maxiter > 1:
                for run in range(self._several_runs_maxiter):
                    self.vprint(1, f"\n--------------- 1D run number {run} ---------------\n")
                    self._final_diffusivities_old = self._final_diffusivities
                    for i in self._cells_to_infer:
                        if self._inference_mode == 'DD':
                            self._final_diffusivities[i], self._final_drifts[i, :] = self.estimate(i)
                        elif self._inference_mode == 'D':
                            self._final_diffusivities[i] = self.estimate(i)
                        self.vprint(2, f"Current parameters with tol = {self._tol}")
                        self.vprint(2, f"diffusivities={self._final_diffusivities}")
                        if self._inference_mode == 'DD':
                            self.vprint(2, f"drifts={self._final_drifts}")
                    # self._final_diffusivities = pd.DataFrame(self._final_diffusivities, index=list(self._cells.keys()),
                    #                                         columns=['D'])
                    self._final_drifts = pd.DataFrame(self._final_drifts, index=list(self._cells.keys()),
                                                      columns=['dx', 'dy'])
                    difference = sum(abs(self._final_diffusivities['D'] - self._final_diffusivities_old['D']))
                    self.vprint(1, f"difference = {difference}")
                    if difference < self._several_runs_tolerance:
                        break
            else:
                pass
        else:
            raise ValueError("Invalid argument: parallel")
        self.vprint(1, f"Final diffusivities are:\n {self._final_diffusivities}")
        if self._inference_mode == 'DD':
            self.vprint(1, f"Final drifts are:\n {self._final_drifts}")

    def estimate(self, i):
        """
            Does a local inference of properties in cell of index i
        :param i: The index of the cell to estimate
        :return: The estimated diffusivity
        """
        self.vprint(2, f"\nCell no.: {i}")

        region_indices = self.get_neighbours(i, order=self._neighbourhood_order)
        region_indices = np.insert(region_indices, 0, i)  # insert i in the 0th position

        # --- Fit: ---
        fittime = time()
        if self._scheme == '1D':
            D = self._starting_diffusivities[i]
            d = self._starting_drifts[i, :]
        elif self._scheme == '2D':
            D = array([self._starting_diffusivities[i], self._starting_diffusivities[i]])
            d = np.hstack((self._starting_drifts[i, :], self._starting_drifts[i, :]))
        else:
            raise ValueError("This scheme is not supported. Choose '1D' or '2D'.")
        D_in = self._starting_diffusivities[i]
        drift_in = self._starting_drifts[i, :]

        # Select the BP version
        """ On this point we create a specialized copy of self.
            Its attributes have the same values as self.
            There surely is a simpler way of passing the parent attributes to the child, but I did not find it. Sorry :/
        """
        parent_attributes = (
            self._cells, self._dt, self._tessellation, self._gamma, self._smoothing_factor, self._optimizer, self._tol,
            self._epsilon, self._maxiter, self._phantom_particles, self._messages_type, self._chemPot,
            self._chemPot_gamma, self._chemPot_mu, self._scheme, self._method, self._distribution, self._temperature,
            self._parallel, self._hij_init_takeLast, self._useLq, self._q, self._p_off, self._mu_on,
            self._starting_diffusivities, self._starting_drifts, self._inference_mode, self._cells_to_infer,
            self._neighbourhood_order, self._minlnL, self._dynamic_damping, self._dynamic_damping_delta,
            self._dynamic_damping_gamma_min, self._global, self._MPA_max_distance_cutoff, self._hex_phimu,
            self._ring_phimu, self._correct_p_off, self._correct_p_off_MC, self._verbose)
        if self._chemPot == 'Chertkov':
            local_inferrer = NonTrackingInferrerRegionBPchemPotChertkov(parent_attributes, i, region_indices)
        elif self._chemPot == 'Mezard':
            self.vprint(2, "NonTrackingInferrerRegionBPchemPotMezard created")
            local_inferrer = NonTrackingInferrerRegionBPchemPotMezard(parent_attributes, i, region_indices)
        elif self._chemPot == 'PhanPot':
            local_inferrer = NonTrackingInferrerRegionBPchemPotPartialChertkov(parent_attributes, i, region_indices)
        elif self._chemPot == "phimu":
            local_inferrer = NonTrackingInferrerRegionBPchemPotPartialChertkov(parent_attributes, i, region_indices)
        elif self._chemPot == 'None' and self._messages_type == 'Naive' and self._global is False:
            local_inferrer = NonTrackingInferrerRegionNaive(parent_attributes, i, region_indices)
        elif self._chemPot == 'None' and self._messages_type == 'CLV' and self._global is False:
            local_inferrer = NonTrackingInferrerRegion(parent_attributes, i, region_indices)
        elif self._chemPot == 'None' and self._messages_type == 'CLV' and self._global is True:
            local_inferrer = NonTrackingInferrerRegionGlobal(parent_attributes, i, region_indices)
        elif self._chemPot == 'None' and self._messages_type == 'JB':
            local_inferrer = NonTrackingInferrerRegionBPalternativeUpdateJB(parent_attributes, i, region_indices)
        else:
            raise ValueError(
                "chemPot or messages_type not valid. Suggestion: Select chemPot='None' and messages_type='CLV'.")

        # import pdb; pdb.set_trace()
        try:
            if self._optimizer == 'NM':
                if self._inference_mode == 'D':
                    fit = minimize(local_inferrer.smoothed_posterior, x0=D, method='Nelder-Mead', tol=self._tol,
                                   options={'disp': False, 'maxiter': 50, 'xatol': 1e-2, 'fatol': 1e-2})
                elif self._inference_mode == 'DD':
                    start = np.hstack((D, d))
                    fit = minimize(local_inferrer.smoothed_posterior, x0=start, method='Nelder-Mead', tol=self._tol,
                                   options={'disp': True})
                    if self._scheme == '1D':
                        drift_in = (fit.x)[1:3]
                    elif self._scheme == '2D':
                        drift_in = (fit.x)[2:4]
                    else:
                        raise ValueError("Wrong scheme. Choose 1D or 2D")
                fit_time = local_inferrer.fit_time
                iterations = local_inferrer.iterations
                runtime_error_count = local_inferrer.runtime_error_count
                if local_inferrer.maxiter_attained_counter > 0:
                    self.vprint(3,
                                f"Warning: maxiter={self._maxiter} attained {local_inferrer.maxiter_attained_counter} times during the optimization.")
                    self.vprint(3, f"Suggestions: Increase gamma. Choose a more robust method.")
            else:
                raise ValueError("This optimizer is not supported. Suggestion: Choose 'NM'. ")
            D_in = abs((fit.x)[0])
            self.vprint(2, f"fit = {fit}")
        except FunctionEvaluation as error:
            self.vprint(1, f"Warning: Function Evaluation failed.")
            self.vprint(1, error.message)
            D_in = self._starting_diffusivities[i]
            # drift_in = np.array([np.nan,np.nan])
            drift_in = self._starting_drifts[i, :]
        # import pdb; pdb.set_trace()
        # D_i_ub=max([1.5*(D_i-s2/dt),0.])
        # print("D corrected for motion blur and localization error:", D_i_ub)
        self.vprint(1, f"\nCell no.: {i}")
        self.vprint(1, f"{time() - fittime}s")  # make this an attribute to save it in a file later ?
        self.vprint(1, f"Estimation: D_in: {D_in}\t drift_in: {drift_in}")

        if self._inference_mode == 'D':
            return D_in
        elif self._inference_mode == 'DD':
            return D_in, drift_in

    def get_neighbours(self, cell_index, order=1):
        """ Uses recursive calls
            Computes the indices of the neighbours of cell cell_index. Neighbours of order 1 are just neighbours. Neighbours of order 2 also include neighbours of neighbours. Neighbours of order 3 also include neighbours of neighbours of neighbours, and so on...
        :param cell_index: The index of the central cell
        :param order: The order of the neighbourhood
        :return: An array or list (not sure) of the indices of the neighbour cells
        """
        if order == 'all':
            neighbours = list(self._cells.keys())
            neighbours.remove(cell_index)
            return neighbours
        elif order == 0:
            return []
        elif order == 1:
            return self._cells.neighbours(cell_index)
        else:
            neighbours = []
            for n in self._cells.neighbours(cell_index):
                neighbours = np.hstack((neighbours, self.get_neighbours(n, order=order - 1)))
            neighbours = list(dict.fromkeys(list(neighbours)))  # remove duplicates
            neighbours.remove(cell_index)
            neighbours = list(map(int, neighbours))
            return array(neighbours)

    # -----------------------------------------------------------------------------
    # Generate stack of frames
    # -----------------------------------------------------------------------------
    def rt_to_frames(self, r, t, times):
        """
        :param r: The list of positions
        :param t: The list of times
        :param times: The times array, from minimal to maximal time with adequate step
        :return: The list of frames corresponding to times
        """
        frames = []
        for t_ in times:
            frames.append(np.array(r[(t < t_ + self._dt / 100.) & (t > t_ - self._dt / 100.)]))
        return frames

    def local_distance_matrices(self, region_indices):
        """
        :param region_indices: The indices of the cells in the region to consider
        :return: Should return a list. Element number i contains 2 matrices: The first is the position difference between particles in frame i+1 and i in the x-axis. The second matrix is the same, but on the y-axis.
        """
        r_total = []
        t_total = []
        S_total = 0
        for j in region_indices:
            cell = self._cells[j]
            r = cell.r  # the recorded positions in cell
            t = cell.t  # the recorded times in cell
            S = cell.volume
            r_total.extend(r)
            t_total.extend(t)
            S_total += S
        r_total = np.array(r_total)  # r_total contains all the position in the region
        t_total = np.array(t_total)  # t_total contains the corresponding times in the same order
        # List of times corresponding to frames:
        times = np.arange(min(t_total), max(t_total + self._dt / 100.), self._dt)
        # Build lists of frames corresponding to times. A frame is an array of positions:
        frames = self.rt_to_frames(r_total, t_total, times)
        drs = [self.positions_to_distance_matrix(frames[n], frames[n + 1]) for n in range(len(frames) - 1)]
        return drs

    # -----------------------------------------------------------------------------
    # Helper functions for tracking/non-tracking
    # -----------------------------------------------------------------------------
    def positions_to_distance_matrix(self, frame1, frame2):
        '''Matrix of position differences between all measured points in two succesive frames.'''
        N_ = len(frame1)
        M_ = len(frame2)
        dr_ij = np.zeros([2, N_, M_])
        for i in range(N_):
            dr_ij[0, i] = frame2[:, 0] - frame1[i, 0]
            dr_ij[1, i] = frame2[:, 1] - frame1[i, 1]
        return dr_ij

    def vprint(self, level, string, end_='\n'):
        if self._verbose == 0:
            pass
        elif self._verbose >= level:
            print(string, end=end_)
        else:
            pass


""" If we are to store region-dependent variables as attributes, for the sake of easy access, we have to do so in a separate class, because parallelism would otherwise cause interference between regions.
Hence the working parameters need to be in the child class
Within here, we cannot change the attributes of the calling motherclass
Within here, we consider a fixed diffusivity
"""


class NonTrackingInferrerRegion(NonTrackingInferrer):
    __slots__ = ('_index_to_infer', '_region_indices', '_drs', '_region_area', '_particle_count', '_hij', '_hji',
                 'optimizer_first_iteration', 'maxiter_attained_counter', '_region_mu_on', '_index_to_infer_mu_on',
                 '_parent_p_off', '_working_diffusivities', '_working_drifts', 'Gamma_n', 'fit_time', 'iterations',
                 'runtime_error_count')

    def __init__(self, parentAttributes, index_to_infer, region_indices):
        super(NonTrackingInferrerRegion, self).__init__(
            *parentAttributes)  # need to get all attribute values from parent class
        self._index_to_infer = index_to_infer
        self._region_indices = region_indices.astype(int)
        assert (self._index_to_infer == self._region_indices[0])
        # self._diffusivityMatrix = self.buildDiffusivityMatrix() # maybe not needed yet
        # print(f"self._gamma : {self._gamma}")
        self._drs = self.local_distance_matrices(region_indices)

        area = 0
        for i in self._region_indices:
            cell = self._cells[i]
            area += cell.volume
        self._region_area = area
        self._particle_count = self.particle_count()

        self._hij = [None] * self._particle_count.shape[1]
        self._hji = [None] * self._particle_count.shape[1]
        for f in arange(len(self._hij) - 1):
            # import pdb; pdb.set_trace()
            M = np.sum(self._particle_count[self._region_indices, f + 1]).astype(int)
            self._hij[f] = [None] * (M + 1)
            self._hji[f] = [None] * (M + 1)

        self.optimizer_first_iteration = True
        self.maxiter_attained_counter = 0

        self._region_mu_on = self._mu_on * np.sum(self._particle_count[self._region_indices, :]) / sum(
            self._particle_count)
        self.vprint(2, f"region_mu_on={self._region_mu_on}")
        self._index_to_infer_mu_on = self._mu_on * np.sum(self._particle_count[self._index_to_infer, :]) / sum(
            self._particle_count)

        self._parent_p_off = self._p_off

        self.Gamma_n = 0  # unused. I don't remember what it was for... Ah yes, it was the Gamma_n from the report

        self.fit_time = np.nan  # saves the time it took after the optimization
        self.iterations = np.nan  # saves the number of iterations after the optimization
        self.runtime_error_count = np.nan  # saves the count of runtime errors encountered

    # hij, hji getters and setters #
    def get_hij(self, frame, non):
        """
        :param frame: The index of the first frame of the transisiton
        :param noff: The number of disappearing particles in first frame
        :return: The currently stored `hij` matrix for frame transition `frame` to `frame+1`
        """
        try:
            return self._hij[frame][non]
        except IndexError:
            print(self._hij[frame])
            # import pdb; pdb.set_trace()
            raise IndexError("hij index error")

    def set_hij(self, hij, frame, non):
        """
            Sets a value for `hij`
        :param frame: The index of the first frame of the transisiton
        :param noff: The number of disappearing particles in first frame
        """
        try:
            self._hij[frame][non] = hij
        except IndexError:
            pass
            # import pdb; pdb.set_trace()

    def get_hji(self, frame, non):
        """
        :param frame: The index of the first frame of the transisiton
        :param noff: The number of disappearing particles in first frame
        :return: The currently stored `hji` matrix for frame transition `frame` to `frame+1`
        """
        return self._hji[frame][non]

    def set_hji(self, hji, frame, non):
        """
            Sets a value for `hji`
        :param frame: The index of the first frame of the transisiton
        :param noff: The number of disappearing particles in first frame
        """
        self._hji[frame][non] = hji

    # p_off correction MC
    def draw_p1(self, size=1):
        """
            Draws a point in the region from the uniform distribution
            1. get cell center
            2. get length of the Spokes
            3. draw a Spoke uniformly (Not drawing a position directly keeps us independent from the inclination of the hex)
            4. draw a random position from the uniform triangle
            The uniform triangle having the same shape, this can be done for N draws efficiently
            WARNING: This code relies on an implicit order in the vertices.
                        This order gets transformed in a more natural order as shown below
                        But if the initial order is not anymore as drawn here, the code will yield wrong results
                            5                      3
                        2       4      ->      2       4
                        1       3              1       5
                            0                      0
        :param size: (int) number of points to ba drawn
        :return: a 2-d np.array: the coordinate vector of the drawn point,
                 an array of hexes in which the drawn points lie
        """
        # Generate uniform points in the unit square
        x_square = np.random.uniform(low=0, high=1, size=size)
        y_square = np.random.uniform(low=0, high=1, size=size)

        # Fold the points into the triangle (half the square)
        x_half_square = np.minimum(x_square, 1 - y_square)
        y_half_square = np.minimum(1 - x_square, y_square)
        x_half_square = x_half_square.reshape((-1, 1))
        y_half_square = y_half_square.reshape((-1, 1))
        # Generates hexes
        hexes = np.random.choice(self._region_indices, size=size, replace=True)
        cell_centers = self._tessellation.cell_centers
        hex_centers = cell_centers[hexes]

        # Preliminary formatting of vertex data
        cell_vertices_as_list = [v for k, v in self._tessellation.cell_vertices.items()]
        cell_vertices_as_array = np.asarray(cell_vertices_as_list)
        cell_vertices_as_array.T[[3, 5]] = cell_vertices_as_array.T[[5, 3]]  # exchange the columns to fix the order

        # Generates spokes
        spokes = np.random.randint(low=0, high=6, size=size)
        spokes_plus_one = (spokes + 1) % 6
        hex_vertex_indices = cell_vertices_as_array[hexes, array([spokes, spokes_plus_one])].T
        hex_chosen_vertices_y = self._tessellation.vertices[hex_vertex_indices[:, 0]]
        hex_chosen_vertices_x = self._tessellation.vertices[hex_vertex_indices[:, 1]]

        # Synthetize into p1 coordinates
        p = hex_centers \
            + (hex_chosen_vertices_x - hex_centers) * np.hstack((x_half_square, x_half_square)) \
            + (hex_chosen_vertices_y - hex_centers) * np.hstack((y_half_square, y_half_square))
        if self._plot is True:
            fig = plt.figure()
            plt.plot(p[:, 0], p[:, 1], 'g,', alpha=0.5)
            plt.plot(cell_centers[:, 0], cell_centers[:, 1], 'k+')
            for i in range(cell_vertices_as_array.shape[0]):
                plt.plot((self._tessellation.vertices[cell_vertices_as_array[i, :]])[:, 0],
                         (self._tessellation.vertices[cell_vertices_as_array[i, :]])[:, 1], 'r-')
        return p, hexes

    def draw_p2(self, p1, hexes, size=1):
        """
            Draws a p2 vector from p1
        :param p1: The position vector from which to translocate
        :param hexes: The indices of the hexes in which the points p1 lie
        :param size: Number of translocations
        :return: a 2-d np.array: The translocated vector
        """

        # draw N(0,1) samples
        x_scaled = np.random.normal(loc=0, scale=1, size=size)
        y_scaled = np.random.normal(loc=0, scale=1, size=size)
        x_scaled = x_scaled.reshape((-1, 1))
        y_scaled = y_scaled.reshape((-1, 1))

        # multiply by standard deviation
        diffusivities = self._working_diffusivities[hexes]
        diffusivities = diffusivities.reshape((-1, 1))
        x = x_scaled * sqrt(2 * diffusivities * self._dt)
        y = y_scaled * sqrt(2 * diffusivities * self._dt)
        p_transloc = np.hstack((x, y))

        # add
        p2 = p1 + p_transloc

        # if self._plot is True:
        # plt.plot(p2[:, 0], p2[:, 1], 'b.', alpha=0.5)

        return p2

    def is_in_cell(self, cell_index, points):
        """
            Checks if point `point` is in the domain of the cell `cell_index`

        :param cell_index: Index of the cell to test
        :param points: array of points to test
        :return: (boolean) True if point is in cell
        """
        center = self._tessellation.cell_centers[cell_index]
        tilt = self._tessellation.tilt  # not used yet. But should not influence the final result for current usage
        radius = self._tessellation.hexagon_radius
        # Use hexagon with flat top and bottom orientation $\hexagon$.
        diff_scaled = (points - center) / radius
        x, y = diff_scaled[:, 0], diff_scaled[:, 1]
        # Test membership of the point in a partition of the hexagon
        truth1 = (abs(y) <= 1 / sqrt(3)) * (abs(x) <= 1)  # middle rectangle
        truth2 = (1 / sqrt(3) <= abs(y)) * (abs(y) <= 2 / sqrt(3)) * (
                abs(x) <= 2 - abs(y) * sqrt(3))  # left and right triangles
        truth = truth1 * 1 + truth2 * 1 - truth1 * truth2 * 1  # logical or
        return truth == 1

    def is_in_region(self, p2):
        """
            Checks if a point of coordinates (x,y) is in the region
        :param p2: 2d-array: The points to test
        :return: True if the point is inside the region
        """
        truth = np.ones(p2.shape[0]) * False
        for i in self._region_indices:
            truth_i = self.is_in_cell(i, p2) * 1
            truth = truth + truth_i - truth * truth_i
        truth = (truth == 1)
        if self._plot is True:
            plt.plot(p2[~truth, 0], p2[~truth, 1], 'b.')
            plt.plot(p2[truth, 0], p2[truth, 1], 'r.')
            plt.show()
        return truth

    def compute_p_out(self, N_draws=100000):
        """
            Computes the value of p_out, i.e the probability that a particle a priori uniformly distributed in the comb goes out of it.
        :param N_draws: The sample size
        :return: a floating value between 0 and 1
        """
        p1, hexes = self.draw_p1(N_draws)
        p2 = self.draw_p2(p1, hexes, N_draws)
        pin = sum(self.is_in_region(p2)) / len(p2)
        assert (0 <= pin <= 1)
        return 1 - pin

    def compute_corrected_p_off(self):
        """
            Computes p_off including the correction due to segmenting the domain into regions
        :return: (float) corrected p_off
        """
        p_out = self.compute_p_out()
        return 1 - (1 - self._parent_p_off) * (1 - p_out)

    # p_off correction integration
    def find_xmin_xmax(self, i):
        """
            Find the minimal and maximal x-coordinate of cell i
        :param i: index of the cell
        :return: The minimum and maximum coordinates
        """
        xmin = self._tessellation.cell_centers[i] - self._tessellation.hexagon_radius
        xmax = self._tessellation.cell_centers[i] - self._tessellation.hexagon_radius
        return xmin, xmax

    def ymin(self, i, x):
        """
            Computes the minimum y-coordinate in cell i at slice x
        :param i: index of the cell
        :param x: x-coordinate of the slice
        :return: The minimum y-coordinate
        """
        slope = 2 * self._tessellation.hexagon_radius / sqrt(3)
        centre = self._tessellation.cell_centers[i]
        if x <= centre:
            intercept = centre[1] - 2 * self._tessellation.hexagon_radius / sqrt(3) \
                        - slope * centre[0]
            return intercept - slope * x
        elif x >= centre:
            intercept = centre[1] - 2 * self._tessellation.hexagon_radius / sqrt(3) \
                        - slope * centre[0]
            return intercept + slope * x

    def ymax(self, i, x):
        """
            Computes the maximum y-coordinate in cell i at slice x
        :param i: index of the cell
        :param x: x-coordinate of the slice
        :return: The maximum y-coordinate
        """
        slope = 2 * self._tessellation.hexagon_radius / sqrt(3)
        centre = self._tessellation.cell_centers[i]
        if x <= centre:
            intercept = centre[1] + 2 * self._tessellation.hexagon_radius / sqrt(3) \
                        - slope * centre[0]
            return intercept + slope * x
        elif x >= centre:
            intercept = centre[1] + 2 * self._tessellation.hexagon_radius / sqrt(3) \
                        - slope * centre[0]
            return intercept - slope * x

    def pout_given_p1(self, p1):
        """
            Computes the probability of going out of the region in `self._dt` seconds, given an initial position p1
        :param p1: The initial position
        :return: The probability of going out of the region
        """
        for i in self._region_indices:
            if self.is_in_cell(i, p1):
                cell_index = i
        try:
            D = self._final_diffusivities[cell_index]
        except KeyError:
            self.vprint(1, f"KeyError. p1 is contained in no cell of the region.")
        I_array = []
        for i in self._region_indices:
            xmin, xmax = self.find_xmin_xmax(i)
            Ii = dblquad(
                lambda x, y: exp(-((x - p1[0]) ** 2 + (y - p1[1]) ** 2) / (4 * D * self._dt)) / (4 * pi * D * self._dt),
                xmin, xmax, lambda x: self.ymin(i, x), lambda x: self.ymax(i, x))
            I_array.append(Ii)
        I = sum(I_array)
        return 1 - I

    def corrected_poff_array(self, frame_index):
        """
            Computes the corrected poff* array of the points in frame `frame_index` using numerical integration
        :param frame_index: The index of the concerned frame
        :return: array of exit probabilities
        """
        r_total = []
        t_total = []
        for j in self._region_indices:
            cell = self._cells[j]
            r = cell.r  # the recorded positions in cell
            t = cell.t  # the recorded times in cell
            r_total.extend(r)
            t_total.extend(t)
        r_total = np.array(r_total)  # r_total contains all the position in the region
        t_total = np.array(t_total)  # t_total contains the corresponding times in the same order
        # List of times corresponding to frames:
        times = np.arange(min(t_total), max(t_total + self._dt / 100.), self._dt)
        # Build lists of frames corresponding to times. A frame is an array of positions:
        frames = self.rt_to_frames(r_total, t_total, times)
        positions = frames[frame_index]

        corrected_poff_array = []
        for p1 in positions:
            corrected_poff = 1 - (1 - self._parent_p_off) * (1 - self.pout_given_p1(p1))
            corrected_poff_array.append(corrected_poff)

        return corrected_poff_array

    # Other helper functions #
    def particle_count(self):
        """
            Used for __init__ of nonTrackingInferrerRegion
        :return: The matrix of particle numbers in the cells. First w.r.t cells, then w.r.t times
        """

        # Step 1: Compute the times
        r_total = []
        t_total = []
        S_total = 0
        index = list(self._cells.keys())
        for j in self._region_indices:
            cell = self._cells[j]
            r = cell.r
            t = cell.t
            S = cell.volume
            r_total.extend(r)
            t_total.extend(t)
            S_total += S
        r_total = np.array(r_total)
        t_total = np.array(t_total)
        # List of times corresponding to frames:
        times = np.arange(min(t_total), max(t_total + self._dt / 100.), self._dt)

        # Step 2: Compute the particle number matrix
        particle_number = np.zeros((len(index), len(times)))
        print(f"region_indices={self._region_indices}")
        for j in index:
            cell = self._cells[j]
            r = cell.r
            t = cell.t
            frames_j = self.rt_to_frames(r, t, times)
            N_j = [len(frame) for frame in frames_j]
            particle_number[j, :] = N_j

        return particle_number

    def estimate_mu(self):
        """
            Estimates the common mu parameter for the Poisson distributions of appearance and disappearances
        :return: (float) an estimate for a global mu using the method of moments
        """
        N_particles_array = np.sum(self._particle_count, axis=0).astype(int)
        Delta_array = N_particles_array[1:len(N_particles_array)] - N_particles_array[0:(len(N_particles_array) - 1)]
        ME = np.mean(Delta_array ** 2 / 2)
        return ME

    def estimate_muon_muoff(self):
        """
            Estimates muon and muoff jointly using maximum likelihood
        :return: 2 x (float) the estimates
        """
        # TODO

    def p_m(self, x, Delta, N):
        '''Poissonian probability for x particles to appear between frames given Delta=M-N
        and blinking parameters mu_on and mu_off. Consult the article for reference'''
        mu_off = N * self._p_off
        return (self._region_mu_on * mu_off) ** (x - Delta / 2.) / (
            g(x + 1.) * g(x + 1. - Delta) * iv(Delta, 2 * sqrt(self._region_mu_on * mu_off)))

    def lnp_m(self, x, Delta, N):
        """
            The logarithm of `p_m`
        :param x: The value of which we want to evaluate the probability
        :param Delta: The observed particle number difference
        :param N: The number of particles in the first frame
        :return:
        """
        mu_off = N * self._p_off
        return log(self._region_mu_on * mu_off) * (x - Delta / 2.) - lng(x + 1.) - lng(x + 1. - Delta) \
               - log(iv(Delta, 2 * sqrt(self._region_mu_on * mu_off)))

    def lnp_ij(self, dr, frame_index):
        """
            Computes the log-probability matrix of displacements
        :param dr: the displacement vectors
        :param frame_index: The index of the current frame
        :return: Matrix of log-probabilities for each distance.
        """
        D, drift_x, drift_y = self.build_diffusivity_drift_matrices(frame_index)
        try:
            if self._distribution == 'gaussian':
                lnp = - log(4 * pi * D * self._dt) \
                      - ((dr[0] - drift_x * self._dt) ** 2 + (dr[1] - drift_y * self._dt) ** 2) / (4. * D * self._dt)
            elif self._distribution == 'rayleigh':
                r2 = (dr[0] - drift_x * self._dt) ** 2 + (dr[1] - drift_y * self._dt) ** 2
                lnp = log(sqrt(r2)) - log(2. * D * self._dt) - r2 / (4. * D * self._dt)
            else:
                raise ValueError("distribution not supported. Suggestion: Use 'gaussian'. ")
            lnp[lnp < self._minlnL] = self._minlnL  # avoid numerical underflow
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot have 0 diffusivity --> ZeroDivisionError")
        except RuntimeWarning:
            pass
            # import pdb; pdb.set_trace()
            if np.any(np.isnan(lnp)):
                raise FunctionEvaluation(f"nan appeared in lnp_ij")
                pass
                # import pdb; pdb.set_trace()
        if self._cutoff_low_Pij is True:
            lnp[lnp < self._cutoff_log_threshold] = self._minlnL
        if self._method == 'MPA':
            lnp[sqrt(dr[0] ** 2 + dr[1] ** 2) > self._MPA_max_distance_cutoff] = self._minlnL
        return lnp

    def P_ij(self, n_off, n_on, dr, frame_index):
        """
            Computes the probability matrix of displacements. It is the non-logarithmic version of Q_ij
        :param dr: the displacement vectors
        :param frame_index: The index of the current frame
        :return: Matrix of probabilities for each distance.
        """
        Q_ij = self.Q_ij(n_off, n_on, dr, frame_index)
        P_ij = exp(Q_ij)
        if self._cutoff_low_Pij is True:
            P_ij[Q_ij < self._cutoff_log_threshold] = 0
        return P_ij

    def minus_lq_minus_lnp_ij(self, dr, frame_index):
        """
            Computes either the log-probability matrix or its -Lq(-.) transform
        :param dr: the displacement vectors
        :param frame_index: The index of the current frame
        :return: Matrix of log-probabilities for each distance or its -Lq(-.) transform
        """
        if self._useLq is True:
            # import pdb; pdb.set_trace()
            return -Lq(-self.lnp_ij(dr, frame_index), self._q)
        else:
            return self.lnp_ij(dr, frame_index)

    def Q_ij(self, n_off, n_on, dr, frame_index):
        """
        Matrix of log-probabilities for assignments with appearances and disappearances.
        Structure of Q_ij:
         _                        _
        | Q_11  Q_12 .. Q_1M | 1/S |
        | Q_21  Q_22 .. Q_2M | 1/S |
        |  :     :       :   |  :  |
        | Q_N1  Q_N2 .. Q_NM | 1/S |
        |--------------------------|
        | 1/S   1/S  .. 1/S  |  0  |
        |_1/S   1/S  .. 1/S  |  0 _|
        :param n_off: The number of disappearing particles
        :param n_on:  The number of appearing particles
        :param dr:  The displacement vectors
        :param frame_index: The index of the current frame
        :return: The square matrix Q_ij of size N + n_on - n_off
        """
        n_on = int(n_on)
        n_off = int(n_off)
        N = np.sum(self._particle_count[self._region_indices, frame_index])
        N = int(N)
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1])
        M = int(M)
        if self._phantom_particles:
            assert (M == N - n_off + n_on)  # just a consistency check
        out = zeros([n_on + N, n_off + M])
        LM = ones([n_on + N, n_off + M]) * self._region_area
        lnp = self.minus_lq_minus_lnp_ij(dr, frame_index)
        out[:N, :M] = lnp - log(LM[:N, :M])  # / (1-self._p_off))
        if self._correct_p_off_MC is True:
            out[:N, M:] = -log(LM[:N, M:])  # / self._p_off)
            out[N:, :M] = -log(LM[N:, :M])  # / self._p_off)
        elif self._correct_p_off_MC is False:
            out[:N, M:] = -log(LM[:N, M:])  # / self.corrected_poff_array(frame_index))
            out[N:, :M] = -log(LM[N:, :M])  # / self.corrected_poff_array(frame_index + 1))
        else:
            raise ValueError(f"Invalid value for `correct_p_off_MC`. Please choose either `True` or `False`.")
        out[N:, M:] = 2. * self._minlnL  # Note: Why not 0 ??
        return out

    def build_diffusivity_drift_matrices(self, frame_index):
        """
            For internal use only. Needs to be processed into a likelihood matrix and completed with phantom particles
        :param frame_index: the index of the first frame of the transition
        :return: Three N x M matrices containing in each line i the (*) of particle i.
                    First matrix (*) = diffusivity
                    Second matrix (*) = drift in x-axis
                    Third matrix (*) = drift in y-axis
        """
        M = int(np.sum(self._particle_count[self._region_indices, frame_index + 1]))
        N = int(np.sum(self._particle_count[self._region_indices, frame_index]))

        Ds = np.zeros([N, M])
        drift_x = np.zeros([N, M])
        drift_y = np.zeros([N, M])
        N_ix = np.cumsum(self._particle_count[self._region_indices, frame_index]).astype(int)
        N_ix = np.insert(N_ix, 0, 0)  # insert a 0 at the start for the technical purpose of the for loop
        for j in range(0, len(self._region_indices)):
            Ds[N_ix[j]:N_ix[j + 1]] = ones(Ds[N_ix[j]:N_ix[j + 1]].shape) * self._working_diffusivities[
                self._region_indices[j]]
            drift_x[N_ix[j]:N_ix[j + 1]] = ones(drift_x[N_ix[j]:N_ix[j + 1]].shape) * self._working_drifts[
                self._region_indices[j], 0]
            drift_y[N_ix[j]:N_ix[j + 1]] = ones(drift_y[N_ix[j]:N_ix[j + 1]].shape) * self._working_drifts[
                self._region_indices[j], 1]
        return Ds, drift_x, drift_y

    # smoothing functions #
    def smoothing_prior(self):
        """
            Computes the smoothing prior over all cells to infer
            Note: This is done only for the diffusivity, not for drift.
            Drift is not regularized.
        :return: the log of the diffusion smoothing prior
        """
        # print("call: smoothing prior")
        penalization = 0
        index = self._cells_to_infer
        for i in index:
            gradD = self._cells.grad(i, self._working_diffusivities)
            penalization += self._cells.grad_sum(i, gradD * gradD)
        return self._smoothing_factor * penalization

    def smoothing_prior_heuristic(self):
        """
            This heuristic smoothing make use the dual graph of the tessellation.
            Every two adjacent cells are connected by an edge.
            To each of those edges we assign a value equal to the absolute value of the difference between the diffusivities in the two cells.
            This value is weighed by the inverse distance separating the two cell centers
            In the end, all weights are summed. This gives a measure of the amplitude of variations in the tessellation.
        :return: float. The penalization
        """
        penalization = 0
        index = self._cells_to_infer
        for i in index:
            sum_distances = sum(
                [np.linalg.norm(self._tessellation.cell_centers[i] - self._tessellation.cell_centers[j]) for j in
                 self.get_neighbours(i)])
            for j in self.get_neighbours(i):
                value = abs(self._working_diffusivities[i] - self._working_diffusivities[j])
                weight = np.linalg.norm(
                    self._tessellation.cell_centers[i] - self._tessellation.cell_centers[j]) / sum_distances
                penalization += value * weight

        return penalization

    def smoothed_posterior(self, D):
        """
            Computes the smoothed posterior for parameters D
        :param D: depends on whether we are 1D,2D, D,DD. Note: This argument passing is unreadably complicated. To be simplified
        :return: the smoothed posterior minus log-likelihood
        """
        D = abs(D)
        self.vprint(3, D, end_='')
        ''' Nelder-Mead does not have positivity constraints for D_in, so we might find a negative value.
            'smoothedPosterior' is implemented to be symmetric, so that we should still find the optimum.'''
        self._working_diffusivities = self._final_diffusivities  # the most up-to-date parameters
        self._working_drifts = self._final_drifts
        if self._correct_p_off is True:
            self._p_off = self.compute_corrected_p_off()
            self.vprint(3, f"pc={round(self._p_off, 3)}", end_="")
        # Note: Perhaps we should check for a value 0 of the diffusivity. It could cause problems later
        # scheme selector
        if self._inference_mode == 'D':
            if self._scheme == '1D':
                self._working_diffusivities[self._index_to_infer] = D
            elif self._scheme == '2D':
                self._working_diffusivities[self._region_indices] = D[1]
                self._working_diffusivities[self._index_to_infer] = D[
                    0]  # this overwrites the wrongly assigned value in previous line
            else:
                raise ValueError(f"scheme {self._scheme} not supported")
        elif self._inference_mode == 'DD':
            if self._scheme == '1D':
                self._working_diffusivities[self._index_to_infer] = D[0]
                self._working_drifts[self._index_to_infer, :] = D[1]
            elif self._scheme == '2D':
                self._working_diffusivities[self._region_indices] = D[1]
                self._working_diffusivities[self._index_to_infer] = D[0]
                self._working_drifts[self._region_indices] = D[4:6]
                self._working_drifts[self._index_to_infer] = D[2:4]
            else:
                raise ValueError(f"scheme {self._scheme} not supported in 'DD' mode")

        # method selector
        if self._method == 'MPA':
            mlnL = self.MPA_minusLogLikelihood_multiFrame()
        elif self._method == 'BP':
            mlnL = self.marginal_minusLogLikelihood_multiFrame()
        else:
            raise ValueError(f"Invalid method {self._method}. Choose either 'BP' or 'MPA'. ")

        # smoothing selector
        if self._smoothing_factor == 0:
            posterior = mlnL
        else:
            posterior = mlnL + self.smoothing_prior()
        self.vprint(3, f"{posterior}")
        if posterior is np.nan:
            raise FunctionEvaluation()

        self.optimizer_first_iteration = False
        # import pdb; pdb.set_trace()
        return posterior

    # Sum-product BP #
    def initialize_messages(self, N, M):
        """
            Initializes the BP messages to zeros.
            Attention: hji is in fact indexed in a i,j fashion, i.e h_{j\to i) is in row i, column j
        :param N: particles number first frame
        :param M: particles number second frame
        :return: Two matrices, of the same size N x M containing only zeros
        """
        hij = zeros([N, M])
        hji = zeros([N, M])
        return hij, hji

    def compute_damping_factor(self, F, F_minus_one, F_minus_two, score):
        """
            Computes the dynamic damping factor ass explained in report
        :param F: current energy
        :param F_minus_one: previous energy
        :param F_minus_two: "grandparent"-energy
        :param score: previous score
        :return: current score and subsequent damping factor
        """
        oscillation_indicator = ((F - F_minus_one) * (F_minus_one - F_minus_two) < 0)
        score = self._dynamic_damping_delta * (score + oscillation_indicator)
        gamma = self._dynamic_damping_gamma_min + \
                score * (1 - self._dynamic_damping_delta) * (1 - self._dynamic_damping_gamma_min)
        return score, gamma

    def extend_matrix_by_one_row_and_one_column(self, H):
        """
            Extends a matrix H in M(n x n) by one row and one column, thereby obtaining a matrix K in M(n+1 x n+1).
                 _   H   _           _    K    _
                | " " " " |         | " " " " . |
                | " " " " |   -->   | " " " " . |
                | " " " " |         | " " " " . |
                |_" " " "_|         | " " " " . |
                                    |_. . . . ._|
        :param H: The matrix to be extended
        :return: The extended matrix
        """
        assert (H.shape[0] == H.shape[1])  # H needs to be square
        n = H.shape[0]
        K = zeros((n + 1, n + 1))
        # Create the 4 blocks of K
        K[0:n, 0:n] = H
        K[0:n, n] = H[:, n - 1]
        K[n, 0:n] = H[n - 1, :]
        K[n, n] = H[n - 1, n - 1]
        return K

    def initialize_messages_in_a_smart_way(self, N, M, n_on, n_off, frame_index):
        """
            Initializes the messages either at a fixed starting value or -if required- at the last converging value
        :param N: Number of particles in the first frame
        :param M: number of particles in the second frame
        :param n_on: Number of appearing particles
        :param n_off: Number of disappearing particles
        :param frame_index: Index of the first frame of the current frame transition
        :return: The initialized messages hij, hji
        """
        if (self.optimizer_first_iteration is False) and (self._hij_init_takeLast is True):
            # In this case we just take the converged matrices from last call
            hij = self.get_hij(frame_index, n_on)
            hji = self.get_hji(frame_index, n_on)
        elif (self.optimizer_first_iteration is True) and (self._hij_init_takeLast is True):
            hij, hji = self.initialize_messages(n_on + N, n_off + M)
            if self.get_hij(frame_index, n_on - 1) is not None:
                try:
                    hij = self.extend_matrix_by_one_row_and_one_column(self.get_hij(frame_index, n_on - 1))
                    hji = self.extend_matrix_by_one_row_and_one_column(self.get_hji(frame_index, n_on - 1))
                except:
                    self.vprint(1, f"Warning: hij(n_on-1) re-use did not work. Setting a breakpoint.")
                    # import pdb; pdb.set_trace()
        else:
            hij, hji = self.initialize_messages(n_on + N, n_off + M)
        assert (hij.shape == hji.shape)
        return hij, hji

    def boolean_matrix(self, N):
        """
            Boolean matrix for parallel implementation of BP.
        :param N: Size of the matrix
        :return: A square matrix of size N x N with 0 in the diagonal and 1 everywhere else
        """
        bool_array = ones([N, N])
        for i in range(N):
            bool_array[i, i] = 0.
        return bool_array

    def bethe_free_energy(self, hij, hji, Q):
        """
            Bethe free energy given BP messages hij and hji and log-probabilities Q.
            This is the "standard" form of the energy
        :param hij: Left side messages
        :param hji: Right side messages
        :param Q: matrix of log-probabilities
        :return: Bethe free energy approximation
        """
        start = timeit.default_timer()
        # Naive version
        '''
        naive_energy = + sum(log(1. + exp(Q + hij + hji))) \
                       - sum(log(sum(exp(Q + hji), axis=1))) \
                       - sum(log(sum(exp(Q + hij), axis=0)))
        '''

        # logsumexp version
        logsumexp_energy = + sum(np.logaddexp(0, Q + self._temperature * hij + self._temperature * hji)) \
                           - sum(logsumexp(Q + self._temperature * hji, axis=1)) \
                           - sum(logsumexp(Q + self._temperature * hij, axis=0))
        # self.vprint(4,f"Bethe time \t\t= {(timeit.default_timer()-start)*1000} ms")
        return logsumexp_energy / self._temperature

    def sum_product_update_rule(self, Q, hij_old, hji_old):
        """
            The "standard" sum-product update rule
        :param Q: matrix of log-probabilities
        :param hij_old: Old left messages
        :param hji_old: Old right messages
        :return: The new messages (undamped)
        """
        start = timeit.default_timer()
        hij_new = -log(dot(exp(Q + self._temperature * hji_old), self.boolean_matrix(Q.shape[1]))) / self._temperature
        hji_new = -log(dot(self.boolean_matrix(Q.shape[0]), exp(Q + self._temperature * hij_old))) / self._temperature
        # self.vprint(4,f"sum-product time \t= {(timeit.default_timer()-start)*1000} ms")
        # logdotexp update
        # hij_new = -logdotexp(Q + self._temperature * hji_old, ... ) -  / self._temperature
        return hij_new, hji_new

    def sum_product_BP(self, Q, hij, hji):
        """
            Sum-product (BP) free energy (minus-log-likelihood) and BP messages.
        :param Q: The log-probability matrix
        :param hij: The left side messages
        :param hji: The right side messages
        :return: F_BP : The final Bethe free energy
                 hij  : The final left side message
                 hji  : The final right side message
                 n    : The number of iterations at convergence
        """
        hij_old = hij
        hji_old = hji
        F_BP_old = self.bethe_free_energy(hij_old, hji_old, Q)
        # Bethe = [F_BP_old]
        # Gamma = []
        # self.vprint(4,"")
        F_minus_one = 0
        F_minus_two = 0
        score = 0
        for n in range(self._maxiter):
            hij_new, hji_new = self.sum_product_update_rule(Q, hij, hji)
            if self._dynamic_damping is True:
                score, gamma = self.compute_damping_factor(F_BP_old, F_minus_one, F_minus_two, score)
            else:
                gamma = self._gamma
            hij = (1. - gamma) * hij_new + gamma * hij_old
            hji = (1. - gamma) * hji_new + gamma * hji_old
            # Stopping condition:
            F_BP = self.bethe_free_energy(hij, hji, Q)
            F_minus_two = F_minus_one
            F_minus_one = F_BP_old
            self.vprint(4, f"n={n} \t\t Bethe={F_BP} \t\t ||hij_old-hij-new||={np.linalg.norm(hij_old - hij_new)}\r",
                        end_="")
            if F_BP is np.nan or F_BP == np.inf:
                raise FunctionEvaluation(f"Bethe energy got value F_BP={F_BP}")
            try:
                if abs(F_BP - F_BP_old) < self._epsilon:
                    self.vprint(4, "")
                    break
            except RuntimeWarning:
                pass
                # import pdb; pdb.set_trace()
            # Update old values of energy and messages:
            F_BP_old = F_BP
            hij_old = hij
            hji_old = hji
            # Bethe.append(F_BP)
            # Gamma.append(gamma)
        # fig = plt.figure()
        # plt.plot(Bethe)
        # plt.show()
        # fig = plt.figure()
        # plt.plot(Gamma, color='r')
        # plt.show()
        return F_BP, hij, hji, n  # shall we store n somewhere and send it back to the user in some sort of diagnosis object ?

    def sum_product_energy(self, dr, frame_index, n_on, n_off, N, M):
        """
            Bethe free energy of given graph. The BP algorithm is run. F_BP is returned after convergence
        :param dr:
        :param frame_index:
        :param n_on: The number of appearing particles
        :param n_off: The number of disappearing particles
        :param N: The number of particles in the first frame (not including phantom particles)
        :param M: The number of particles in the second frame (not including phantom particles)
        :return: F_BP the Bethe free energy approximation
        """
        # If a single particle has been recorded in each frame and tracers are permanent, the link is known:
        if (N == 1) & (n_off == 0) & (n_on == 0) & (M == 1):
            F_BP = -self.minus_lq_minus_lnp_ij(dr, frame_index)
        # Else, perform BP to obtain the Bethe free energy:
        else:
            if (self.optimizer_first_iteration is False) and (self._hij_init_takeLast is True):
                hij = self.get_hij(frame_index, n_on)
                hji = self.get_hji(frame_index, n_on)
            else:
                hij, hji = self.initialize_messages(n_on + N, n_off + M)
            try:
                assert (hij.shape == hji.shape)
            except AttributeError:
                hij, hji = self.initialize_messages(n_on + N, n_off + M)
            Q = self.Q_ij(n_off, n_on, dr, frame_index)
            try:
                F_BP, hij, hji, n = self.sum_product_BP(Q, hij, hji)
            except RuntimeWarning:
                n = 0
                F_BP = np.nan
                raise FunctionEvaluation(f"Runtime Warning happened in BP. Stopping the BP")
            self.set_hij(hij, frame_index, n_on)
            self.set_hji(hji, frame_index, n_on)
            if n == self._maxiter - 1:
                self.vprint(1, f"Warning: BP attained maxiter before converging.")
                self.maxiter_attained_counter += 1
        # print(f"n={n}") # the number of iterations
        return F_BP

    def marginal_minusLogLikelihood_phantom(self, dr, frame_index):
        # self._stored_hij[frame_index] = [None] * (M - max([0, int(Delta)]))
        '''
        mlnL = -logsumexp([log(self.p_m(n_on, delta, N)) + lng(M + 1 - n_on) - lng(M + 1) - lng(N + 1) \
                           - self.sum_product_energy(dr, frame_index, n_on, n_on - delta, N, M) \
                           for n_on in range(max([0, int(delta)]), int(M))])  # Why use lng ?
        '''
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1]).astype(int)
        N = np.sum(self._particle_count[self._region_indices, frame_index]).astype(int)
        delta = M - N
        if self._cutoff_low_p_non is False:
            mlnL = -logsumexp([log(self.p_m(n_on, delta, N)) + lng(M + 1 - n_on) - lng(M + 1) - lng(N + 1) \
                               - self.sum_product_energy(dr, frame_index, n_on, n_on - delta, N, M) \
                               for n_on in range(max([0, int(delta)]), int(M))])
        else:
            self.vprint(4, f"M = {M} \t N = {N} \t delta = {delta}")
            noff_array = []
            for n_on in range(max([0, int(delta)]), int(M)):
                # import pdb; pdb.set_trace()
                p = self.lnp_m(n_on, delta, N)
                if p == np.inf or p == -np.inf or p == np.nan:
                    raise (f"Numerical Error, p={p}")
                self.vprint(4, f"n_on={n_on} \t p={p}")
                if p > self._cutoff_log_threshold and self._cutoff_low_p_non is True:
                    val = self.lnp_m(n_on, delta, N) + lng(M + 1 - n_on) - lng(M + 1) - lng(N + 1) \
                          - self.sum_product_energy(dr, frame_index, n_on, n_on - delta, N, M)
                    # self.vprint(4,f"lnL_non={n_on} = {val}")
                    noff_array.append(val)
            mlnL = -logsumexp(noff_array)
        return mlnL

    def marginal_minusLogLikelihood(self, dr, frame_index):
        """
            Minus-log-likelihood marginalized over possible graphs and assignments using BP.
            in is the inside cell, out is the outside cells (array)
        :param dr: the displacement vectors
        :param frame_index: the index of the current frame
        :return: the minus log-likelihood between frame frame_index and frame frame_index+1
        """
        # self.vprint(3, f"call: marginal_minusLogLikelihood, frame_index={frame_index}")
        self.vprint(3, ".", end_='')
        M_index_to_infer = self._particle_count[self._index_to_infer, frame_index + 1].astype(int)
        N_index_to_infer = self._particle_count[self._index_to_infer, frame_index].astype(int)
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1]).astype(int)
        N = np.sum(self._particle_count[self._region_indices, frame_index]).astype(int)
        delta = M - N
        if (M_index_to_infer == 0) or (N_index_to_infer == 0):
            mlnL = 0.  # if the cell to infer is empty in one frame, we gain 0 information
        elif ((self._p_off == 0.) & (self._mu_on == 0.) & (delta == 0)) or (self._phantom_particles is False):
            # Q = self.Q_ij(0, 0, dr, frame_index)
            mlnL = self.sum_product_energy(dr, frame_index, 0, 0, N, M)
            # self.vprint(3,f"mlnL={mlnL}")
        elif self._phantom_particles is True:
            mlnL = self.marginal_minusLogLikelihood_phantom(dr, frame_index)
        else:
            raise ValueError(f"Error: check value of phantom_particles")
        return mlnL

    def marginal_minusLogLikelihood_multiFrame(self):
        '''Minus-log-likelihood marginalized over possible graphs and assignments using BP.'''
        # self.vprint(3, f"call: marginal_minusLogLikelihood_multiFrame")
        self.vprint(2, "*", end_='')
        mlnL = 0
        for frame_index, dr in enumerate(self._drs):
            # print(f"frame_index={frame_index} \t dr={dr}")
            mlnL += self.marginal_minusLogLikelihood(dr, frame_index)
            # import pdb; pdb.set_trace()
        return mlnL

    # -----------------------------------------------------------------------------
    # Kuhn-Munkres MPA | ok
    # -----------------------------------------------------------------------------
    def MPA_energy(self, Q, mpa_row, mpa_col):
        return -sum(Q[mpa_row, mpa_col])

    def MPAscore(self, dr, frame_index):
        """
        :param dr: The distance matrix for frame_index
        :param frame_index: The current frame index
        :return: the MPAscore for frame_index
        """
        M_index_to_infer = self._particle_count[self._index_to_infer, frame_index + 1]
        N_index_to_infer = self._particle_count[self._index_to_infer, frame_index]
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1])
        N = np.sum(self._particle_count[self._region_indices, frame_index])
        Delta = M - N
        if (self._p_off == 0.) & (self._mu_on == 0.) & (Delta == 0):
            Q = self.Q_ij(0, 0, dr, frame_index)
            mpa_row, mpa_col = kuhn_munkres(-Q)
            max_score = -self.MPA_energy(Q, mpa_row, mpa_col)
        elif (M_index_to_infer == 0) or (N_index_to_infer == 0):
            max_score = 0.  # if the cell to infer is empty in one frame, we gain 0 information
        else:
            scores_ = []
            for n_on in arange(max([0, Delta]), M):
                Q = self.Q_ij(n_on - Delta, n_on, dr, frame_index)
                mpa_row, mpa_col = kuhn_munkres(-Q)
                logp_m = log(self.p_m(n_on, Delta, N))
                E_MPA = self.MPA_energy(Q, mpa_row, mpa_col)
                score = logp_m - E_MPA + lng(M + 1 - n_on) - lng(M + 1) - lng(N + 1)  # Note: why are there lng terms ?
                scores_.append(score)
            max_score = max(scores_)
        return -max_score

    def MPA_minusLogLikelihood_multiFrame(self):
        """
            Computes the minus log-likelihood for all the frames
        :return: a float, the minus log-likelihood
        """
        mlnL = 0
        for frame_index, dr in enumerate(self._drs):
            mlnL += self.MPAscore(dr, frame_index)

        return mlnL


# -----------------------------------------------------------------------------
# Child classes for different methods ? MPA, BP, BPchemPotChertkov, BPchemPotMezard, BP_JB, maybe others
# -----------------------------------------------------------------------------
class NonTrackingInferrerRegionBPchemPotMezard(NonTrackingInferrerRegion):
    """
        Implements the chemical potential solution with the Mezard formulation
    """

    def sum_product_update_rule(self, Q, hij_old, hji_old):
        hij_new = -log(dot(exp(Q + self._temperature * hji_old), self.boolean_matrix(Q.shape[1])) \
                       + exp(-self._chemPot_gamma)) / self._temperature
        hji_new = -log(dot(self.boolean_matrix(Q.shape[0]), exp(Q + self._temperature * hij_old)) \
                       + exp(-self._chemPot_gamma)) / self._temperature
        return hij_new, hji_new

    def bethe_free_energy(self, hij, hji, Q):
        """
            Computes the Bethe free energy for the Mezard formulation
        :param hij: left-side messages
        :param hji: right-side messages
        :param Q: log-likelihood matrix
        :return: The Bethe free energy
        """
        # print("call bethe_free_energy, Mezard")
        # import pdb; pdb.set_trace()
        # We use logsumexp to avoid numerical underflow or overflow problems. The two versions do return different results and the logsumexp version gives realistic results on instances where the naive versions gives abusive results

        # Naive version
        '''
        term1 = + sum(log(1. + exp(Q + hij + hji)))
        term2 = - sum(log(sum(exp(Q + hji), axis=1) + exp(-self._chemPot_gamma)))
        term3 = - sum(log(sum(exp(Q + hij), axis=0) + exp(-self._chemPot_gamma)))
        term4 = - sum(Q.shape) * self._chemPot_gamma
        naive_energy = term1 + term2 + term3 + term4
        '''

        # logsumexp version, no constant term
        # gamma_matrix_ = ones(Q.shape)*self._chemPot_gamma
        term1_bis = + sum(np.logaddexp(0, Q + self._temperature * hij + self._temperature * hji))
        term2_bis = - sum(
            logsumexp(np.hstack((Q + self._temperature * hji, -ones((Q.shape[0], 1)) * self._chemPot_gamma)), axis=1))
        term3_bis = - sum(
            logsumexp(np.vstack((Q + self._temperature * hij, -ones((1, Q.shape[1])) * self._chemPot_gamma)), axis=0))
        term4_bis = - sum(Q.shape) * self._chemPot_gamma
        logsumexp_energy = term1_bis + term2_bis + term3_bis + term4_bis

        return logsumexp_energy / self._temperature


class NonTrackingInferrerRegionBPchemPotChertkov(NonTrackingInferrerRegion):
    """
        Implements the chemical potential solution with the Chertkov formulation
    """

    def sum_product_update_rule(self, Q, hij_old, hji_old):
        hij_new = -log(dot(exp(Q + self._temperature * hji_old), self.boolean_matrix(Q.shape[1])) \
                       + exp(-self._chemPot_mu)) / self._temperature
        hji_new = -log(dot(self.boolean_matrix(Q.shape[0]), exp(Q + self._temperature * hij_old)) \
                       + exp(-self._chemPot_mu)) / self._temperature
        return hij_new, hji_new

    def bethe_free_energy(self, hij, hji, Q):
        """
            Computes the Bethe free energy for the Chertkov formulation
        :param hij: left-side messages
        :param hji: right-side messages
        :param Q: log-likelihood matrix
        :return: The Bethe free energy
        """
        # We use logsumexp to avoid numerical underflow or overflow problems. The two versions do return different results and the logsumexp version gives realistic results on instances where the naive versions gives abusive results
        # Naive version
        '''
        term1 = + sum(log(1. + exp(Q + hij + hji)))
        term2 = - sum(log(sum(exp(Q + hji), axis=1) + exp(-self._chemPot_mu)))
        term3 = - sum(log(sum(exp(Q + hij), axis=0) + exp(-self._chemPot_mu)))
        naive_energy = term1 + term2 + term3
        '''

        # logsumexp version, no constant term
        # gamma_matrix_ = ones(Q.shape)*self._chemPot_gamma
        term1_bis = + sum(np.logaddexp(0, Q + self._temperature * hij + self._temperature * hji))
        term2_bis = - sum(
            logsumexp(np.hstack((Q + self._temperature * hji, -ones((Q.shape[0], 1)) * self._chemPot_mu)), axis=1))
        term3_bis = - sum(
            logsumexp(np.vstack((Q + self._temperature * hij, -ones((1, Q.shape[1])) * self._chemPot_mu)), axis=0))
        logsumexp_energy = term1_bis + term2_bis + term3_bis

        return logsumexp_energy / self._temperature


class NonTrackingInferrerRegionBPphimu(NonTrackingInferrerRegion):
    """
        Implements the class of phimu methods with all possible combinations of phantom and chemical potential in the inner hex and outer rings
    """

    def p_m(self, x, Delta, N):
        '''Poissonian probability for x particles to appear between frames given Delta=M-N
        and blinking parameters mu_on and mu_off. Consult the article for reference'''
        mu_off = N * self._p_off
        if (self._ring_phimu == "phi" or self._hex_phimu == "phimu") and (
            self._hex_phimu == "phi" or self._hex_phimu == "phimu"):
            mu_on = self._region_mu_on
        elif (self._ring_phimu == "" or self._hex_phimu == "mu") and (
            self._hex_phimu == "phi" or self._hex_phimu == "phimu"):
            mu_on = self._index_to_infer_mu_on
        else:
            raise ValueError(f"Either phimu has a non admissible value, or the code shouldn't have come here")
        return (self._index_to_infer_mu_on * mu_off) ** (x - Delta / 2.) / (
            g(x + 1.) * g(x + 1. - Delta) * iv(Delta, 2 * sqrt(self._index_to_infer_mu_on * mu_off)))

    def lnp_m(self, x, Delta, N):
        """
            The logarithm of `p_m`
        :param x: The value of which we want to evaluate the probability
        :param Delta: The observed particle number difference
        :param N: The number of particles in the first frame
        :return:
        """
        mu_off = N * self._p_off
        if (self._ring_phimu == "phi" or self._hex_phimu == "phimu") and (
            self._hex_phimu == "phi" or self._hex_phimu == "phimu"):
            mu_on = self._region_mu_on
        elif (self._ring_phimu == "" or self._hex_phimu == "mu") and (
            self._hex_phimu == "phi" or self._hex_phimu == "phimu"):
            mu_on = self._index_to_infer_mu_on
        else:
            raise ValueError(f"Either phimu has a non admissible value, or the code shouldn't have come here")
        return log(self._index_to_infer_mu_on * mu_off) * (x - Delta / 2.) - lng(x + 1.) - lng(x + 1. - Delta) \
               - log(iv(Delta, 2 * sqrt(self._index_to_infer_mu_on * mu_off)))

    def Q_ij(self, n_off, n_on, dr, frame_index):
        """
        Matrix of log-probabilities for assignments with appearances and disappearances.
        :param n_off: The number of disappearing particles
        :param n_on:  The number of appearing particles
        :param dr:  The displacement vectors
        :param frame_index: The index of the current frame
        :return: The square matrix Q_ij of size N + n_on - n_off
        """
        n_on = int(n_on)
        n_off = int(n_off)
        M_index_to_infer = self._particle_count[self._index_to_infer, frame_index + 1].astype(int)
        N_index_to_infer = self._particle_count[self._index_to_infer, frame_index].astype(int)
        N = np.sum(self._particle_count[self._region_indices, frame_index]).astype(int)
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1]).astype(int)
        if self._phantom_particles is True:
            assert (M_index_to_infer == N_index_to_infer - n_off + n_on)
        out = zeros([n_on + N, n_off + M])  # initializing the returned matrix with the right size
        LM = ones([n_on + N, n_off + M]) * self._region_area
        lnp = self.minus_lq_minus_lnp_ij(dr, frame_index)

        # common part
        out[:N, :M] = lnp - log(LM[:N, :M])  # Q part (top left)
        out[N:, M:] = 2. * self._minlnL  # bottom right

        # dependent part
        if self._hex_phimu == "phi" or self._hex_phimu == "phimu":
            out[:N_index_to_infer, M:] = -log(LM[:N_index_to_infer, M:])  # top right
            out[N:, :M_index_to_infer] = -log(LM[N:, :M_index_to_infer])  # bottom left
        else:
            out[:N_index_to_infer, M:] = 2. * self._minlnL  # middle right
            out[N:, :M_index_to_infer] = 2. * self._minlnL  # bottom middle
        if self._ring_phimu == "phi" or self._ring_phimu == "phimu":
            out[N_index_to_infer:, M:] = -log(LM[N_index_to_infer:, M:])  # middle right
            out[N:, M_index_to_infer:] = -log(LM[N:, M_index_to_infer:])  # bottom middle
        else:
            out[N_index_to_infer:, M:] = 2. * self._minlnL  # middle right
            out[N:, M_index_to_infer:] = 2. * self._minlnL  # bottom middle
        return out

    def compute_exp_mu_i_and_mu_j(self, shape, N_in, M_in, N, M, n_on, n_off):
        """
            Computes the chemical potential matrices for i and j.
        :param shape: The shape of the exp_mu matrices
        :param N_in:
        :param M_in:
        :param N:
        :param M:
        :param n_on:
        :param n_off:
        :return:
        """
        assert (N_in + n_on == M_in + n_off)  # equilibrium condition
        out_i = zeros(shape) + exp(-self._chemPot_mu)
        out_j = zeros(shape) + exp(-self._chemPot_mu)
        if self._hex_phimu == "phi" or self._hex_phimu == "":
            out_i[:N_in, :] = 0
            out_j[:, :M_in] = 0
        if self._ring_phimu == "phi" or self._ring_phimu == "":
            out_i[N_in:N, :] = 0
            out_j[:, M_in:M] = 0
        out_i[N:(N + n_on), :] = 0
        out_j[:, M:(M + n_off)] = 0
        self.exp_mu_i = out_i
        self.exp_mu_j = out_j

    def marginal_minusLogLikelihood_phantom(self, dr, frame_index):
        """
            Computes the marginal minus log-likelihood, marginalizing over phantom particles
        :param dr: The distance matrix for frame_index
        :param frame_index: The current frame index
        :return:
        """
        M_index_to_infer = self._particle_count[self._index_to_infer, frame_index + 1].astype(int)
        N_index_to_infer = self._particle_count[self._index_to_infer, frame_index].astype(int)
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1]).astype(int)
        N = np.sum(self._particle_count[self._region_indices, frame_index]).astype(int)
        if (self._hex_phimu == "phi" or self._hex_phimu == "phimu") and (
            self._ring_phimu == "phi" or self._ring_phimu == "phimu"):
            delta = M - N
        elif (self._hex_phimu == "phi" or self._hex_phimu == "phimu") and not (
            self._ring_phimu == "phi" or self._ring_phimu == "phimu"):
            delta = M_index_to_infer - N_index_to_infer
        elif not (self._hex_phimu == "phi" or self._hex_phimu == "phimu") and (
            self._ring_phimu == "phi" or self._ring_phimu == "phimu"):
            delta = M - N - (M_index_to_infer - N_index_to_infer)
        elif not (self._hex_phimu == "phi" or self._hex_phimu == "phimu") and not (
            self._ring_phimu == "phi" or self._ring_phimu == "phimu"):
            raise ValueError(
                f"Either the phimu argument is not admissible or the code should never have reached this place")
        if self._cutoff_low_p_non is False:
            raise (f"Error: Need to set _exp_mu first")
            mlnL = -logsumexp([log(self.p_m(n_on, delta_index_to_infer, N_index_to_infer)) + lng(M + 1 - n_on) - lng(
                M + 1) - lng(N + 1) \
                               - self.sum_product_energy(dr, frame_index, n_on, n_on - delta_index_to_infer, N, M) \
                               for n_on in range(max([0, int(delta_index_to_infer)]), int(M_index_to_infer))])
        else:
            assert (self._cutoff_low_p_non is True)
            self.vprint(4, f"M = {M_index_to_infer} \t N = {N_index_to_infer} \t delta = {delta}")
            noff_array = []
            for n_on in range(max([0, int(delta)]), int(M_index_to_infer)):
                n_off = n_on - delta
                # import pdb; pdb.set_trace()
                p = self.lnp_m(n_on, delta, N_index_to_infer)
                if p == np.inf or p == -np.inf or p == np.nan:
                    raise (f"Numerical Error, p={p}")
                self.vprint(4, f"n_on={n_on} \t p={p}")
                if p > self._cutoff_log_threshold and self._cutoff_low_p_non is True:
                    self.compute_exp_mu_i_and_mu_j((N + n_on, M + n_off), N_index_to_infer, M_index_to_infer,
                                                   N, M, n_on, n_off)
                    val = self.lnp_m(n_on, delta, N_index_to_infer) + lng(M + 1 - n_on) - lng(M + 1) - lng(N + 1) \
                          - self.sum_product_energy(dr, frame_index, n_on, n_on - delta, N, M)
                    # self.vprint(4,f"lnL_non={n_on} = {val}")
                    noff_array.append(val)
                else:
                    pass
            try:
                mlnL = -logsumexp(np.array(noff_array))
            except ValueError:
                pass
                # import pdb; pdb.set_trace()
        return mlnL

    def marginal_minusLogLikelihood(self, dr, frame_index):
        """
            Minus-log-likelihood marginalized over possible graphs and assignments using BP.
            in is the inside cell, out is the outside cells (array)
        :param dr: the displacement vectors
        :param frame_index: the index of the current frame
        :return: the minus log-likelihood between frame frame_index and frame frame_index+1
        """
        # self.vprint(3, f"call: marginal_minusLogLikelihood, frame_index={frame_index}")
        self.vprint(3, ".", end_='')
        M_index_to_infer = self._particle_count[self._index_to_infer, frame_index + 1].astype(int)
        N_index_to_infer = self._particle_count[self._index_to_infer, frame_index].astype(int)
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1]).astype(int)
        N = np.sum(self._particle_count[self._region_indices, frame_index]).astype(int)
        delta = M - N
        if (M_index_to_infer == 0) or (N_index_to_infer == 0):
            mlnL = 0.  # if the cell to infer is empty in one frame, we gain 0 information
            '''
        elif M_index_to_infer == 1 and N_index_to_infer == 1 and (M == 1 or N == 1):
            # The two extra limit cases in the PhanPot method
            mlnL = self.Q_ij(0, 0, dr, frame_index)[0,0] \
                   + (M-1)*exp(-self._chemPot_mu) \
                   + (N-1)*exp(-self._chemPot_mu)
            '''
        elif ((self._p_off == 0.) & (self._mu_on == 0.) & (delta == 0)) or (self._phantom_particles is False):
            # Q = self.Q_ij(0, 0, dr, frame_index)
            mlnL = self.sum_product_energy(dr, frame_index, 0, 0, N, M)
            # self.vprint(3,f"mlnL={mlnL}")
        elif self._phantom_particles is True:
            mlnL = self.marginal_minusLogLikelihood_phantom(dr, frame_index)
        else:
            raise ValueError(f"Error: check value of phantom_particles")
        return mlnL

    def sum_product_update_rule(self, Q, hij_old, hji_old):
        # try:
        hij_new = -log(dot(exp(Q + self._temperature * hji_old), self.boolean_matrix(Q.shape[1])) \
                       + self.exp_mu_i) / self._temperature
        hji_new = -log(dot(self.boolean_matrix(Q.shape[0]), exp(Q + self._temperature * hij_old)) \
                       + self.exp_mu_j) / self._temperature
        # except RuntimeWarning:
        # pass
        # import pdb; pdb.set_trace()
        return hij_new, hji_new

    def bethe_free_energy(self, hij, hji, Q):
        """
            Computes the Bethe free energy for the Chertkov formulation
        :param hij: left-side messages
        :param hji: right-side messages
        :param Q: log-likelihood matrix
        :return: The Bethe free energy
        """
        # We use logsumexp to avoid numerical underflow or overflow problems. The two versions do return different results and the logsumexp version gives realistic results on instances where the naive versions gives abusive results
        assert (self._temperature == 1)

        # Naive version
        term1 = + sum(log(1. + exp(Q + hij + hji)))
        term2 = - sum(log(self.exp_mu_i[:, 0] + sum(exp(Q + hji), axis=1)))
        term3 = - sum(log(self.exp_mu_j[0, :] + sum(exp(Q + hij), axis=0)))
        naive_energy = term1 + term2 + term3

        # logsumexp version, no constant term
        # logsumexp can't work here, because exp_mu may contain zeros
        # Possibility: Manually adapt the logsumexp algorithm
        '''
        term1_bis = + sum(np.logaddexp(0, Q + hij + hji))
        term2_bis = - sum(np.logaddexp(log(self.exp_mu_i[:,0]), logsumexp(Q + hji, axis=1) ))
        term3_bis = - sum(np.logaddexp(log(self.exp_mu_j[0,:]), logsumexp(Q + hji, axis=0) ))
        logsumexp_energy = term1_bis + term2_bis + term3_bis
        '''

        return naive_energy


class NonTrackingInferrerRegionBPchemPotPartialChertkov(NonTrackingInferrerRegionBPchemPotChertkov):
    """
        At the moment this is written, I believe this corresponds to one of the phimu methods
    """

    def p_m(self, x, Delta, N):
        '''Poissonian probability for x particles to appear between frames given Delta=M-N
        and blinking parameters mu_on and mu_off. Consult the article for reference'''
        mu_off = N * self._p_off
        return (self._index_to_infer_mu_on * mu_off) ** (x - Delta / 2.) / (
            g(x + 1.) * g(x + 1. - Delta) * iv(Delta, 2 * sqrt(self._index_to_infer_mu_on * mu_off)))

    def lnp_m(self, x, Delta, N):
        """
            The logarithm of `p_m`
        :param x: The value of which we want to evaluate the probability
        :param Delta: The observed particle number difference
        :param N: The number of particles in the first frame
        :return:
        """
        mu_off = N * self._p_off
        return log(self._index_to_infer_mu_on * mu_off) * (x - Delta / 2.) - lng(x + 1.) - lng(x + 1. - Delta) \
               - log(iv(Delta, 2 * sqrt(self._index_to_infer_mu_on * mu_off)))

    def Q_ij(self, n_off, n_on, dr, frame_index):
        """
        Matrix of log-probabilities for assignments with appearances and disappearances.
        Structure of Q_ij:
                 _Din    D1  ..  D6       _
            Din | Q_21  Q_22 .. Q_2M | 1/S |
             :  |  :     :       :   |  :  |
             D6 | Q_N1  Q_N2 .. Q_NM |  0  |
                |--------------------------|
                | 1/S    0   ..  0   |  0  |
                |_1/S    0   ..  0   |  0 _|
        :param n_off: The number of disappearing particles
        :param n_on:  The number of appearing particles
        :param dr:  The displacement vectors
        :param frame_index: The index of the current frame
        :return: The square matrix Q_ij of size N + n_on - n_off
        """
        n_on = int(n_on)
        n_off = int(n_off)
        M_index_to_infer = self._particle_count[self._index_to_infer, frame_index + 1].astype(int)
        N_index_to_infer = self._particle_count[self._index_to_infer, frame_index].astype(int)
        N = np.sum(self._particle_count[self._region_indices, frame_index]).astype(int)
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1]).astype(int)
        if self._phantom_particles is True:
            assert (M_index_to_infer == N_index_to_infer - n_off + n_on)
        out = zeros([n_on + N, n_off + M])  # initializing the returned matrix with the right size
        LM = ones([n_on + N, n_off + M]) * self._region_area
        lnp = self.minus_lq_minus_lnp_ij(dr, frame_index)
        out[:N, :M] = lnp - log(LM[:N, :M])  # Q part (top left)
        out[:N_index_to_infer, M:] = -log(LM[:N_index_to_infer, M:])  # top right
        out[N:, :M_index_to_infer] = -log(LM[N:, :M_index_to_infer])  # bottom left
        out[N_index_to_infer:, M:] = 2. * self._minlnL  # middle right
        out[N:, M_index_to_infer:] = 2. * self._minlnL  # bottom middle
        out[N:, M:] = 2. * self._minlnL  # bottom right
        return out

    def compute_exp_mu_i_and_mu_j(self, shape, N_in, M_in, N, M, n_on, n_off):
        """
            Computes the chemical potential matrices for i and j.
                 _                         _
                |   0     0  ..   0  |  0   |
            Cin |   :     :       :  |  :   |
                |   0     0  ..   0  |  0   |
                | e^-µ  e^-µ .. e^-µ | e^-µ |
            Cout|  :    :       :    |  :   |
                | e^-µ  e^-µ .. e^-µ | e^-µ |
                |---------------------------|
             Ph |  0     0   ..  0   |  0   |
                |_ 0     0   ..  0   |  0  _|
        :param shape:
        :param N_in:
        :param M_in:
        :param N:
        :param M:
        :param n_on:
        :param n_off:
        :return:
        """
        assert (N_in + n_on == M_in + n_off)  # equilibrium condition
        out = zeros(shape) + exp(-self._chemPot_mu)
        out[:N_in, :] = 0
        out[N:(N + n_on), :] = 0
        self.exp_mu_i = out

        out = zeros(shape) + exp(-self._chemPot_mu)
        out[:, :M_in] = 0
        out[:, M:(M + n_off)] = 0
        self.exp_mu_j = out

    def marginal_minusLogLikelihood_phantom(self, dr, frame_index):
        """
            Computes the marginal minus log-likelihood, marginalizing over phantom particles
        :param dr: The distance matrix for frame_index
        :param frame_index: The current frame index
        :return:
        """
        M_index_to_infer = self._particle_count[self._index_to_infer, frame_index + 1].astype(int)
        N_index_to_infer = self._particle_count[self._index_to_infer, frame_index].astype(int)
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1]).astype(int)
        N = np.sum(self._particle_count[self._region_indices, frame_index]).astype(int)
        delta_index_to_infer = M_index_to_infer - N_index_to_infer
        if self._cutoff_low_p_non is False:
            raise (f"Error: Need to set _exp_mu first")
            mlnL = -logsumexp([log(self.p_m(n_on, delta_index_to_infer, N_index_to_infer)) + lng(M + 1 - n_on) - lng(
                M + 1) - lng(N + 1) \
                               - self.sum_product_energy(dr, frame_index, n_on, n_on - delta_index_to_infer, N, M) \
                               for n_on in range(max([0, int(delta_index_to_infer)]), int(M_index_to_infer))])
        else:
            assert (self._cutoff_low_p_non is True)
            self.vprint(4, f"M = {M_index_to_infer} \t N = {N_index_to_infer} \t delta = {delta_index_to_infer}")
            noff_array = []
            for n_on in range(max([0, int(delta_index_to_infer)]), int(M_index_to_infer)):
                n_off = n_on - delta_index_to_infer
                # import pdb; pdb.set_trace()
                p = self.lnp_m(n_on, delta_index_to_infer, N_index_to_infer)
                if p == np.inf or p == -np.inf or p == np.nan:
                    raise (f"Numerical Error, p={p}")
                self.vprint(4, f"n_on={n_on} \t p={p}")
                if p > self._cutoff_log_threshold and self._cutoff_low_p_non is True:
                    self.compute_exp_mu_i_and_mu_j((N + n_on, M + n_off), N_index_to_infer, M_index_to_infer,
                                                   N, M, n_on, n_off)
                    val = self.lnp_m(n_on, delta_index_to_infer, N_index_to_infer) + lng(M + 1 - n_on) - lng(
                        M + 1) - lng(N + 1) \
                          - self.sum_product_energy(dr, frame_index, n_on, n_on - delta_index_to_infer, N, M)
                    # self.vprint(4,f"lnL_non={n_on} = {val}")
                    noff_array.append(val)
                else:
                    pass
            try:
                mlnL = -logsumexp(np.array(noff_array))
            except ValueError:
                pass
                # import pdb; pdb.set_trace()
        return mlnL

    def marginal_minusLogLikelihood(self, dr, frame_index):
        """
            Minus-log-likelihood marginalized over possible graphs and assignments using BP.
            in is the inside cell, out is the outside cells (array)
        :param dr: the displacement vectors
        :param frame_index: the index of the current frame
        :return: the minus log-likelihood between frame frame_index and frame frame_index+1
        """
        # self.vprint(3, f"call: marginal_minusLogLikelihood, frame_index={frame_index}")
        self.vprint(3, ".", end_='')
        M_index_to_infer = self._particle_count[self._index_to_infer, frame_index + 1].astype(int)
        N_index_to_infer = self._particle_count[self._index_to_infer, frame_index].astype(int)
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1]).astype(int)
        N = np.sum(self._particle_count[self._region_indices, frame_index]).astype(int)
        delta = M - N
        if (M_index_to_infer == 0) or (N_index_to_infer == 0):
            mlnL = 0.  # if the cell to infer is empty in one frame, we gain 0 information
        elif M_index_to_infer == 1 and N_index_to_infer == 1 and (M == 1 or N == 1):
            # The two extra limit cases in the PhanPot method
            mlnL = self.Q_ij(0, 0, dr, frame_index)[0, 0] \
                   + (M - 1) * exp(-self._chemPot_mu) \
                   + (N - 1) * exp(-self._chemPot_mu)
        elif ((self._p_off == 0.) & (self._mu_on == 0.) & (delta == 0)) or (self._phantom_particles is False):
            # Q = self.Q_ij(0, 0, dr, frame_index)
            mlnL = self.sum_product_energy(dr, frame_index, 0, 0, N, M)
            # self.vprint(3,f"mlnL={mlnL}")
        elif self._phantom_particles is True:
            mlnL = self.marginal_minusLogLikelihood_phantom(dr, frame_index)
        else:
            raise ValueError(f"Error: check value of phantom_particles")
        return mlnL

    def sum_product_update_rule(self, Q, hij_old, hji_old):
        # try:
        hij_new = -log(dot(exp(Q + self._temperature * hji_old), self.boolean_matrix(Q.shape[1])) \
                       + self.exp_mu_i) / self._temperature
        hji_new = -log(dot(self.boolean_matrix(Q.shape[0]), exp(Q + self._temperature * hij_old)) \
                       + self.exp_mu_j) / self._temperature
        # except RuntimeWarning:
        # pass
        # import pdb; pdb.set_trace()
        return hij_new, hji_new

    def bethe_free_energy(self, hij, hji, Q):
        """
            Computes the Bethe free energy for the Chertkov formulation
        :param hij: left-side messages
        :param hji: right-side messages
        :param Q: log-likelihood matrix
        :return: The Bethe free energy
        """
        # We use logsumexp to avoid numerical underflow or overflow problems. The two versions do return different results and the logsumexp version gives realistic results on instances where the naive versions gives abusive results
        assert (self._temperature == 1)

        # Naive version
        term1 = + sum(log(1. + exp(Q + hij + hji)))
        term2 = - sum(log(self.exp_mu_i[:, 0] + sum(exp(Q + hji), axis=1)))
        term3 = - sum(log(self.exp_mu_j[0, :] + sum(exp(Q + hij), axis=0)))
        naive_energy = term1 + term2 + term3

        # logsumexp version, no constant term
        # logsumexp can't work here, because exp_mu may contain zeros
        # Possibility: Manually adapt the logsumexp algorithm
        '''
        term1_bis = + sum(np.logaddexp(0, Q + hij + hji))
        term2_bis = - sum(np.logaddexp(log(self.exp_mu_i[:,0]), logsumexp(Q + hji, axis=1) ))
        term3_bis = - sum(np.logaddexp(log(self.exp_mu_j[0,:]), logsumexp(Q + hji, axis=0) ))
        logsumexp_energy = term1_bis + term2_bis + term3_bis
        '''

        return naive_energy


class NonTrackingInferrerRegionBPalternativeUpdateJB(NonTrackingInferrerRegion):

    def initialize_messages(self, N, M):
        """
            Initializes the BP messages to ones
        :param N: particles number first frame
        :param M: particles number second frame
        :return: Two matrices, of the same size N x M containing only ones
        """
        hij = ones([N, M])
        hji = ones([N, M])
        return hij, hji

    def sum_product_update_rule(self, Q, h_ij, alpha_i, alpha_j):
        """
            Perform one sum-product update in the space of hij-messages.
        :param Q: The matrix of log-probabilities log(pij)
        :param h_ij:
        :param alpha_i:
        :param alpha_j:
        :return:
        """
        Xi = np.square(dot(h_ij, np.ones(h_ij.shape) / 2) + dot(np.ones(h_ij.shape) / 2, h_ij) - h_ij)
        hij_new = exp(Q) * dot(exp(alpha_i), exp(alpha_j)) / (exp(Q) * dot(exp(alpha_i), exp(alpha_j)) + Xi)

        hij_new = self.sinkhornKnoppIteration(hij_new, rows=False)
        Ai_new = (1 - np.sum(hij_new ** 2, axis=1)) / dot(exp(Q), exp(alpha_j))

        hij_new = self.sinkhornKnoppIteration(hij_new, columns=False)
        Aj_new = (1 - np.sum(hij_new ** 2, axis=0)) / dot(exp(alpha_i), exp(Q))
        return hij_new, log(Ai_new), log(Aj_new)

    def free_energy(self, hij, Q, alpha_i, alpha_j):
        """
            Computes the free energy with Lagrangian relaxation of the normalization condition
        :param hij: the message matrix (equal to the belief that s_ij = 1)
        :param Q: The matrix of log-probabilities log(pij)
        :param alpha_i: Lagrangian multiplier for normalized columns
        :param alpha_j: Lagrangian multiplier for normalized rows
        :return:
        """
        return + sum(hij * (log(hij) - Q) - (1 - hij) * log(1 - hij)) \
               - sum(alpha_i * (sum(hij, axis=1) - 1)) \
               - sum(alpha_j * (sum(hij, axis=0) - 1))

    def sinkhornKnoppIteration(self, hij, rows=True, columns=True):
        """
            Performs one iteration of the Sinkhorn-Knopp algorithm: Sequentially normalizing rows and columns.
            The hope is that hij becomes 'more bistochastic' after this iteration
        :param hij: The message matrix
        :return: The transformed messages. Same shape as hij.
        """

        # Normalize the columns
        if columns:
            norm = np.sum(hij, axis=1)
            hij = hij / np.tile(norm, (hij.shape[1], 1)).T

        # Normalize the rows
        if rows:
            norm = np.sum(hij, axis=0)
            hij = hij / np.tile(norm, (hij.shape[0], 1))

        return hij

    def sum_product_BP(self, Q, hij, hji):
        """
            Sum-product (BP) free energy (minus-log-likelihood) and BP messages.
        :param Q: The matrix of log-probabilities log(pij)
        :param hij: The left side messages
        :param hji: The right side messages
        :return: F_BP : The final free energy
                 hij  : The final left side message
                 hji  : The final right side message (only kept for compatibility with mother code)
                 n    : The number of iterations at convergence
        """
        hij_old = hij
        alpha_i = np.ones(hij.shape[0])
        alpha_j = np.ones(hij.shape[1])
        F_BP_old = self.free_energy(hij_old, Q, alpha_i, alpha_j)
        for n in range(self._maxiter):
            hij_new, alpha_i, alpha_j = self.sum_product_update_rule(Q, hij, alpha_i, alpha_j)
            hij = (1. - self._gamma) * hij_new + self._gamma * hij_old
            hij = self.sinkhorn_knopp_iteration(hij)
            # Stopping condition:
            F_BP = self.free_energy(hij, Q, alpha_i, alpha_j)
            if abs(F_BP - F_BP_old) < self._epsilon:
                break
            # Update old values of energy and messages:
            F_BP_old = F_BP
            hij_old = hij
        return F_BP, hij, hji, n


class NonTrackingInferrerRegionGlobal(NonTrackingInferrerRegion):
    """
        Global method
    """

    def __init__(self, parentAttributes, index_to_infer, region_indices):
        super(NonTrackingInferrerRegion, self).__init__(
            *parentAttributes)  # need to get all attribute values from parent class
        self._index_to_infer = index_to_infer

        self._local_indices = region_indices
        self._region_indices = self.create_global_indices()
        # assert (np.all(self._local_indices == self._region_indices[range(len(self._local_indices))])) # The region's indices must be at the first place # fixme: Assertion error because indices are not in the right order
        assert (self._index_to_infer == self._local_indices[0])

        # self._diffusivityMatrix = self.buildDiffusivityMatrix() # maybe not needed yet
        # print(f"self._gamma : {self._gamma}")
        self._drs = self.local_distance_matrices(self._region_indices)

        area = 0
        for i in self._local_indices:
            cell = self._cells[i]
            area += cell.volume
        self._local_area = area
        area = 0
        for i in self._region_indices:
            cell = self._cells[i]
            area += cell.volume
        self._region_area = area

        self._particle_count_local = self.particle_count_local()
        self._particle_count = self.particle_count()

        self._hij = [None] * self._particle_count.shape[1]
        self._hji = [None] * self._particle_count.shape[1]
        for f in arange(len(self._hij) - 1):
            M = np.sum(self._particle_count[self._region_indices, f + 1]).astype(int)
            self._hij[f] = [None] * (M + 1)
            self._hji[f] = [None] * (M + 1)

        self.optimizer_first_iteration = True
        self.global_update_rule_bis = False
        self.maxiter_attained_counter = 0

    def create_global_indices(self):
        """
            Creates the array of all indices in a specific order.
            First the cell to infer, then the regional indices and lastly the rest.
        :return: the array of indices
        """
        # global_indices = np.insert(region_indices, 0, index_to_infer)  # insert i in the 0th position
        global_indices = self.get_neighbours(self._index_to_infer, order='all')
        global_indices = np.insert(global_indices, 0, self._index_to_infer)

        global_indices = list(dict.fromkeys(list(global_indices)))
        for j in self._local_indices:
            global_indices.remove(j)
        global_indices = list(map(int, global_indices))
        global_indices = array(global_indices)

        for i in self._local_indices:
            global_indices = np.insert(global_indices, 0,
                                       i)  # insert i in the 0th position to ensure local indices come first

        global_indices = list(dict.fromkeys(list(global_indices)))
        global_indices.remove(self._index_to_infer)
        global_indices = list(map(int, global_indices))
        global_indices = array(global_indices)

        global_indices = np.insert(global_indices, 0,
                                   self._index_to_infer)  # insert self._index_to_infer in the 0th position
        return global_indices

    def particle_count_local(self):
        """
            Used for __init__ of nonTrackingInferrerRegion
        :return: The matrix of particle numbers in the cells. First w.r.t cells, then w.r.t times
        """

        # Step 1: Compute the times
        r_total = []
        t_total = []
        S_total = 0
        index = list(self._cells.keys())
        for j in self._local_indices:
            cell = self._cells[j]
            r = cell.r
            t = cell.t
            S = cell.volume
            r_total.extend(r)
            t_total.extend(t)
            S_total += S
        r_total = np.array(r_total)
        t_total = np.array(t_total)
        # List of times corresponding to frames:
        times = np.arange(min(t_total), max(t_total + self._dt / 100.), self._dt)

        # Step 2: Compute the particle number
        particle_number = np.zeros((len(self._local_indices), len(times)))
        print(f"global_indices={index}")
        for j in range(len(self._local_indices)):
            cell = self._cells[j]
            r = cell.r
            t = cell.t
            frames_j = self.rt_to_frames(r, t, times)
            N_j = [len(frame) for frame in frames_j]
            particle_number[j, :] = N_j

        return particle_number

    def sum_product_BP(self, Q, hij, hji, frame_index):
        """
            Sum-product (BP) free energy (minus-log-likelihood) and BP messages.
        :param Q: The log-probability matrix
        :param hij: The left side messages
        :param hji: The right side messages
        :return: F_BP : The final Bethe free energy
                 hij  : The final left side message
                 hji  : The final right side message
                 n    : The number of iterations at convergence
        """
        hij_old = hij
        hji_old = hji
        F_BP_old = self.bethe_free_energy(hij_old, hji_old, Q)
        # Bethe = [F_BP_old]
        # self.vprint(4,"")
        # assert(self._temperature==1) # not putting a temperature in the implementation leads to a speed-up of around 25%
        boolean_matrix = self.boolean_matrix(Q.shape[0])
        for n in range(self._maxiter):
            force_global = False
            if self._sparse_energy_computation is True:
                for j in range(self._sparse_energy_computation_sparsity):
                    start = timeit.default_timer()
                    hij_new, hji_new = self.sum_product_update_rule_chooser(Q, hij, hji, frame_index, force_global)
                    hij = (1. - self._gamma) * hij_new + self._gamma * hij_old
                    hji = (1. - self._gamma) * hji_new + self._gamma * hji_old
                    hij_old = hij
                    hji_old = hji
                    # self.vprint(4, f"sum-product time \t= {(timeit.default_timer() - start) * 1000} ms")
            else:
                hij_new, hji_new = self.sum_product_update_rule_chooser(Q, hij, hji, frame_index, force_global)
                hij = (1. - self._gamma) * hij_new + self._gamma * hij_old
                hji = (1. - self._gamma) * hji_new + self._gamma * hji_old
            # Stopping condition Bethe free energy:
            start = timeit.default_timer()
            F_BP = + sum(np.logaddexp(0, Q + hij + hji)) \
                   - sum(logsumexp(Q + hji, axis=1)) \
                   - sum(logsumexp(Q + hij, axis=0))
            # self.vprint(4, f"Bethe time \t\t= {(timeit.default_timer() - start) * 1000} ms")
            self.vprint(4, f"n={n} \t\t Bethe={F_BP} \t\t ||hij_old-hij||={np.linalg.norm(hij_old - hij)}\r", end_="")
            if abs(F_BP - F_BP_old) < self._epsilon:
                self.vprint(4, "")
                break
            if F_BP is np.nan:
                raise FunctionEvaluation(f"Bethe energy got value nan")
            # Update old values of energy and messages:
            F_BP_old = F_BP
            hij_old = hij
            hji_old = hji
            # Bethe.append(F_BP)
        # fig = plt.figure()
        # plt.plot(Bethe)
        # plt.show()
        return F_BP, hij, hji, n  # shall we store n somewhere and send it back to the user in some sort of diagnosis object ?

    def sum_product_energy(self, dr, frame_index, n_on, n_off, N, M):
        """
            Bethe free energy of given graph. The BP algorithm is run. F_BP is returned after convergence
        :param dr:
        :param frame_index:
        :param n_on: The number of appearing particles
        :param n_off: The number of disappearing particles
        :param N: The number of particles in the first frame (not including phantom particles)
        :param M: The number of particles in the second frame (not including phantom particles)
        :return: F_BP the Bethe free energy approximation
        """
        # If a single particle has been recorded in each frame and tracers are permanent, the link is known:
        if (N == 1) & (n_off == 0) & (n_on == 0) & (M == 1):
            F_BP = -self.minus_lq_minus_lnp_ij(dr, frame_index)
        # Else, perform BP to obtain the Bethe free energy:
        else:
            hij, hji = self.initialize_messages_in_a_smart_way(N, M, n_on, n_off, frame_index)
            Q = self.Q_ij(n_off, n_on, dr, frame_index)
            F_BP, hij, hji, n = self.sum_product_BP(Q, hij, hji, frame_index)
            self.set_hij(hij, frame_index, n_on)
            self.set_hji(hji, frame_index, n_on)
            if n == self._maxiter - 1:
                self.vprint(1, f"Warning: BP attained maxiter before converging.")
        # print(f"n={n}") # the number of iterations
        return F_BP

    def sum_product_update_rule_chooser(self, Q, hij_old, hji_old, frame_index, force_global=False):
        if self.optimizer_first_iteration is True or force_global is True:
            return self.sum_product_update_rule(Q, hij_old, hji_old)
        elif self.optimizer_first_iteration is False and self.global_update_rule_bis is False:
            return self.sum_product_update_rule_local(Q, hij_old, hji_old, frame_index)
        elif self.optimizer_first_iteration is False and self.global_update_rule_bis is True:
            return self.sum_product_update_rule_local_bis(Q, hij_old, hji_old, frame_index)
        else:
            raise ValueError(f"self.optimizer_first_iteration is neither True nor False")

    def sum_product_update_rule_local(self, Q, hij_old, hji_old, frame_index):
        """
            We use a block-matrix decomposition, with A = exp(Q+Hij) and B = boolean matrix
                _(M)    _         _(M)    _        _(M)    _
            (N)| A11 A12 | x  (M)| B11 B12 | = (N)| C11 C12 |
               |_A21 A22_|       |_B21 B22_|      |_C21 C22_|

            C11 = A11 x B11 + A12 x B21

            The "local" sum-product update rule
        :param Q: matrix of log-probabilities
        :param hij_old: Old left messages
        :param hji_old: Old right messages
        :return: The new messages (undamped)
        """

        N_local = sum(self._particle_count[self._local_indices, frame_index]).astype(int)
        M_local = sum(self._particle_count[self._local_indices, frame_index + 1]).astype(int)
        # TODO: 1) O(n) implementation by diluting long edges
        #       2) --> **Implementation keeping faraway messages constant**
        start = timeit.default_timer()
        # Hij
        A = exp(Q + self._temperature * hji_old)
        A11 = A[0:N_local, 0:M_local]
        A12 = A[0:N_local, M_local + 1:]
        B = self.boolean_matrix(Q.shape[1])
        B21 = B[M_local + 1:, 0:M_local]
        hij_new = hij_old
        hij_new[0:N_local, 0:M_local] = -log(dot(A11, self.boolean_matrix(M_local)) + dot(A12, B21))

        # Hji
        A = exp(Q + self._temperature * hij_old)
        A11 = A[0:N_local, 0:M_local]
        A12 = A[0:N_local, M_local + 1:]
        B = self.boolean_matrix(Q.shape[1])
        B11 = B[0:M_local, 0:M_local]
        B21 = B[M_local + 1:, 0:M_local]
        hji_new = hji_old
        hji_new[0:N_local, 0:M_local] = -log(dot(A11, B11) + dot(A12, B21))

        # hij_new = -log(dot(exp(Q + self._temperature*hji_old), self.boolean_matrix(Q.shape[1])))/self._temperature
        # hji_new = -log(dot(self.boolean_matrix(Q.shape[0]), exp(Q + self._temperature*hij_old)))/self._temperature
        self.vprint(4, f"sum-product time \t= {(timeit.default_timer() - start) * 1000} ms")
        # logdotexp update
        # hij_new = -logdotexp(Q + self._temperature * hji_old, ... ) -  / self._temperature
        return hij_new, hji_new

    def sum_product_update_rule_local_bis(self, Q, hij_old, hji_old, frame_index):
        """
            We use a block-matrix decomposition, with A = exp(Q+Hij) and B = boolean matrix
                _(M)    _         _(M)    _        _(M)    _
            (N)| A11 A12 | x  (M)| B11 B12 | = (N)| C11 C12 |
               |_A21 A22_|       |_B21 B22_|      |_C21 C22_|

            We update C11, C12, C21, but not C22

            The "local" sum-product update rule
        :param Q: matrix of log-probabilities
        :param hij_old: Old left messages
        :param hji_old: Old right messages
        :return: The new messages (undamped)
        """

        N_local = sum(self._particle_count[self._local_indices, frame_index]).astype(int)
        start = timeit.default_timer()
        # Hij
        A1 = exp(Q + hji_old)[0:N_local, :]
        hij_new = hij_old
        hij_new[0:N_local, :] = -log(dot(A1, self.boolean_matrix(Q.shape[1])))

        # Hji
        A1 = exp(Q + hij_old)[0:N_local, :]
        hji_new = hji_old
        hji_new[0:N_local, :] = -log(dot(A1, self.boolean_matrix(Q.shape[1])))

        # hij_new = -log(dot(exp(Q + self._temperature*hji_old), self.boolean_matrix(Q.shape[1])))/self._temperature
        # hji_new = -log(dot(self.boolean_matrix(Q.shape[0]), exp(Q + self._temperature*hij_old)))/self._temperature
        self.vprint(4, f"sum-product time \t= {(timeit.default_timer() - start) * 1000} ms")
        # logdotexp update
        # hij_new = -logdotexp(Q + self._temperature * hji_old, ... ) -  / self._temperature
        return hij_new, hji_new


class NonTrackingInferrerExplicit(NonTrackingInferrerRegion):
    """
        Computes the permanent almost explicitely (cutting off long distances and being implicit for short distances).
        It works only in the case when the situation is clustered into groups of 3, 4 particles.
    """
    # TODO: write it


class NonTrackingInferrerRegionNaive(NonTrackingInferrerRegion):
    """
        This class implements the Naïve method, where we only sum over the matching and not over non.
        It is naïve in the sense that it avoids a priori unneccesarily complicated formulations.
        It is theoretically justified in a weak sense detailed in the report.
    """

    def marginal_minusLogLikelihood_phantom(self, dr, frame_index):
        # self._stored_hij[frame_index] = [None] * (M - max([0, int(Delta)]))
        '''
        mlnL = -logsumexp([log(self.p_m(n_on, delta, N)) + lng(M + 1 - n_on) - lng(M + 1) - lng(N + 1) \
                           - self.sum_product_energy(dr, frame_index, n_on, n_on - delta, N, M) \
                           for n_on in range(max([0, int(delta)]), int(M))])  # Why use lng ?
        '''
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1]).astype(int)
        N = np.sum(self._particle_count[self._region_indices, frame_index]).astype(int)
        delta = M - N
        n_on = int(M)  # choosing N_off such that p_on == p_off (on average... sort of)
        self.vprint(4, f"n_on={n_on}")
        mlnL = self.sum_product_energy(dr, frame_index, n_on, n_on - delta, N, M)
        return mlnL

    def Q_ij(self, n_off, n_on, dr, frame_index):
        """
        Matrix of log-probabilities for assignments with appearances and disappearances.
        Structure of Q_ij:
         _                          _
        | Q_11  Q_12 .. Q_1M | poff  |
        | Q_21  Q_22 .. Q_2M | poff  |
        |  :     :       :   |   :   |
        | Q_N1  Q_N2 .. Q_NM | poff  |
        |----------------------------|
        | pon   pon  .. pon  | 1-pon |
        |_pon   pon  .. pon  | 1-pon_|
        :param n_off: The number of disappearing particles
        :param n_on:  The number of appearing particles
        :param dr:  The displacement vectors
        :param frame_index: The index of the current frame
        :return: The square matrix Q_ij of size N + n_on - n_off
        """
        n_on = int(n_on)
        n_off = int(n_off)
        N = np.sum(self._particle_count[self._region_indices, frame_index])
        N = int(N)
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1])
        M = int(M)
        if self._phantom_particles:
            assert (M == N - n_off + n_on)  # just a consistency check
        out = zeros([n_on + N, n_off + M])
        LM = ones([n_on + N, n_off + M])  # * self._region_area
        lnp = self.minus_lq_minus_lnp_ij(dr, frame_index)
        out[:N, :M] = lnp - log(LM[:N, :M] / (1 - self._parent_p_off))
        if self._correct_p_off_MC is True:
            out[:N, M:] = -log(LM[:N, M:] / self._p_off)
            out[N:, :M] = -log(LM[N:, :M] * self._region_area / self._p_off)
        elif self._correct_p_off_MC is False:
            out[:N, M:] = -log(LM[:N, M:] / self.corrected_poff_array(frame_index))
            out[N:, :M] = -log(LM[N:, :M] / self.corrected_poff_array(frame_index + 1))
        else:
            raise ValueError(f"Invalid value for `correct_p_off_MC`. Please choose either `True` or `False`.")
        out[N:, M:] = 2. * self._minlnL  # Note: Why not 0 ??
        naive_sum_non = False
        if naive_sum_non is False:
            out[N:, M:] = (1 - self._parent_p_off)
        return out

    def MPAscore(self, dr, frame_index):
        """
        :param dr: The distance matrix for frame_index
        :param frame_index: The current frame index
        :return: the MPAscore for frame_index
        """
        M_index_to_infer = self._particle_count[self._index_to_infer, frame_index + 1]
        N_index_to_infer = self._particle_count[self._index_to_infer, frame_index]
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1])
        N = np.sum(self._particle_count[self._region_indices, frame_index])
        Delta = M - N
        if (self._p_off == 0.) & (self._mu_on == 0.) & (Delta == 0):
            Q = self.Q_ij(0, 0, dr, frame_index)
            mpa_row, mpa_col = kuhn_munkres(-Q)
            max_score = -self.MPA_energy(Q, mpa_row, mpa_col)
        elif (M_index_to_infer == 0) or (N_index_to_infer == 0):
            max_score = 0.  # if the cell to infer is empty in one frame, we gain 0 information
        else:
            Q = self.Q_ij(M - Delta, M, dr, frame_index)
            mpa_row, mpa_col = kuhn_munkres(-Q)
            E_MPA = self.MPA_energy(Q, mpa_row, mpa_col)
            max_score = -E_MPA
        return -max_score
