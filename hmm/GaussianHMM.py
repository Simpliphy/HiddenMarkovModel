import numpy as np
import numbers
from scipy.stats import norm, multivariate_normal
import math
from numpy.linalg import det
from tqdm import tqdm

from hmm.base.baseHMM import baseHMM
from hmm.GaussianSoftClustering import GaussianSoftClustering
np.random.seed(42)

class GaussianHMM(baseHMM):

    """
    implementation details:

    - The states are defined as an integer between 0 (inlcuded) and the number of states (excluded) to be used
        as an index

    - Transition matrix line = initial state
                    columns = outgoing state

    - numpy arrays are used for the implementation of the properties

    """
    def __init__(self, number_of_states):

        super().__init__()

        self._parameters = list()
        self._number_of_possible_states = number_of_states

        self._inital_state_calculation = None
        self._transition_probabilitities_calculation = None
        self._states_distribution_calculation = None

    def generate_sample(self, number_of_data):
        """
        Pure
        """

        self._is_ready_to_generate_sample()

        states_list = list()
        values_list = list()

        states_list.append(self._get_initial_state_index())
        current_state = states_list[-1]

        values_list.append(self._emit_value(current_state))

        for _ in range(1, number_of_data):

            previous_state = states_list[-1]
            new_state = self._simulate_transition(previous_state)

            states_list.append(new_state)
            values_list.append(self._emit_value(new_state))

        return {"states": states_list, "observations": values_list}

    def _do_M_step(self, observations):
        """
        find parameters maximization of likelihood based on states
        :return:
        """
        print("M-step")

        self._calculate_new_transition_matrix(observations)
        self._calculate_new_emission_probabilities_parameters(observations)


    def _calculate_new_transition_matrix(self, observations):

        transition_probababilities = np.zeros((self._number_of_possible_states,
                                               self._number_of_possible_states))

        states_distributions = self._states_distribution_calculation.copy()

        for index_time_step in range(len(observations) - 1):
            for from_state in range(self._number_of_possible_states):
                for to_state in range(self._number_of_possible_states):

                    from_state_probability = states_distributions[index_time_step, from_state]
                    to_state_probability = states_distributions[index_time_step + 1, to_state]

                    from_state_observation = observations[index_time_step]
                    to_state_observation = observations[index_time_step + 1]

                    from_state_sigma = self._get_sigma_for_state(from_state)
                    from_state_mu = self._get_mu_for_state(from_state)
                    from_state_distribution = norm(loc=from_state_mu, scale=from_state_sigma )

                    to_state_sigma = self._get_sigma_for_state(to_state)
                    to_state_mu = self._get_mu_for_state(to_state)
                    to_state_distribution = norm(loc=to_state_mu, scale=to_state_sigma)

                    p_emission_from = np.nan_to_num(from_state_distribution.pdf(from_state_observation))
                    p_emission_to = np.nan_to_num(to_state_distribution.pdf(to_state_observation))

                    probability_from = from_state_probability*p_emission_from
                    probability_to = to_state_probability*p_emission_to

                    transition_probababilities[from_state, to_state] += probability_from*probability_to

        transition_probababilities = np.nan_to_num(transition_probababilities)

        random_matrix =  np.random.normal(1e-3, 1e-4, transition_probababilities.shape)
        transition_probababilities = np.maximum(transition_probababilities, random_matrix)

        row_sums = transition_probababilities.sum(axis=1)
        transition_probababilities = transition_probababilities / row_sums[:, np.newaxis]


        self._transition_probabilitities_calculation = transition_probababilities

    def _calculate_new_emission_probabilities_parameters(self, observations):

        parameters = np.zeros((self._number_of_possible_states, 2))
        gamma = self._states_distribution_calculation
        X = np.array([[item] for item in observations])

        N = X.shape[0]  # number of objects
        C = gamma.shape[1]  # number of clusters
        d = X.shape[1]  # dimension of each object

        normalizer = np.sum(gamma, 0)  # (K,)
        # print normalizer.shape
        mu = np.dot(gamma.T, X) / normalizer.reshape(-1, 1)
        pi = normalizer / N
        sigma = np.zeros((C, d, d))
        # for every k compute cov matrix
        for k in range(C):
            x_mu = X - mu[k]
            gamma_diag = np.diag(gamma[:, k])

            sigma_k = np.dot(np.dot(x_mu.T, gamma_diag), x_mu)
            sigma[k,...] = (sigma_k) / normalizer[k]

            #parameters[k, 1] = math.sqrt(np.sum(x_mu ** 2)) / normalizer[k]
        for k in range(C):
            parameters[k,0] = mu[k]
            parameters[k, 1] = sigma[k,0]
        """
        for state_index in range(self._number_of_possible_states):
            prob_for_state = self._states_distribution_calculation[:, state_index]
            cst = sum(prob_for_state)
           # print(cst)
            mu = np.sum(observations*prob_for_state)/cst
            #print(mu)
            x_mu = (observations - mu)*prob_for_state
            sigma = math.sqrt(np.sum(x_mu**2))/cst

            parameters[state_index, 0] = mu
            parameters[state_index, 1] = sigma
        """

        self._parameters = parameters

    def _do_E_step(self, observations):
        """
        find states most probable based on parameters

        :param observations:
        :return:
        """
        print("E_step")

        number_of_observations = len(observations)
        states_distribution = np.zeros((number_of_observations, self._number_of_possible_states))

        for index_time_step in range(number_of_observations):
            for index_state in range(self._number_of_possible_states):

                state_sigma = self._get_sigma_for_state(index_state)
                state_mu = self._get_mu_for_state(index_state)

                state_emission_distribution = norm(loc=state_mu, scale=state_sigma)
                probability_of_state = state_emission_distribution.pdf(observations[index_time_step])

                probability_of_state = np.nan_to_num(probability_of_state)
                probability_of_state = max(probability_of_state, np.random.normal(1e-3, 1e-4, 1)[0])

                states_distribution[index_time_step, index_state] = probability_of_state

        row_sums = states_distribution.sum(axis=1)
        states_distribution = states_distribution / row_sums[:, np.newaxis]

        self._states_distribution_calculation = states_distribution

    def fit(self, observations):

        best_pi, best_mu, best_sigma, best_gamma = self._calculate_initial_states_distribution(observations)

        self._states_distribution_calculation = best_gamma
        self._train_with_expectation_maximization(observations)

        return self._states_distribution_calculation, self._transition_probabilitities_calculation, self._parameters

    @property
    def initial_state(self):
       return self._initial_state

    @initial_state.setter
    def initial_state(self, value):
        assert len(value) > 1, "the initial state must have a length greater than one"
        assert isinstance(value, np.ndarray), "the initial state must be an numpy array"
        assert (sum(value) == 1 and max(value) == 1), "only one value must be one"

        self._initial_state = value

    @property
    def transition_probabilities(self):
        return self._transition_probabilitities

    @transition_probabilities.setter
    def transition_probabilities(self, transition_matrix):

        # TODO check square and size
        assert isinstance(transition_matrix, np.ndarray), "the initial state must be an numpy array"

        for line in transition_matrix:
            assert sum(line) == 1.0, "the line mut sum to one"

        assert transition_matrix.min() >= 0, "all value must be positive"
        assert transition_matrix.max() <= 1, "all value must be lower or equal to one"

        self._transition_probabilitities = transition_matrix

    @property
    def emission_probabilities_parameters(self):
        return self._parameters

    @emission_probabilities_parameters.setter
    def emission_probabilities_parameters(self, parameters):
        """
        The expected format is a list. Element is a combinaison [mu, sigma]. The values are real numbers.

        :param parameters:
        :return:
        """
        assert isinstance(parameters, np.ndarray), "A numpy array is expected"
        for combinaison in parameters:
            assert len(combinaison) == 2, "Two elements are expected per state"
            assert combinaison[1] >= 0, "The sigma must be positive"
            assert isinstance(combinaison[0], numbers.Real), "The mean (mu) must be a real number"
            assert isinstance(combinaison[1], numbers.Real), "The std deviation (sigma) must be a real number"

        self._parameters = parameters

    def _is_ready_to_generate_sample(self):

        assert self._initial_state is not None, "inital state must be initialized"
        assert self._transition_probabilitities is not None, "Transition probabilities mst be initialized"
        assert self._parameters is not None, "The parameters for the emission probabilities must be defined"
        assert self._number_of_possible_states is not None

    def _get_initial_state_index(self):
        return int(np.argmax(self.initial_state))

    def _emit_value(self, state):

        assert isinstance(state, int)
        assert state >= 0

        sigma = self._get_sigma_for_state(state)
        mu = self._get_mu_for_state(state)

        return np.random.normal(loc=mu, scale=sigma, size=1)

    def _get_sigma_for_state(self, state):
        assert self._parameters is not None, "parameters must be defined"

        return self._parameters[state][1]

    def _get_mu_for_state(self, state):
        assert self._parameters is not None, "parameters must be defined"

        return self._parameters[state][0]

    def _simulate_transition(self, current_state):

        transition_probabilities = self.transition_probabilities[current_state]
        new_state = np.random.choice(self._number_of_possible_states, p=transition_probabilities)
        new_state = int(new_state)

        return new_state

    def _calculate_variational_lower_bound(self,  X, pi, mu, sigma, gamma):


        """
        Each input is numpy array:
        X: (N x d), data points
        gamma: (N x C), distribution q(T)
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)

        Returns value of variational lower bound
        """
        N = X.shape[0]  # number of objects
        C = gamma.shape[1]  # number of clusters
        d = X.shape[1]  # dimension of each object


        loss = 0
        for cluster_index in range(C):
            dist = multivariate_normal(mu[k], sigma[k], allow_singular=True)
            for n in range(N):
                loss += gamma[n, cluster_index] * (
                            np.log(pi[cluster_index] + 0.00001) + dist.logpdf(X[n, :]) - np.log(gamma[n, k] + 0.000001))

        loss = np.zeros(N)
        EPSILON = 1e-10
        for k in range(C):
            loss += gamma[:, k] * (np.log(pi[k]) + multivariate_normal.logpdf(X, mean=mu[k, :], cov=sigma[k, ...]) - \
                                   np.log(gamma[:, k]))
            # loss+=gamma[:,k]*(np.log(pi[k]*multivariate_normal.pdf(X, mean=mu[k,:], cov=sigma[k,...])+0)-np.log(gamma[:,k]))
            # loss+=gamma[:,k]*(np.log(pi[k]*gauss_den(X,mu[k,:],sigma[k,...],d)+EPSILON)-np.log(gamma[:,k]+EPSILON))

        return np.sum(loss)

    def _calculate_initial_states_distribution(self, observations):

        gaussian_clustering_model = GaussianSoftClustering()
        X = np.array([[item] for item in observations])
        best_loss, best_pi, best_mu, best_sigma, best_gamma = gaussian_clustering_model.train_EM(X, 3, restarts=3)

        return best_pi, best_mu, best_sigma, best_gamma

    def _train_with_expectation_maximization(self, observations):

        max_iterations = 100

        for _ in tqdm(range(max_iterations)):

            self._do_M_step(observations)
            self._do_E_step(observations)

