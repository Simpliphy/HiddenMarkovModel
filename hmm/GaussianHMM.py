import numpy as np
import numbers
from scipy.stats import norm, multivariate_normal
import math
from tqdm import tqdm

from hmm.base.baseHMM import baseHMM
from hmm.GaussianSoftClustering import GaussianSoftClustering
from hmm.convergence_monitor import ConvergenceMonitor

np.random.seed(42)
EPSILON = 1e-6

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
        self._normal_random_variables = None
        self._alpha = None
        self._beta = None

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

        self._calculate_new_emission_probabilities_parameters(observations)
        self._initialize_normal_random_variables()
        self._calculate_new_transition_matrix(observations)


    def _calculate_new_transition_matrix(self, observations):

        if self._alpha is None:
            self._do_forward_pass(observations)

        if self._beta is None:
            self._do_backward_pass(observations)

        alpha = self._alpha
        beta = self._beta

        xi = np.zeros((alpha.shape[0],
                        self._number_of_possible_states,
                        self._number_of_possible_states))

        for index_time_step in range(alpha.shape[0] - 1):

            for from_state in range(self._number_of_possible_states):
                for to_state in range(self._number_of_possible_states):

                    prob = alpha[index_time_step, from_state]
                    prob *= self._transition_probabilitities_calculation[from_state, to_state]
                    logprob = np.nan_to_num(self._normal_random_variables[to_state].logpdf(observations[index_time_step+1]))
                    prob *= np.exp(logprob + EPSILON)
                    prob *= beta[index_time_step + 1, to_state]

                    xi[index_time_step, from_state, to_state] += prob

            xi[index_time_step] /= xi[index_time_step].sum()

        transition_probababilities = xi.sum(axis=0)

        row_sums = transition_probababilities.sum(axis=1)
        transition_probababilities = transition_probababilities / row_sums[:, np.newaxis]

        self._transition_probabilitities_calculation = transition_probababilities


    def _calculate_new_emission_probabilities_parameters(self, observations):

        from tools.visualisation import plot_observations_with_states
        import matplotlib.pyplot as plt

        states = np.argmax(self._states_distribution_calculation, axis=1)
        f, ax1 = plt.subplots()
        plot_observations_with_states(observations, states, ax=ax1)
        plt.show()

        for index_state in range(self._number_of_possible_states):

            weights_for_state = self._states_distribution_calculation[:, index_state]
            normalization_constant = weights_for_state.sum()

            #mu = np.sum(observations*weights_for_state)/normalization_constant
            mu = approximate_weighted_median(observations, weights_for_state)
            self._parameters[index_state][0] = mu

            sigma = approximate_weighted_median(np.abs(observations - mu), weights_for_state)
            #variance = np.sum(weights_for_state*(observations - mu)**2)/normalization_constant
            #sigma = math.sqrt(variance)
            self._parameters[index_state][1] = sigma

    def _do_E_step(self, observations):
        """
        find states most probable based on parameters

        :param observations:
        :return:
        """
        self._states_distribution_calculation = self._combined_forward_and_backward_result()

        return return_full_probability(self._alpha)

    def fit(self, observations):

        best_pi, best_mu, best_sigma, best_gamma = self._calculate_initial_states_distribution(observations)

        self._states_distribution_calculation = best_gamma
        self._initialize_transition_matrix(best_gamma)
        self._calculate_new_emission_probabilities_parameters(observations)
        self._train_with_expectation_maximization(observations)

        return self._states_distribution_calculation, self._transition_probabilitities_calculation, self._parameters

    def _initialize_transition_matrix(self, best_gamma):

        states = np.argmax(best_gamma, axis=1)

        transition_matrix = np.zeros((self._number_of_possible_states, self._number_of_possible_states))

        for index_time_step in range(best_gamma.shape[0] -1 ):
               transition_matrix[states[index_time_step], states[index_time_step+1]] += 1

        sum_axis = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / sum_axis[:, np.newaxis]

        self._transition_probabilitities_calculation = transition_matrix


    def _initialize_normal_random_variables(self):

        random_variable = list()

        for index_state in range(self._number_of_possible_states):

            random_variable.append(norm(loc=self._get_mu_for_state(index_state),
                                        scale=self._get_sigma_for_state(index_state)))

        self._normal_random_variables = random_variable

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

    def _calculate_variational_lower_bound(self, observations):
        """
        Each input is numpy array:
        X: (N x d), data points
        gamma: (N x C), distribution q(T)
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)

        Returns value of variational lower bound
        """

        number_of_observations = observations.shape[0]
        number_of_hidden_states = self._states_distribution_calculation.shape[1]

        loss_per_observation = np.zeros(number_of_observations)

        for state_index in range(number_of_hidden_states):

            mu = self._get_mu_for_state(state_index)
            sigma = self._get_sigma_for_state(state_index)
            gamma_state = self._states_distribution_calculation[:, state_index]

            entropy = gamma_state * np.log(gamma_state + 1e-6)

            loss_per_observation += gamma_state * norm.logpdf(observations, loc=mu, scale=sigma)
            loss_per_observation -= entropy

        total_loss = np.sum(loss_per_observation)

        return total_loss

    def _calculate_initial_states_distribution(self, observations):

        k = self._number_of_possible_states

        gaussian_clustering_model = GaussianSoftClustering()
        X = np.array([[item] for item in observations])
        best_loss, best_pi, best_mu, best_sigma, best_gamma = gaussian_clustering_model.train_EM(X,
                                                                                                 number_of_clusters=k,
                                                                                                 rtol=1e-5,
                                                                                                 max_iter=100,
                                                                                                 restarts=40)

        return best_pi, best_mu, best_sigma, best_gamma

    def _train_with_expectation_maximization(self, observations):

        max_iterations = 10
        min_iteration = 5

        self._initialize_normal_random_variables()
        self._do_M_step(observations)
        previous_likelihood = self._do_E_step(observations)

        convergence_monitor = ConvergenceMonitor()
        convergence_monitor.append(self._parameters,
                                   previous_likelihood,
                                   -1,
                                   self._states_distribution_calculation,
                                   self._transition_probabilitities_calculation)

        for iteration in tqdm(range(max_iterations)):

            self._do_M_step(observations)
            likelihood = self._do_E_step(observations)
            convergence_monitor.append(self._parameters,
                                       likelihood,
                                       iteration,
                                       self._states_distribution_calculation,
                                       self._transition_probabilitities_calculation)

            if abs(likelihood - previous_likelihood)/previous_likelihood <= 0.001 and iteration >= min_iteration :
                break

        convergence_monitor.show_sigmas()
        convergence_monitor.show_mus()
        convergence_monitor.show_likelihood()
        convergence_monitor.show_states()

    def _do_forward_pass(self, observations):

        number_of_observations = len(observations)
        alpha = np.zeros((number_of_observations, self._number_of_possible_states))

        for index_state in range(self._number_of_possible_states):

            logprob = self._normal_random_variables[index_state].logpdf(observations[0])
            probability_of_state = np.exp(np.nan_to_num(logprob) + EPSILON)
            alpha[0, index_state] = probability_of_state

        alpha[0] /= sum(alpha[0])

        for index_time_step in range(1, number_of_observations):

            new_transitional_probabilities = np.dot(alpha[index_time_step-1], self._transition_probabilitities_calculation)
            for index_state in range(self._number_of_possible_states):

                logprob = self._normal_random_variables[index_state].logpdf(observations[index_time_step])
                probability_of_state = np.exp(np.nan_to_num(logprob) + EPSILON)

                probability_of_state = probability_of_state*new_transitional_probabilities[index_state]
                alpha[index_time_step, index_state] = probability_of_state

        self._alpha = alpha

    def _do_backward_pass(self, observations):

        number_of_observations = len(observations)
        beta = np.zeros((number_of_observations, self._number_of_possible_states))

        for index_state in range(self._number_of_possible_states):
            beta[-1, index_state] = 1.0

        for index_time_step in range(number_of_observations - 2, -1, -1):

            observation_probabilities = np.zeros(self._number_of_possible_states)
            for index_state in range(self._number_of_possible_states):

                logprob = norm.logpdf(observations[index_time_step + 1],
                                                     loc=self._get_mu_for_state(index_state),
                                                     scale=self._get_sigma_for_state(index_state))

                observation_probabilities[index_state] = math.exp(np.nan_to_num(logprob) + EPSILON)

            new_transitional_probabilities = np.dot(self._transition_probabilitities_calculation,
                                                    observation_probabilities)

            for index_state in range(self._number_of_possible_states):

                probability_of_state = beta[index_time_step + 1, index_state] * new_transitional_probabilities[index_state]
                beta[index_time_step, index_state] = probability_of_state

        self._beta = beta


    def _combined_forward_and_backward_result(self):

        assert self._alpha is not None, "The attribute self._alpha must have been calculate using the forward pass"
        assert self._beta is not None, "The attribute self._beta must have been calculate using the backward pass"

        alpha = self._alpha
        beta = self._beta

        product = alpha*beta

        row_sums = product.sum(axis=1)
        states_distribution = product / row_sums[:, np.newaxis]

        return states_distribution

def return_full_probability(alpha):
    return alpha[-1].sum()


def normpdf(x, mean, sd):

    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


def norm_logpdf(x, mean, sd):

    var = float(sd) ** 2

    denom = (2 * math.pi * var) ** .5
    num = -(float(x) - float(mean)) ** 2 / (2 * var)
    num = min(num, 1e-6)

    return num - np.log(denom)

def approximate_weighted_median(observations, weights):

    total_weight = sum(weights)
    indexes_for_sorted_values = np.argsort(observations)

    sorted_values = observations[indexes_for_sorted_values]
    sorted_weights = weights[indexes_for_sorted_values]/total_weight

    index_upper_median = np.argmax(np.cumsum(sorted_weights) >= 0.5)
    upper_median = sorted_values[index_upper_median]

    return upper_median



