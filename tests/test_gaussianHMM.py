import numpy as np
import pytest

from hmm.GaussianHMM import GaussianHMM


@pytest.fixture(scope='module')
def gaussian_hmm():

    gaussianHMM = GaussianHMM(number_of_states=3)

    gaussianHMM.initial_state = np.array([0, 0, 1.0])
    gaussianHMM.transition_probabilities = np.array([[0.8, 0.1, 0.1],
                                                     [0.2, 0.6, 0.2],
                                                     [0.05, 0.05, 0.9]])

    gaussianHMM.emission_probabilities_parameters = np.array([[-3.0, 0.4],
                                                              [0.0, 0.2],
                                                              [3.0, 0.2]])
    return gaussianHMM

def test_emit_value(gaussian_hmm):

    values = list()
    for _ in range(100):
        values.append(gaussian_hmm._emit_value(state=0))

    values = np.array(values)

    assert np.mean(values) >= -3 - 0.4*1.96/10 and np.mean(values) <= -3 + 0.4*1.96/10
    assert np.std(values) >= 0.1 and np.std(values) <= 0.7 # TODO used chi2 confidence interval

def test_get_sigma_for_state(gaussian_hmm):

    assert gaussian_hmm._get_sigma_for_state(state=0) == 0.4

def test_get_mu_for_state(gaussian_hmm):

    assert gaussian_hmm._get_mu_for_state(state=0) == -3

def test_simulate_transition(gaussian_hmm):

    state = 0
    states_list = [state]

    for _ in range(20):
        state = gaussian_hmm._simulate_transition(state)
        states_list.append(state)

    assert min(states_list) >= 0
    assert max(states_list) <= 2
    assert all([isinstance(state, int) for state in states_list])

def test_generate_sample(gaussian_hmm):

    result = gaussian_hmm.generate_sample(100)

    states = result["states"]
    observations = result["observations"]

    assert len(states) == 100
    assert len(observations) == 100
    assert min(states) == 0
    assert max(states) == 2

def test_initialization(gaussian_hmm):

    observations = np.load("observation_test.npy")
    #gaussian_hmm.fit(observations)

    best_pi, best_mu, best_sigma, best_gamma = gaussian_hmm._calculate_initial_states_distribution(observations)

    assert best_pi is not None
    assert best_mu is not None
    assert best_sigma is not None
    assert best_gamma is not None

def test_E_step(gaussian_hmm):

    observations = np.array([3.0 for _ in range(100)])

    gaussian_hmm.emission_probabilities_parameters = np.array([[-3.0, 0.4],
                                                              [0.0, 0.2],
                                                              [3.0, 0.2]])
    gaussian_hmm._do_E_step(observations)

    states_distribution = gaussian_hmm._states_distribution_calculation
    states = np.argmax(states_distribution, axis=1)

    assert states_distribution.min() >= 0.0
    assert states_distribution.max() <= 1.0
    assert len(states) == 100
    assert all([state_index == 2 for state_index in states])


def test_calculate_new_transition_matrix(gaussian_hmm):

    observations = np.array([3.0 for _ in range(100)])

    gaussian_hmm.emission_probabilities_parameters = np.array([[-3.0, 0.4],
                                                               [0.0, 0.2],
                                                               [3.0, 0.2]])

    gaussian_hmm._calculate_new_transition_matrix(observations)
    transition_matrix_calculated = gaussian_hmm._transition_probabilitities_calculation

    sum_axis = np.sum(transition_matrix_calculated, axis=1)

    assert transition_matrix_calculated[2, 2] >= 0.98
    assert all([0.99 <= sum_of_line <= 1.0 for sum_of_line in sum_axis])


def test_calculate_new_emission_probabilities_parameters(gaussian_hmm):

    number_of_observations = 100
    observations = np.array([3.0 for _ in range(number_of_observations)])
    states_distribution = np.array([[1.0, 0.0, 0.0] for _ in range(number_of_observations)])

    gaussian_hmm.emission_probabilities_parameters = np.array([[-3.0, 0.4],
                                                               [0.0, 0.2],
                                                               [3.0, 0.2]])

    gaussian_hmm._states_distribution_calculation = states_distribution

    gaussian_hmm._calculate_new_emission_probabilities_parameters(observations)
    parameters = gaussian_hmm._parameters

    assert 2.9 <= parameters[0][0] <= 3.1

def test_train_with_expectation_maximization(gaussian_hmm):
    assert False

def test_variational_lower_bound(gaussian_hmm):
    assert False

