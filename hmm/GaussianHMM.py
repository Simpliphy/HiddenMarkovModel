import numpy as np
import numbers

from hmm.base.baseHMM import baseHMM

class GaussianHMM(baseHMM):
    """
    implementation details:

    - The states are defined as an integer between 0 (inlcuded) and the number of states (excluded) to be used
        as an index

    - Transition matrix line = initial state
                    columns = outgoing state

    """
    def __init__(self, number_of_states):

        self._parameters = list()
        self._number_of_possible_states = number_of_states
        super().__init__()

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

    def _do_M_step(self):
        raise NotImplementedError

    def _do_E_step(self):
        raise NotImplementedError

    def fit(self, observations):
        raise NotImplementedError

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
        return self._initial_state

    @transition_probabilities.setter
    def transition_probabilities(self, transition_matrix):

        # TODO check square and size
        assert isinstance(transition_matrix, np.ndarray), "the initial state must be an numpy array"

        for line in transition_matrix:
            assert sum(line) == 1.0, "the line mut sum to one"

        assert transition_matrix.min() >= 0, "all value must be positive"
        assert transition_matrix.max() <= 1, "all value must be lower or equal to one"

        self._initial_state = transition_matrix

    @property
    def emission_probabilities_parameters(self):
        return self._emission_probabilities_parameters

    @emission_probabilities_parameters.setter
    def emission_probabilities_parameters(self, parameters):
        """
        The expected format is a list. Element is a combinaison [mu, sigma]. The values are real numbers.

        :param parameters:
        :return:
        """
        assert isinstance(parameters, list), "A list is expected"
        for combinaison in parameters:
            assert len(combinaison) == 2, "Two elements are expected per state"
            assert combinaison[1] >= 0, "The sigma must be positive"
            assert isinstance(combinaison[0], numbers.Real), "The mean (mu) must be a real number"
            assert isinstance(combinaison[1], numbers.Real), "The std deviation (sigma) must be a real number"

        self._emission_probabilities_parameters = parameters

    def _is_ready_to_generate_sample(self):

        assert self._initial_state is not None, "inital state must be initialized"
        assert self._transition_probabilitities is not None, "Transition probabilities mst be initialized"
        assert self._parameters is not None, "The parameters for the emission probabilities must be defined"

    def _get_initial_state_index(self):
        return np.argmax(self.initial_state)

    def _emit_value(self, state):

        assert isinstance(state, int)
        assert state >= 0

        sigma = self._get_sigma_for_state(state)
        mu = self._get_mu_for_state(state)

        return np.random.normal(loc=mu, scale=sigma, size=1)

    def _get_sigma_for_state(self, state):
        assert self._parameters is not None, "parameters must be defined"

        return self.emission_probabilities_parameters[state][1]

    def _get_mu_for_state(self, state):
        assert self._parameters is not None, "parameters must be defined"

        return self.emission_probabilities_parameters[state][0]

    def _simulate_transition(self, current_state):

        transition_probabilities = self.transition_probabilities[current_state]
        new_state = np.random.choice(self._number_of_possible_states, p=transition_probabilities)

        return new_state

