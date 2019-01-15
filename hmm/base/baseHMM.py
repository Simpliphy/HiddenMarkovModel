from abc import ABCMeta, abstractmethod

class baseHMM(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):

        self._initial_state = None
        self._number_of_possible_states = None
        self._transition_probabilitities = None
        self._current_state = None
        self._states_names = None
        self._emission_probabilities_parameters = None

    @abstractmethod
    def generate_sample(self, number_of_data):
        raise NotImplementedError

    @abstractmethod
    def _do_M_step(self):
        raise NotImplementedError

    @abstractmethod
    def _do_E_step(self):
        raise NotImplementedError

    @abstractmethod
    @property
    def transition_probabilities(self):
        raise NotImplementedError

    @abstractmethod
    @transition_probabilities.setter
    def transition_probabilities(self, value):
        raise NotImplementedError

    @abstractmethod
    @property
    def emission_probabilities_parameters(self):
        raise NotImplementedError

    @abstractmethod
    @emission_probabilities_parameters.setter
    def emission_probabilities_parameters(self, value):
        raise NotImplementedError




