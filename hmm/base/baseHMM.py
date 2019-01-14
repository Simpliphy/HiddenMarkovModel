from abc import ABCMeta, abstractmethod

class baseHMM(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def generate_sample(self):
        raise NotImplementedError

    @abstractmethod
    def do_M_step(self):
        raise NotImplementedError

    @abstractmethod
    def do_E_step(self):
        raise NotImplementedError

    @abstractmethod
    def get_transition_probabilities(self):
        raise NotImplementedError

    @abstractmethod
    def get_emission_probabilities(self):
        raise NotImplementedError




