from hmm.base.baseHMM import baseHMM


class GaussianHMM(baseHMM):

    def __init__(self):

        self._parameters = dict()
        super().__init__()

    def generate_sample(self):
        raise NotImplementedError

    def do_M_step(self):
        raise NotImplementedError

    def do_E_step(self):
        raise NotImplementedError

    def get_transition_probabilities(self):
        raise NotImplementedError

    def get_emission_probabilities(self):
        raise NotImplementedError




