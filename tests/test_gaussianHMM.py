import numpy as np
import pytest

from hmm.GaussianHMM import GaussianHMM


@pytest.fixture(scope='module')
def gaussian_hmm():

    gaussianHMM = GaussianHMM(number_of_states=3)

    gaussianHMM.initial_state = np.array([0, 0, 1.0])
    gaussianHMM.transition_probabilities = np.array([[0.2, 0.4, 0.4],
                                                     [0.2, 0.6, 0.2],
                                                     [0.33, 0.33, 0.34]])

    gaussianHMM.emission_probabilities_parameters = np.array([[-1.0, 1.0],
                                                     [0.0, 2.0],
                                                     [2.0, 1.0]])
    return gaussianHMM

def test_generate_sample(gaussian_hmm):

    result = gaussian_hmm.generate_sample(100)

    states = result["states"]
    observations = result["observations"]

    assert len(states) == 100
    assert len(observations) == 100
    assert min(states) == 0
    assert max(states) == 2
