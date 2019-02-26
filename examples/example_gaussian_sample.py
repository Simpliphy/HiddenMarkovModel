import numpy as np

from hmm.gaussian.GaussianHMM import GaussianHMM
from tools.visualisation import plot_observations_with_states

if __name__=="__main__":

    gaussianHMM = GaussianHMM(number_of_states=3)

    gaussianHMM.initial_state = np.array([0, 0, 1.0])
    gaussianHMM.transition_probabilities = np.array([[0.8, 0.1, 0.1],
                                                     [0.2, 0.6, 0.2],
                                                     [0.05, 0.05, 0.9]])

    gaussianHMM.emission_probabilities_parameters = np.array([[-1.0, 0.4],
                                                              [0.0, 0.2],
                                                              [1.0, 0.2]])

    result = gaussianHMM.generate_sample(100)

    states = result["states"]
    observations = result["observations"]

    plot_observations_with_states(observations, states)

