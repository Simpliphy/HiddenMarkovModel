import numpy as np

from hmm.GaussianHMM import GaussianHMM
from tools.visualisation import plot_observations_with_states

import matplotlib.pyplot as plt

if __name__=="__main__":

    gaussianHMM = GaussianHMM(number_of_states=3)

    gaussianHMM.initial_state = np.array([0, 0, 1.0])
    gaussianHMM.transition_probabilities = np.array([[0.8, 0.1, 0.1],
                                                     [0.2, 0.6, 0.2],
                                                     [0.05, 0.05, 0.9]])

    gaussianHMM.emission_probabilities_parameters = np.array([[-3.0, 0.4],
                                                              [0.0, 0.2],
                                                              [3.0, 0.2]])

    result = gaussianHMM.generate_sample(200)

    states = result["states"]
    observations = np.array([item for sublist in result["observations"] for item in sublist])

    states_distribution_calculation, transition_probabilitities, parameters = gaussianHMM.fit(observations)

    states_obtain = np.argmax(states_distribution_calculation, axis=1)

    f, (ax1, ax2) = plt.subplots(1, 2)
    plot_observations_with_states(observations, states, ax=ax1)
    plot_observations_with_states(observations, states_obtain, ax=ax2)
    ax1.set_title("simulated states")
    ax2.set_title("states obtain")
    plt.show()

    print("transition prob")
    print(transition_probabilitities)

    print("param")
    print(parameters)

    print("enf of script")

