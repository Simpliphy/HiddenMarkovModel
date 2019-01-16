import matplotlib.pyplot as plt
import numpy as np


def plot_observations_with_states(observations, states):

    if not isinstance(observations, np.ndarray):
        observations = np.array(observations)

    if not isinstance(states, np.ndarray):
        states = np.array(states)

    possible_states = np.unique(states)

    number_of_observation = len(observations)
    observations_index = np.array(range(number_of_observation))

    plt.plot(observations_index, observations, "k-")

    for state_index in possible_states:

        indexes_for_state = np.where(states == state_index)

        x = observations_index[indexes_for_state]
        y = observations[indexes_for_state]

        plt.plot(x, y, ".", label=state_index, markersize=10)

    plt.ylabel("observations")
    plt.xlabel("time step")
    plt.legend()
    plt.tight_layout()
    plt.show()

