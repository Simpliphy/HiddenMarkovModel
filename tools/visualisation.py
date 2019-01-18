import matplotlib.pyplot as plt
import numpy as np


def plot_observations_with_states(observations, states, ax = None):

    if not isinstance(observations, np.ndarray):
        observations = np.array(observations)

    if not isinstance(states, np.ndarray):
        states = np.array(states)

    possible_states = np.unique(states)

    number_of_observation = len(observations)
    observations_index = np.array(range(number_of_observation))

    if ax is None:
        f, ax = plt.subplots()

    ax.plot(observations_index, observations, "k-")

    for state_index in possible_states:

        indexes_for_state = np.where(states == state_index)

        x = observations_index[indexes_for_state]
        y = observations[indexes_for_state]

        ax.plot(x, y, ".", label=state_index, markersize=10)

    ax.set_ylabel("observations")
    ax.set_xlabel("time step")
    ax.legend()

    return ax

