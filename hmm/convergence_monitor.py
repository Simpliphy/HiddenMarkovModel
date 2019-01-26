import matplotlib.pyplot as plt

class ConvergenceMonitor(object):

    def __init__(self):

        self._parameters_list = list()
        self._likelihood_list = list()
        self._iterations_list = list()
        self._states_distributions_list = list()
        self._transition_probabilities_list = list()

    def show_sigmas(self):

        lists_sigmas = list()
        number_of_hidden_states = len(self._parameters_list[0])
        for _ in range(number_of_hidden_states):
            lists_sigmas.append(list())

        #print(len(self._parameters_list))
        for iteration_index in range(len(self._parameters_list)):
            for index_state in range(number_of_hidden_states):
                #print(self._parameters_list[iteration_index])
                #print(index_state)
                lists_sigmas[index_state].append(self._parameters_list[iteration_index][index_state, 1])

        for index_state in range(number_of_hidden_states):
            plt.plot(self._iterations_list, lists_sigmas[index_state], label=str(index_state))

        plt.xlabel("iteration")
        plt.ylabel("sigma")
        plt.title("sigma versus iteration")
        plt.legend()
        plt.show()

    def show_mus(self):

        lists_mus = list()
        number_of_hidden_states = len(self._parameters_list[0])
        for _ in range(number_of_hidden_states):
            lists_mus.append(list())

        for iteration_index in range(len(self._parameters_list)):
            for index_state in range(number_of_hidden_states):
                lists_mus[index_state].append(self._parameters_list[iteration_index][index_state, 0])

        for index_state in range(number_of_hidden_states):
            plt.plot(self._iterations_list, lists_mus[index_state], label=str(index_state))

        plt.xlabel("iteration")
        plt.ylabel("mu")
        plt.title("mu versus iteration")
        plt.legend()
        plt.show()

    def show_likelihood(self):

        plt.plot(self._iterations_list, self._likelihood_list)

        plt.xlabel("iteration")
        plt.ylabel("likelihood")
        plt.title("likelihood versus iteration")
        plt.show()

    def show_states(self):
        pass

    def append(self, parameter, likelihood, iteration,states_distribution, transition_matrix):

        self._parameters_list.append(parameter)
        self._likelihood_list.append(likelihood)
        self._iterations_list.append(iteration)
        self._states_distributions_list.append(states_distribution)
        self._transition_probabilities_list.append(transition_matrix)




