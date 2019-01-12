import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Network:

    def __init__(self, dimension_array, learning_rate):
        self.net = []
        self.dimension_array = dimension_array  # to access it from other methods
        self.results = []
        for x in range(1, len(dimension_array)):  # appends an array for each layer of the network
            self.net.append([])
            for y in range(dimension_array[x]):
                self.net[x - 1].append(Neuron(dimension_array[x - 1]))  # appends neurons in each layer
                print(self.net[x - 1][y].weights)

    def calc(self, input):

        for x in range(1, len(self.dimension_array)):
            new_input = []  # resets variable new_input each iteration
            for y in range(self.dimension_array[x]):
                new_input.append(self.net[x - 1][y].calc(input))  # appends the calculation
                # of the current layer to new_input
            self.results.append(new_input)
            input = new_input  # sets input = new_input so that for the next layer input is not the
            # input into the network anymore but rather the output from the previous layer
        print("results:", self.results)

        return input

    # evolve improves the network via backpropagation, unfinished so far.
    def evolve(self, input, expectation):
        prediction = self.calc(input)
        error = np.subtract(prediction, expectation)
        correction = 0
        for x in reversed(range(1, len(self.dimension_array))):
            new_error = error
            for y in range(self.dimension_array[x]):
                correction = error[y] * self.net[x - 1][y].weights
                self.net[x - 1][y].weights += correction
                # self.net[x-1][y].error =


class Neuron:
    def __init__(self, neurons_in_layer_before):
        self.weights = []  # 'strength' of connection between this neuron and the ones in the previous layer
        for x in range(neurons_in_layer_before):  # appends as many weights as there are neurons
            # in the previous layer with values that follow a random gaussian distribution
            self.weights.append(random.gauss(0, 1))
        self.act_function = sigmoid  # sets the activation function to be sigmoid, for easy modification later
        self.act_function_der = sigmoid_prime  # sets the derivative of the activation function
        self.bias = 0  # bias needed in case an input is 0 since multiplication by the weight would never change a 0
        self.output = 0  # output of the neuron

    def calc(self, input):
        #   multiplies the output of all previous layers' neurons (contained in the input variable) with the weight
        #   of this neurons and adds the bias
        self.output = self.act_function(float(np.dot(input, self.weights)) + self.bias)
        return self.output

    def print(self):  # prints weights for debugging
        print(self.weights)


if __name__ == '__main__':
    net = Network([1, 2, 2], 0.3)
    net.calc(1)
    # net.evolve(1,[0,1])
    # training_data = [[1,0,0],[1,1,0],[1,1,1],[1,0,1],[0,1,0],[0,0,1]]
    # result_data = [[0,1],[0,1],[1,0],[1,0],[0,1],[1,0]]
    # for i in range(1000):
    #     for y in range(6):
    #         net.evolve(training_data[y],result_data[y])
