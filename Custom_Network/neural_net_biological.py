from scipy.stats import truncnorm
import numpy as np
import random
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))



class Network:

    def __init__(self, dimension_array, learning_rate):
        self.net = []
        self.dimension_array = dimension_array # to access it from other methods
        self.results = []
        for x in range(1,len(dimension_array)):
            self.net.append([])
            for y in range(dimension_array[x]):
                self.net[x-1].append(Neuron(dimension_array[x-1]))
                print(self.net[x-1][y].weights)

    def calc(self, input):

        for x in range(1,len(self.dimension_array)):
            new_input = []
            for y in range(self.dimension_array[x]):
                new_input.append(self.net[x-1][y].calc(input))
            self.results.append(new_input)
            input = new_input
        print("results:", self.results)

        return input



    def evolve(self, input, expectation):
        prediction = self.calc(input)
        error = prediction - expectation
        for x in reversed(1,len(self.dimension_array)):
            new_error = error
            for y in range(self.dimension_array[x]):
                None
                # self.net[x-1][y].error =




class Neuron:
    def __init__(self, neurons_in_layer_before):
        self.weights = []
        for x in range(neurons_in_layer_before):
            self.weights.append(random.gauss(0,1))
        self.act_function = sigmoid
        self.act_function_der = sigmoid_prime
        self.bias = 0
        self.output = 0
    def truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def calc(self, input):
        self.output = self.act_function(float(np.dot(input,self.weights))+self.bias)
        return self.output

    def print(self):
        print(self.weights)


if __name__ == '__main__':

    net = Network([1, 2, 1],0.3)
    net.calc(1)
    # training_data = [[1,0,0],[1,1,0],[1,1,1],[1,0,1],[0,1,0],[0,0,1]]
    # result_data = [[0,1],[0,1],[1,0],[1,0],[0,1],[1,0]]
    # for i in range(1000):
    #     for y in range(6):
    #         net.evolve(training_data[y],result_data[y])
