from scipy.stats import truncnorm
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x)*(oneMinus(sigmoid(x)))

def oneMinus(x):
    return np.subtract(1, x)


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class Network:

    def __init__(self, dimensionArray, learning_rate):
        self.dimensionArray = dimensionArray
        self.weights = []
        self.layerResultArray = []
        self.learning_rate = learning_rate
        self.fitness = [0, 0]  # subject to change later
        self.count = 0
        self.layer_errors = []
        self.layerResultArray = [] # stores results of each layer normalized
        self.layerResultArrayNonNorm = [] # stores result of each layer before normalization
        self.createWeights() # (duh)

    def calc(self, input):
        print("Input:", input)
        # input should come in array form
        input = np.array(input, ndmin=2).T
        self.layerResultArray.clear()
        self.layerResultArrayNonNorm.append(input)
        self.layerResultArray.append(sigmoid(input))
        layerResult = input  # sets the result of first layer to the input
        for i in range(len(self.weights)):
            layerResult = np.dot(self.weights[i], layerResult)
            self.layerResultArrayNonNorm.append(layerResult)
            layerResult = sigmoid(layerResult)
            self.layerResultArray.append(layerResult)

        return self.layerResultArray[len(self.layerResultArray) - 1]

    def evolve(self, input, result):
        self.layer_errors.clear()
        output = np.array(self.calc(input), ndmin=2)
        result = np.array(result, ndmin=2).T
        # for i in range(len(self.layerResultArray)-1):
        #     self.layerResultArray[i] = sigmoid(self.layerResultArray[i])

        network_error = (result - output)[0][0]*sigmoid_prime(output)  # getting the integer value of the network error, it's initially a double array ([[network_error]]
        print("network_error:",network_error)
        self.layer_errors.append(np.dot(self.weights[len(self.weights) - 1].T, network_error))

        print("original weights:", self.weights)

        # updating the weights
        tmp = network_error * sigmoid_prime(self.layerResultArray[len(self.layerResultArray) - 1])
        tmp = self.learning_rate * np.dot(tmp, self.layerResultArray[len(self.layerResultArray) - 2].T)
        self.weights[len(self.weights) - 1] += tmp
        print("tmp:", tmp)

        for i in reversed(range(len(self.weights) - 1)):
            tmp = self.layer_errors[len(self.layer_errors)-1] * sigmoid_prime(self.layerResultArray[i+1])

            tmp = self.learning_rate * np.dot(tmp, self.layerResultArray[i].T)
            self.weights[i] += tmp

            self.layer_errors.append(sigmoid(np.dot( self.weights[i].T, self.layer_errors[len(self.layer_errors)-1])))
        # print(self.layerResultArray[len(self.layerResultArray) - 1])
        # print(self.weights)


    def createWeights(self):
        trunc = truncated_normal(mean=0, sd=1, low=-3, upp=3)
        for i in range(1, len(self.dimensionArray)):
            self.weights.append(trunc.rvs((self.dimensionArray[i], self.dimensionArray[i - 1])))


if __name__ == '__main__':
    net = Network([2, 2, 2, 1], 0.3)
    training_data = [[0, 1], [1, 1], [1, 0], [0, 0]]
    # result_data = [0,0,1,1,0,1]
    result_data = [[1], [1], [0], [0]]
    for i in range(1000):
        for y in range(4):
            net.evolve(training_data[y], result_data[y])

    for i in range(len(training_data)):
        print(net.calc(training_data[i]))
