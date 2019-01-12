from scipy.stats import truncnorm
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (oneMinus(sigmoid(x)))


# necessary because 1-matrix isn't possible
def oneMinus(x):
    return np.subtract(1, x)


# special normal distribution used to set the weights
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class Network:

    def __init__(self, dimensionArray, learning_rate):
        self.dimensionArray = dimensionArray
        # defines dimension of the neural net, e.g [5,10,7,2] would take 5 inputs, have 2 hidden layers with 10 and 7 neurons respectively and output and
        # array of 2 values
        self.weights = []  # array containing the weights ('strength of connection between neurons') of the network
        self.layerResultArray = []  # array that contains the value each neuron contains
        self.learning_rate = learning_rate  # rate with which the weights are adjusted via backpropagation
        self.layer_errors = []  # difference between the value that the neurons actually contain and the value that they should contain
        self.layerResultArray = []  # contains the same thing as layerResultArray but after normalizing (between 0 and 1)
        self.layerResultArrayNonNorm = []  # stores result of each layer before normalization
        self.createWeights()  # (duh)

    # Responsible for the forward-propagation/simple calculation
    def calc(self, input):
        print("Input:", input)
        input = np.array(input, ndmin=2).T  # transforms input into numpy array with 2 dimensions
        self.layerResultArray.clear()  # clears the content of the neurons from previous calculation
        self.layerResultArrayNonNorm.append(input)  # adds the input as the value of the first neuron layer
        self.layerResultArray.append(sigmoid(input))
        layerResult = input  # sets the result of first layer to the input
        for i in range(len(self.weights)):
            layerResult = np.dot(self.weights[i],
                                 layerResult)  # neuron values = dotproduct of weights * previous neuron values
            self.layerResultArrayNonNorm.append(layerResult)
            layerResult = sigmoid(layerResult)
            self.layerResultArray.append(layerResult)

        return self.layerResultArray[
            len(self.layerResultArray) - 1]  # returns last value of layerResultArray (and therefore the thing
        # that the last neuron layer contained, hence the result

    # evolve is responsible for improving the network performance via backpropagation
    def evolve(self, input, result):
        self.layer_errors.clear()  # clears errors from previous round
        output = np.array(self.calc(input), ndmin=2)  # matrix conversion
        result = np.array(result, ndmin=2).T
        network_error = (result - output)[0][0]  # the error of the entire network is
        # the difference between prediction and actual  result (the [0][0] is because the
        # output is stored in arrayform [[output]]
        self.layer_errors.append(np.dot(self.weights[len(self.weights) - 1].T, network_error))  # the error
        # of the last layer is the dotproduct of the last weight * network error

        # updating the weights (tmp = temporary variable, a bit too complicated to explain via commenting. watch some tutorial on
        # backpropagation and try to understand it/ask, takes a bit to get behing the math
        tmp = network_error * sigmoid_prime(self.layerResultArray[len(self.layerResultArray) - 1])
        tmp = np.dot(tmp, self.layerResultArray[len(self.layerResultArray) - 2].T)
        self.weights[len(self.weights) - 1] += self.learning_rate * tmp  # adjusts the weight by tmp * the learning rate

        # does the same thing that has been done before for the rest of all layers in a for loop
        for i in reversed(range(len(self.weights) - 1)):
            tmp = self.layer_errors[len(self.layer_errors) - 1] * sigmoid_prime(self.layerResultArray[i + 1])

            tmp = self.learning_rate * np.dot(tmp, self.layerResultArray[i].T)
            self.weights[i] += tmp

            self.layer_errors.append(sigmoid(np.dot( self.weights[i].T, self.layer_errors[len(self.layer_errors)-1])))

    # initiates the weights to random values when network is created
    def createWeights(self):
        trunc = truncated_normal(mean=0, sd=1, low=-3, upp=3)
        for i in range(1, len(self.dimensionArray)):
            self.weights.append(trunc.rvs((self.dimensionArray[i], self.dimensionArray[i - 1])))


if __name__ == '__main__':
    net = Network([2, 2, 2, 1],
                  0.01)  # creates network with dimensions specified in array and learning rate as specified
    training_data = [[0, 1], [1, 1], [1, 0], [0, 0]]  # an array containing the training data
    # result_data = [0,0,1,1,0,1]
    result_data = [[1], [1], [0], [0]]  # the results that the network should find
    for i in range(1000):
        for y in range(4):
            net.evolve(training_data[y], result_data[y])  # trains network with training data

    for i in range(len(training_data)):
        print(net.calc(training_data[i]))  # checks whether training worked by letting it calculate the the same values
