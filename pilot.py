"""
Problem set

            INPUTS          OUTPUTS
Example 1:  0   0   1       0
Example 2:  1   1   1       1
Example 3:  1   0   1       1
Example 4:  0   1   1       0

New Situation: 1    0   0   ?

What should be the new output?

When the first input is 1, the output should be 1 as well and when its 0, the output should be zero.

A neural network with no hidden layer is called a Perceptron.

Input   Synapses   Neuron   Output

      w1
X1   ------
      w2  |
X2   --------------- Y -------- O
      w3  |
X3  -------

Synapses are the connection between the inputs and the neuron.
Each synapse is given a weight. weights are really important to get the right outcome.(w1, w2, w3).

Neuron is where all the callculations happens. It takes the sum of all the input values, mulitply it by their respective weights. The addendum, then goes through a normalization function. For our normalization funtion, we will use the sigmoid funcion. The correponding value for our solution will always be between 0  and 1.     

Once sigmoid calculates the error using sigmoid, we need to backpropagate to fix the error.

ERROR WEIGHTED DERIVATIVE

adjust weights by = error.input.output
error = output - actual output
"""

import numpy as np

## the normalizing function - sigmoid function
def sigmoid(x):
    return 1/ ( 1 + np.exp(-x))

def sigmoid_derivative(x):
      return x * ( 1-x )

training_inputs = np.array([
                        [ 0, 0, 1],
                        [ 1, 1, 1],
                        [ 1, 0, 1],
                        [ 0, 1, 1]
                  ])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptics weights: ')
print( synaptic_weights )

for iteration in range(100000):
      input_layer = training_inputs
      outputs = sigmoid(np.dot( input_layer, synaptic_weights )) ##dot products
      error = training_outputs - outputs
      adjustments = error * sigmoid_derivative( outputs )
      synaptic_weights += np.dot( input_layer.T, adjustments )

print(" Synaptic Weights after trianing ")
print( synaptic_weights )

print('Output after training: ')
## This will give the training deltas.
print(outputs)
