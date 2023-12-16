from pythonai.activations import *
import numpy as np

# Test linear activation class
def test_linear() -> None:
	# Initialize layer
	linear = Linear()

	# Test forward method
	test_values = list(range(-5, 6))
	linear.forward(test_values)
	assert linear.output == test_values
	assert linear.inputs == test_values

	# Test backward method
	linear.backward(test_values)
	assert linear.dinputs == test_values

	# Test predictions method
	assert linear.predictions(test_values) == test_values

# Test ReLU activation class
def test_relu() -> None:
	# Initialize layer
	relu = ReLU()

	# Test forward method
	test_values = np.array(range(-5, 6))
	relu.forward(test_values)
	masked_values = np.array([0,0,0,0,0,0,1,2,3,4,5])
	assert np.array_equal(relu.output, masked_values)
	assert np.array_equal(relu.inputs, test_values)

	# Test backward method
	relu.backward(test_values)
	assert np.array_equal(relu.dinputs, masked_values)

	# Test predictions method
	assert np.array_equal(relu.predictions(test_values), test_values)


# Test Softmax activation class
def test_softmax() -> None:
	# Initialize layer
	softmax = Softmax()

	# Test forward method
	test_values = np.arange(16)
	test_values = test_values.reshape(4, 4)
	softmax.forward(test_values)
	assert np.mean(np.sum(softmax.output, axis=1)) == 1.0

	# Test backward method on matching input and dvalues
	softmax.backward(test_values)
	assert np.sum(softmax.dinputs) == 0

	# Test backward method on different dvalues
	softmax.backward(np.exp(test_values))
	assert np.sum(softmax.dinputs) != 0
	
	# Test predictions method
	assert np.array_equal(softmax.predictions(test_values), np.array([3, 3, 3, 3]))


# Test Sigmoid activation class
def test_sigmoid() -> None:
	# Initialize layer
	sigmoid = Sigmoid()

	# Test forward method
	test_values = np.zeros((4,))
	sigmoid.forward(test_values)
	assert np.array_equal(sigmoid.inputs, test_values)
	assert np.mean(sigmoid.output) == 0.5

	# Test backward method
	sigmoid.backward(test_values)
	assert np.sum(sigmoid.dinputs) == 0.

	# Test prediction method
	assert np.sum(sigmoid.predictions(test_values)) == 0.
