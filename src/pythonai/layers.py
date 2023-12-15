import numpy as np

# Input "layer" for Model class training loop
class Layer_Input:

	# Forward Pass
	def forward(self, inputs, training):
		self.output = inputs

# Dense layer
class Dense:

	# Initialization Function
	def __init__(self, input_features, output_features, weight_regularizer_l1=0, 
				weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):

		# Initialize Weights and Biases
		self.weights = 0.01 * np.random.randn(input_features, output_features)
		self.biases = np.zeros((1, output_features))
		# Set Regularization Strength
		self.weight_regularizer_l1 = weight_regularizer_l1
		self.weight_regularizer_l2 = weight_regularizer_l2
		self.bias_regularizer_l1 = bias_regularizer_l1
		self.bias_regularizer_l2 = bias_regularizer_l2

	# Forward Pass Function
	def forward(self, inputs, training):

		# Save inputs for Backpropagation
		self.inputs = inputs

		# Calculate output
		self.output = np.dot(inputs, self.weights) + self.biases

	# Backward Pass Function
	def backward(self, dvalues):

		# Gradients on parameters
		self.dweights = np.dot(self.inputs.T, dvalues)
		self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

		# Gradients on regularization
		# L1 on weights
		if self.weight_regularizer_l1 > 0:
			dL1 = np.ones_like(self.weights)
			dL1[self.weights < 0] = -1
			self.dweights += self.weight_regularizer_l1 * dL1
		# L2 on weights
		if self.weight_regularizer_l2 > 0:
			self.dweights += 2 * self.weight_regularizer_l2 * self.weights
		# L1 on biases
		if self.bias_regularizer_l1 > 0:
			dL1 = np.ones_like(self.biases)
			dL1[self.biases < 0] = -1
			self.dbiases += self.bias_regularizer_l1 * dL1
		# L2 on biases
		if self.bias_regularizer_l2 > 0:
			self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

		# Gradient on values
		self.dinputs = np.dot(dvalues, self.weights.T)

	# Retrieve layer parameters
	def get_parameters(self):
		return self.weights, self.biases

	# Set weights and biases in layer instance
	def set_parameters(self, weights, biases):
		self.weights = weights
		self.biases = biases

# Dropout Layer
class Dropout:

	# Initialization
	def __init__(self, rate):
		# Rate is fraction to drop
		# stored as fraction to keep
		self.rate = 1 - rate

	# Forward Pass
	def forward(self, inputs, training):

		# Save input values
		self.inputs = inputs

		# If not training, return values
		if not training:
			self.output = inputs.copy()
			return

		# Generate and save scaled mask
		self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
		# Apply mask to output values
		self.output = inputs * self.binary_mask

	# Backward Pass
	def backward(self, dvalues):
		# Gradient on values
		self.dinputs = dvalues * self.binary_mask