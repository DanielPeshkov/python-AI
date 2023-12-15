import numpy as np

# ReLU Activation
class ReLU:

	# Forward Pass Function
	def forward(self, inputs, training):

		# Save inputs for Backpropagation
		self.inputs = inputs

		# Calculate output
		self.output = np.maximum(0, inputs)

	# Backward Pass Function
	def backward(self, dvalues):

		# Copy values to not affect original variable
		self.dinputs = dvalues.copy()

		# Zero gradient where inputs were negative
		self.dinputs[self.inputs <= 0] = 0

	# Calculate predictions for outputs
	def predictions(self, outputs):
		return outputs


# Softmax Activation
class Softmax:

	# Forward Pass Function
	def forward(self, inputs, training):

		# Calculate unnormalized probabilities
		probabilities = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

		# Normalize outputs
		self.output = probabilities / np.sum(probabilities, axis=1, keepdims=True)

	# Backward Pass Function
	def backward(self, dvalues):

		# Create uninitialized array
		self.dinputs = np.empty_like(dvalues)

		# Enumerate outputs and gradients
		for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
			# Flatten output array
			single_output = single_output.reshape(-1, 1)
			# Calculate Jacobian matrix of the output
			jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
			# Calculate sample-wise gradient
			self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

	# Calculate predictions for outputs
	def predictions(self, outputs):
		return np.argmax(outputs, axis=1)

# Sigmoid Activation
class Sigmoid:

	# Forward Pass Function
	def forward(self, inputs, training):

		# Save input and calculate output
		self.inputs = inputs
		self.output = 1 / (1 + np.exp(-inputs))

	# Backward Pass
	def backward(self, dvalues):
		# Derivative calculates from output of sigmoid
		self.dinputs = dvalues * (1 - self.output) * self.output

	# Calculate predictions for outputs
	def predictions(self, outputs):
		return (outputs > 0.5) * 1

# Linear activation for regression
class Linear:

	# Forward Pass
	def forward(self, inputs, training):

		# Save values
		self.inputs = inputs
		self.output = inputs

	# Backward Pass
	def backward(self, dvalues):
		# Derivative is 1
		self.dinputs = dvalues.copy()

	# Calculate predictions for outputs
	def predictions(self, outputs):
		return outputs
