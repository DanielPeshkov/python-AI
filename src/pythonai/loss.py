import numpy as np

# Parent Loss Class
class Loss:

	# Calculates the data and regularization losses
	# given model output and ground truth values
	def calculate(self, output, y, *, include_regularization=False):

		# Calculate sample losses
		sample_losses = self.forward(output, y)

		# Calculate mean loss
		data_loss = np.mean(sample_losses)

		# Add accumulated sum of losses and sample count
		self.accumulated_sum += np.sum(sample_losses)
		self.accumulated_count += len(sample_losses)

		# If no regularization loss
		if not include_regularization:
			return data_loss

		# Return data and regularization losses
		return data_loss, self.regularization_loss()

	# Calculates accumulated loss
	def calculate_accumulated(self, *, include_regularization=False):

		# Calculate mean loss
		data_loss = self.accumulated_sum / self.accumulated_count

		# If no regularization loss
		if not include_regularization:
			return data_loss

		# Return data and regularization loss
		return data_loss, self.regularization_loss()

	# Reset variables for accumulated loss
	def new_pass(self):

		self.accumulated_sum = 0
		self.accumulated_count = 0

	# Set trainable layers
	def remember_trainable_layers(self, trainable_layers):

		self.trainable_layers = trainable_layers

	# Regularization loss calculation
	def regularization_loss(self):
		
		# 0 by default
		regularization_loss = 0

		# Calculate loss for each trainable layer
		for layer in self.trainable_layers:

			# L1 regularization - weights
			if layer.weight_regularizer_l1 > 0:
				regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

			# L2 regularization - weights
			if layer.weight_regularizer_l2 > 0:
				regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

			# L1 regularization - biases
			if layer.bias_regularizer_l1 > 0:
				regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

			# L2 regularization - biases
			if layer.bias_regularizer_l2 > 0:
				regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

		return regularization_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

	# Forward Pass
	def forward(self, y_pred, y_true):

		# Number of samples
		samples = len(y_pred)

		# Clip data to prevent divide by 0
		# Clip both sides to prevent bias towards 1
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		# Probabilities for target values - Categorical
		if len(y_true.shape) == 1:
			correct_confidences = y_pred_clipped[range(samples), y_true]

		# Probabilities for target values - One-hot Encoded
		elif len(y_true.shape) == 2:
			correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

		# Losses
		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods

	# Backward Pass
	def backward(self, y_pred, y_true):

		# Number of samples
		samples = len(y_pred)
		# Number of outputs per sample
		outputs = len(y_pred[0])

		# Convert outputs from sparse to one-hot
		if len(y_true.shape) == 1:
			y_true = np.eye(outputs)[y_true]

		# Calculate gradient
		self.dinputs = -y_true / y_pred
		# Normalize gradient
		self.dinputs = self.dinputs / samples

# Softmax Classifier - Softmax activation with cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy:

	# Backward Pass
	def backward(self, y_pred, y_true):

		# Number of samples
		samples = len(y_pred)

		# Convert labels from one-hot to sparse
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis=1)

		# Copy values for modification
		self.dinputs = y_pred.copy()
		# Calculate gradient
		self.dinputs[range(samples), y_true] -= 1
		# Normalize gradient
		self.dinputs = self.dinputs / samples

# Binary cross-entropy loss
class Loss_BinaryCrossentropy(Loss):

	# Forward Pass
	def forward(self, y_pred, y_true):

		# Clip data to prevent divide by 0
		y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

		# Calculate sample-wise loss
		sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
		sample_losses = np.mean(sample_losses, axis=1)

		# Return losses
		return sample_losses

	# Backward Pass
	def backward(self, y_pred, y_true):

		# Number of samples
		samples = len(y_pred)
		# Number of outputs per sample
		outputs = len(y_pred[0])

		# Clip data to prevent divide by 0
		y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

		# Calculate gradient
		self.dinputs = -(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped)) / outputs 
		# Normalize gradient
		self.dinputs = self.dinputs / samples

# Mean Squared Error
class Loss_MeanSquaredError(Loss):

	# Forward Pass
	def forward(self, y_pred, y_true):

		# Calculate loss
		sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

		# Return loss
		return sample_losses

	# Backward Pass
	def backward(self, y_pred, y_true):

		# Number of samples
		samples = len(y_pred)
		# Number of outputs per sample
		outputs = len(y_pred[0])

		# Gradient on values
		self.dinputs = -2 * (y_true - y_pred) / outputs
		# Normalize the gradient
		self.dinputs = self.dinputs / samples

# Mean Absolute Error
class Loss_MeanAbsoluteError(Loss):

	# Forward Pass
	def forward(self, y_pred, y_true):

		# Calculate loss
		sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

		# Return the loss
		return sample_losses

	# Backward Pass
	def backward(self, y_pred, y_true):

		# Number of samples
		samples = len(y_pred)
		# Number of outputs per sample
		outputs = len(y_pred[0])

		# Gradient on values
		self.dinputs = -np.sign(y_true - y_pred) / outputs
		# Normalize the gradient
		self.dinputs = self.dinputs / samples