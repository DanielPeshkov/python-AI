import numpy as np

# SGD Optimizer
class Optimizer_SGD:

	# Initialize optimizer with default settings
	def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.momentum = momentum
		self.iterations = 0

	# Call before parameter updates
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

	# Update Parameters
	def update_params(self, layer):

		# If momentum is used
		if self.momentum:

			# initialize momentum array if it doesn't exist
			if not hasattr(layer, 'weight_momentums'):
				layer.weight_momentums = np.zeros_like(layer.weights)
				layer.bias_momentums = np.zeros_like(layer.biases)

			# Build weight updates with momentum
			weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
			layer.weight_momentums = weight_updates

			# Build bias updates with momentum
			bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
			layer.bias_momentums = bias_updates

		# Vanilla SGD without momentum
		else:
			weight_updates = -self.current_learning_rate * layer.dweights
			bias_updates = -self.current_learning_rate * layer.dbiases

		# Update weights and biases
		layer.weights += weight_updates
		layer.biases += bias_updates

	# Call after parameter updates
	def post_update_params(self):
		self.iterations += 1

# AdaGrad Optimizer
class Optimizer_Adagrad:

	# Initialize optimizer with default settings
	def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.epsilon = epsilon
		self.iterations = 0

	# Call before parameter updates
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

	# Update Parameters
	def update_params(self, layer):

		# initialize cache array if it doesn't exist
		if not hasattr(layer, 'weight_cache'):
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_cache = np.zeros_like(layer.biases)

		# Update Cache with squared gradients
		layer.weight_cache += layer.dweights**2
		layer.bias_cache += layer.dbiases**2

		# Update weights and biases with cache
		layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
		layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

	# Call after parameter updates
	def post_update_params(self):
		self.iterations += 1

# RMSprop Optimizer
class Optimizer_RMSprop:

	# Initialize optimizer with default settings
	def __init__(self, learning_rate=1e-3, decay=0., epsilon=1e-7, rho=0.9):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.epsilon = epsilon
		self.rho = rho
		self.iterations = 0

	# Call before parameter updates
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

	# Update Parameters
	def update_params(self, layer):

		# initialize cache array if it doesn't exist
		if not hasattr(layer, 'weight_cache'):
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_cache = np.zeros_like(layer.biases)

		# Update Cache with squared gradients
		layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
		layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

		# Update weights and biases with cache
		layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
		layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

	# Call after parameter updates
	def post_update_params(self):
		self.iterations += 1

# Adam Optimizer
class Optimizer_Adam:

	# Initialize optimizer with default settings
	def __init__(self, learning_rate=1e-3, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.epsilon = epsilon
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.iterations = 0

	# Call before parameter updates
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

	# Update Parameters
	def update_params(self, layer):

		# initialize cache array if it doesn't exist
		if not hasattr(layer, 'weight_cache'):
			layer.weight_momentums = np.zeros_like(layer.weights)
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_momentums = np.zeros_like(layer.biases)
			layer.bias_cache = np.zeros_like(layer.biases)

		# Update momentum with current gradients
		layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
		layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

		# Get corrected momentum
		weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
		bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

		# Update cache with squared current gradients
		layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
		layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

		# Get corrected cache
		weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
		bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

		# Update weights and biases with cache
		layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
		layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

	# Call after parameter updates
	def post_update_params(self):
		self.iterations += 1