import numpy as np
import pickle
import copy

# Model Class
class Model:

	def __init__(self):
		# List of layer objects
		self.layers = []
		# Set if model is softmax classifier for faster backprop
		self.softmax_classifier_output = None

	# Add layers to the model
	def add(self, layer):
		self.layers.append(layer)

	# Set loss and optimizer
	def set(self, *, loss=None, optimizer=None, accuracy=None):
		self.loss = loss
		self.optimizer = optimizer
		self.accuracy = accuracy

	# Finalize the model
	def finalize(self):

		# Create input layer
		self.input_layer = Layer_Input()

		# Count the layers
		layer_count = len(self.layers)

		# List of trainable layers
		self.trainable_layers = []

		# Iterate the layers
		for i in range(layer_count):
			# Connect first layer to Input Layer
			if i == 0:
				self.layers[i].prev = self.input_layer
				self.layers[i].next = self.layers[i+1]

			# All layers except first and last
			elif i < layer_count - 1:
				self.layers[i].prev = self.layers[i-1]
				self.layers[i].next = self.layers[i+1]

			# Connect last layer to loss
			else:
				self.layers[i].prev = self.layers[i-1]
				self.layers[i].next = self.loss
				self.output_layer_activation = self.layers[i]

			# Add layers with weights to trainable layers list
			if hasattr(self.layers[i], 'weights'):
				self.trainable_layers.append(self.layers[i])

		# Update loss object with trainable layers
		if self.loss is not None:
			self.loss.remember_trainable_layers(self.trainable_layers)

		# If model is softmax classifier, combined activation and loss 
		# is faster in backpropagation
		if isinstance(self.layers[-1], Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
			self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

	# Training function
	def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):

		# Initialize accuracy object
		self.accuracy.init(y)

		# Default value if batch size not being set
		train_steps = 1

		# Set default steps for validation data
		if validation_data is not None:
			validation_steps = 1

			# For better readability
			X_val, y_val = validation_data

		# Calculate number of steps per epoch
		if batch_size is not None:
			train_steps = len(X) // batch_size
			# Add one if batch size doesn't evenly divide data size
			if train_steps * batch_size < len(X):
				train_steps += 1

			if validation_data is not None:
				validation_steps = len(X_val) // batch_size
				# Add one if batch size doesn't evenly divide data size
				if validation_steps * batch_size < len(X_val):
					validation_steps += 1

		# Main training loop
		for epoch in range(1, epochs+1):

			# Print epoch number
			print(f'epoch: {epoch}')

			# Reset accumulated values in loss and accuracy
			self.loss.new_pass()
			self.accuracy.new_pass()

			for step in range(train_steps):

				# If batch size not set, train in one step
				if batch_size is None:
					batch_X = X
					batch_y = y

				# Otherwise slice a batch
				else:
					batch_X = X[step*batch_size:(step+1)*batch_size]
					batch_y = y[step*batch_size:(step+1)*batch_size]
			
				# Forward Pass
				output = self.forward(batch_X, training=True)

				# Calculate loss
				data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
				loss = data_loss + regularization_loss

				# Get predictions and calculate accuracy
				predictions = self.output_layer_activation.predictions(output)
				accuracy = self.accuracy.calculate(predictions, batch_y)

				# Backward Pass
				self.backward(output, batch_y)

				# Update parameters
				self.optimizer.pre_update_params()
				for layer in self.trainable_layers:
					self.optimizer.update_params(layer)
				self.optimizer.post_update_params()

				# Print a summary
				if not step % print_every or step == train_steps - 1:
					print(f'step: {step}, ' +
						f'acc: {accuracy:.3f}, ' + 
						f'loss: {loss:.3f}, ' +
						f'data_loss: {data_loss:.3f}, ' +
						f'reg_loss: {regularization_loss:.3f}, ' +
						f'lr: {self.optimizer.current_learning_rate}')

			# Get and print epoch loss and accuracy
			epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
			epoch_loss = epoch_data_loss + epoch_regularization_loss
			epoch_accuracy = self.accuracy.calculate_accumulated()

			print(f'training, ' + 
				f'acc: {epoch_accuracy:.3f}, ' + 
				f'loss: {epoch_loss:.3f}, ' +
				f'data_loss: {epoch_data_loss:.3f}, ' +
				f'reg_loss: {epoch_regularization_loss:.3f}, ' +
				f'lr: {self.optimizer.current_learning_rate}')

			# If there is validation data
			if validation_data is not None:
				self.evaluate(*validation_data, batch_size=batch_size)				

	# Evaluates the model using passed-in dataset
	def evaluate(self, X_val, y_val, *, batch_size=None):
		# Default value if batch size not set
		validation_steps = 1

		# Calculate number of steps
		if batch_size is not None:
			validation_steps = len(X_val) // batch_size

			# Add one step if batch size doen't evenly divide data
			if validation_steps * batch_size < len(X_val):
				validation_steps += 1

		# Reset accumulated values in loss and accuracy
		self.loss.new_pass()
		self.accuracy.new_pass()

		# Iterate over steps
		for step in range(validation_steps):

			# If batch size not set, evaluate in one step
			if batch_size is None:
				batch_X = X_val
				batch_y = y_val

			# Otherwise slice a batch
			else:
				batch_X = X_val[step*batch_size:(step+1)*batch_size]
				batch_y = y_val[step*batch_size:(step+1)*batch_size]

			# Forward pass
			output = self.forward(batch_X, training=False)

			# Calculate loss
			self.loss.calculate(output, batch_y)

			# Get predictions and accuracy
			predictions = self.output_layer_activation.predictions(output)
			accuracy = self.accuracy.calculate(predictions, batch_y)

		# Get accumulated validation loss and accuracy
		validation_loss = self.loss.calculate_accumulated()
		validation_accuracy = self.accuracy.calculate_accumulated()

		# Print a summary
		print(f'validation, ' + 
			f'acc: {validation_accuracy:.3f}, ' + 
			f'loss: {validation_loss:.3f}')

	# Predicts on the samples
	def predict(self, X, *, batch_size=None):
		# Default value if batch size is not being set
		prediction_steps = 1

		# Calculate number of steps
		if batch_size is not None:
			prediction_steps = len(X) // batch_size
			# Add step if batch size doesn't evenly divide data size
			if prediction_steps * batch_size < len(X):
				prediction_steps += 1

		# Model outputs
		output = []

		# Iterate over steps
		for step in range(prediction_steps):

			# If batch size not set, predict using all data
			if batch_size is None:
				batch_X = X

			# Otherwise slice a batch
			else:
				batch_X = X[step*batch_size:(step+1)*batch_size]

			# Perform the forward pass
			batch_output = self.forward(batch_X, training=False)

			# Append batch prediction to output list
			output.append(batch_output)

		# Stack and return results
		return np.vstack(output)

	# Forward Pass
	def forward(self, X, training):

		# Put data through Input Layer
		self.input_layer.forward(X, training)

		# Call forward method of each layer
		for layer in self.layers:
			layer.forward(layer.prev.output, training)

		# Return output of last layer
		return layer.output

	# Backward Pass
	def backward(self, output, y):

		# If softmax classifier
		if self.softmax_classifier_output is not None:
			# Call backward on combined activation/loss
			self.softmax_classifier_output.backward(output, y)

			# Set dinputs on last layer
			self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

			# Iterate over rest of layers
			for layer in reversed(self.layers[:-1]):
				layer.backward(layer.next.dinputs)
			return

		# Set dinputs on loss object
		self.loss.backward(output, y)

		# Call backward on all layers in reverse
		for layer in reversed(self.layers):
			layer.backward(layer.next.dinputs)

	# Retrieves and returns parameters of trainable layers
	def get_parameters(self):

		# Create a list for parameters
		parameters = []

		# Iterate trainable layers and get their parameters
		for layer in self.trainable_layers:
			parameters.append(layer.get_parameters())

		# Return a list
		return parameters

	# Updates the model with new parameters
	def set_parameters(self, parameters):
		# Iterate and update each layer
		for parameter_set, layer in zip(parameters, self.trainable_layers):
			layer.set_parameters(*parameter_set)

	# Save parameters to file
	def save_parameters(self, path):
		# Open file in binary-write mode and save parameters
		with open(path, 'wb') as f:
			pickle.dump(self.get_parameters(), f)

	# Loads parameters and updates model instance
	def load_parameters(self, path):

		# Open file in binary-read mode, 
		# Load weights and update layers
		with open(path, 'rb') as f:
			self.set_parameters(pickle.load(f))

	# Save the model
	def save(self, path):

		# Make a deep copy of current model instance
		model = copy.deepcopy(self)

		# Reset accumulated values in loss and accuracy objects
		model.loss.new_pass()
		model.accuracy.new_pass()

		# Remove data from Input layer and gradients from loss object
		model.input_layer.__dict__.pop('output', None)
		model.loss.__dict__.pop('dinputs', None)

		# Remove inputs, output and dinputs for each layer
		for layer in model.layers:
			for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
				layer.__dict__.pop(property, None)

		# Open a file in binary-write mode and save the model
		with open(path, 'wb') as f:
			pickle.dump(model, f)

	# Loads and returns a model
	@staticmethod
	def load(path):

		# Open a file in binary-read mode, load a model
		with open(path, 'rb') as f:
			model = pickle.load(f)

		return model