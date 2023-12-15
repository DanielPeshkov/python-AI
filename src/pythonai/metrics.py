import numpy as np

# Common Accuracy class
class Accuracy:

	# Calculate Accuracy
	def calculate(self, predictions, y):

		# Get comparison results
		comparisons = self.compare(predictions, y)

		# Calculate an accuracy
		accuracy = np.mean(comparisons)

		# Add accumulated sum of matching values and sample count
		self.accumulated_sum += np.sum(comparisons)
		self.accumulated_count += len(comparisons)

		return accuracy

	# Calculates accumulated accuracy
	def calculate_accumulated(self):

		# Calculate accuracy
		accuracy = self.accumulated_sum / self.accumulated_count

		return accuracy

	# Reset variables for accumulated accuracy
	def new_pass(self):
		self.accumulated_sum = 0
		self.accumulated_count = 0

# Accuracy calculation for regression
class Regression_Accuracy(Accuracy):

	def __init__(self):
		# Precision property
		self.precision = None

	# Calculates precision value based on ground truth
	def init(self, y, reinit=False):
		if self.precision is None or reinit:
			self.precision = np.std(y) / 250

	# Compares predictions to ground truth
	def compare(self, predictions, y):
		return np.absolute(predictions - y) < self.precision

# Accuracy calculation for classification
class Categorical_Accuracy(Accuracy):

	def __init__(self, *, binary=False):
		# Binary mode
		self.binary = binary

	# No initialization needed
	def init(self, y):
		pass

	# Compares predictions to ground truth
	def compare(self, predictions, y):
		if not self.binary and len(y.shape) == 2:
			y = np.argmax(y, axis=1)
		return predictions == y