import numpy as np

def reLU(n):
	return np.maximum(0, n)

def relu_derivative(n):
	# return (n > 0).astype(float)
	return (n > 0)*1.0

inputs = np.array([
    [8, 7, 1],
    [2, 5, 4],
    [6, 6, 2],
    [1, 4, 0]
])
np.random.seed(42)
labels = np.array([[1], [0], [1], [0]])

weights = np.random.randn(3, 1) * 0.01

bias = np.zeros((1, 1))

epochs = 500
learning_rate = 0.001
for i in range(epochs):
	z = np.dot(inputs, weights) + bias
	y = reLU(z)

	error = y - labels
	adjustment = error * relu_derivative(z)
	dw = np.dot(inputs.T, adjustment)
	db = np.sum(adjustment, axis=0, keepdims=True)
	weights -= learning_rate * dw
	bias -= learning_rate * db

dummy_input = np.array([[1, 8, 1],])
ans = np.dot(dummy_input, weights) + bias
print(reLU(ans))
