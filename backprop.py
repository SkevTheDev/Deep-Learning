import numpy as np
import matplotlib.pyplot as plt

def feed_forward(X, W1, b1, W2, b2):
	Z = 1/(1 + np.exp(-(X.dot(W1)+b1)))
	A = Z.dot(W2) + b2
	expA = np.exp(A)
	Y = expA / expA.sum(axis=1, keepdims=True)
	return Y, Z

def classification_rate(T, Y):
	n_correct = 0
	n_total = 0
	for i in range(len(T)):
		n_total += 1
		if T[i].all() == Y[i].all():
			n_correct += 1
	return float(n_correct) / n_total

def derivative_w2(Z, T, Y):
	# N, K = T.shape
	# M = Z.shape[1]

	#slow implementation
	# ret1 = np.zeros((M, K))
	# for n in range(N):
	# 	for m in range(M):
	# 		for k in range(K):
	# 			ret1[m,k] += (T[n,k] - Y[n,k]) * Z[n, m]
	# return ret1
	return Z.T.dot(T - Y)

def derivative_b2(T, Y):
	return (T - Y).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2):
	# N, D = X.shape
	# M, K = W2.shape

	#slow implementation
	# ret1 = np.zeros((D, M))
	# for n in range(N):
	# 	for k in range(K):
	# 		for m in range(M):
	# 			for d in range(D):
	# 				ret1[d,m] += (T[n,k] - Y[n,k]) * W2[m,k] * Z[n, m]*(1 - Z[n, m]) * X[n,d]
	# return ret1
	dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
	return X.T.dot(dZ)

def derivative_b1(T, Y, W2, Z):
	return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)

def error(T, Y):
	tot = T * np.log(Y)
	return tot.sum()

def main():
	number_of_samples = 500

	D = 2 # dimensionality of input
	M = 3 # hidden layer size
	K = 3 # number of classes

	X1 = np.random.randn(number_of_samples, 2) + np.array([0, -2])
	X2 = np.random.randn(number_of_samples, 2) + np.array([2, 2])
	X3 = np.random.randn(number_of_samples, 2) + np.array([-2, 2])
	X = np.vstack([X1, X2, X3])

	Y = np.array([0]*number_of_samples + [1]*number_of_samples + [2]*number_of_samples)
	N = len(Y)

	# one hot encoding for targets
	T = np.zeros((N,K))
	for i in range(N):
		T[i, Y[i]] = 1

	#randomly initialize weights
	W1 = np.random.randn(D, M)
	b1 = np.random.randn(M)
	W2 = np.random.randn(M, K)
	b2 = np.random.randn(K)

	learning_rate = 10e-7
	errors = []
	for epoch in range(100000):
		output, hidden = feed_forward(X, W1, b1, W2, b2)
		if epoch % 100 == 0:
			e = error(T, output)
			P = np.argmax(output, axis=1)
			r = classification_rate(Y, P)
			print("error: " + str(e) + " classification_rate: " + str(r))
			errors.append(e)

		W2 += learning_rate * derivative_w2(hidden, T, output)
		b2 += learning_rate * derivative_b2(T, output)
		W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
		b1 += learning_rate * derivative_b1(T, output, W2, hidden)

	plt.plot(errors)
	plt.show()

if __name__ == '__main__':
	main()