import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_data

def y2_indicator(y, K):
	N = len(y)
	ind = np.zeros((N,K))
	for i in range(N):
		ind[i, y[i]] = 1
	return ind

Xtrain, Ytrain, Xtest, Ytest = get_data()
D = Xtrain.shape[1]
K = len(set(Ytrain) | set(Ytest))

# convert to indicator
Ytrain_ind = y2_indicator(Ytrain, K)
Ytest_ind = y2_indicator(Ytest, K)


W = np.random.randn(D, K)
b = np.zeros(K)


def softmax(a):
	expA = np.exp(a)
	return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W, b):
	return softmax(X.dot(W) + b)

def predict(P_Y_given_X):
	return np.argmax(P_Y_given_X, axis=1)

def classification_rate(Y, P):
	return np.mean(Y == P)

def cross_entropy(T, pY):
	return -np.mean(T*np.log(pY))




train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000):
	pYtrain = forward(Xtrain, W, b)
	pYtest = forward(Xtest, W, b)

	ctrain = cross_entropy(Ytrain_ind, pYtrain)
	ctest = cross_entropy(Ytest_ind, pYtest)
	train_costs.append(ctrain)
	test_costs.append(ctest)


	W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain_ind)
	b -= learning_rate*(pYtrain - Ytrain_ind).sum(axis=0)
	if i % 1000 == 0:
		print(i, ctrain, ctest)

print("Final train classification_rate: ", classification_rate(Ytrain, predict(pYtrain)))
print("Final test classification_rate: ", classification_rate(Ytest, predict(pYtest)))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')

plt.legend([legend1, legend2])
plt.show()
