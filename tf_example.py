import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#create random training data 
n_class = 500

D = 2 # dimensionality of input
M = 3 # number of neurons in hidden layer
K = 3 # number of classes

X1 = np.random.randn(n_class, D) + np.array([0, -2])
X2 = np.random.randn(n_class, D) + np.array([2, 2])
X3 = np.random.randn(n_class, D) + np.array([-2, 2])
X = np.vstack([X1,X2,X3]).astype(np.float32)

Y = np.array([0]*n_class + [1]*n_class + [2]*n_class)

plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()
N=len(Y)

#convert Y into an indicator matrix for training
T = np.zeros((N,K))
for i in range(N):
	T[i, Y[i]] = 1

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

def feed_forward(X, W1, b1, W2, b2):
	Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
	return tf.matmul(Z, W2) + b2

tfX = tf.placeholder(tf.float32, [None, D])
tfY = tf.placeholder(tf.float32, [None, K])

W1 = init_weights([D, M])
b1 = init_weights([M])
W2 = init_weights([M, K])
b2 = init_weights([K])

py_x = feed_forward(tfX, W1, b1, W2, b2)

error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tfY, logits=py_x))

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(error)
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
	sess.run(train_op, feed_dict={tfX: X, tfY: T})
	pred = sess.run(predict_op, feed_dict={tfX: X, tfY: T})
	if i % 10 == 0:
		print("Accuracy: ", np.mean(Y == pred))