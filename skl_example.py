from process import get_data
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

x_train, y_train, x_test, y_test = get_data()

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test )

model = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=2000)
model.fit(x_train, y_train)

train_accuracy = model.score(x_train, y_train)
test_accuracy = model.score(x_test, y_test)
print("train accuracy: " + str(train_accuracy) + " test accuracy: " + str(test_accuracy))