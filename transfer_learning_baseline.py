from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
# it's a good idea to flatten the array.


y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)


from sklearn.model_selection import train_test_split


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify = y_train)

from traff_sign_class_code import *

X_train_pp = preprocess(make_grey(X_train))
X_valid_pp = preprocess(make_grey(X_valid))



num_epochs = 200
batch_size = 128

valid_accuracies, train_accuracies = train_and_validate(num_epochs, batch_size, X_train_pp, y_train, X_valid_pp, y_valid)