from keras.src.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
X_train_slice = X_train[10000:15000, :, :]

print(X_train_slice.shape)