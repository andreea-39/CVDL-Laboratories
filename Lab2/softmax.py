import numpy as np
from activations import softmax

class SoftmaxClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.W = None
        self.initialize()

    def initialize(self):
        # TODO your code here
        # initialize the weight matrix (remember the bias trick) with small random variables
        # you might find np.random.randn userful here *0.001
        self.W = np.random.randn(self.input_shape, self.num_classes) * 0.001

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = None
        # TODO your code here
        # 0. compute the dot product between the weight matrix and the input X
        # remember about the bias trick!
        # 1. apply the softmax function on the scores
        # 2, returned the normalized scores
        pred = X.dot(self.W)
        scores = softmax(pred, 1)
        return scores

    def predict(self, X: np.ndarray) -> int:
        label = None
        # TODO your code here
        # 0. compute the dot product between the weight matrix and the input X as the scores
        # 1. compute the prediction by taking the argmax of the class scores
        pred = X.dot(self.W)
        label = np.argmax(pred, axis=1)
        return label

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            **kwargs) -> dict:

        history = []

        bs = kwargs['bs'] if 'bs' in kwargs else 128
        reg_strength = kwargs['reg_strength'] if 'reg_strength' in kwargs else 1e3
        steps = kwargs['steps'] if 'steps' in kwargs else 100
        lr = kwargs['lr'] if 'lr' in kwargs else 1e-3
        print(bs, reg_strength, steps, lr)
        
        # run mini-batch gradient descent
        for iter in range(0, steps):
            # TODO your code here
            # sample a batch of images from the training set
            # you might find np.random.choice useful
            indices = np.random.choice(X_train.shape[0], bs)
            X_batch, y_batch = X_train[indices, :], y_train[indices]
            
            f = X_batch.dot(self.W)
            f -= np.matrix(np.max(f, axis=1)).T

            loss = np.sum(-f[np.arange(X_batch.shape[0]), y_batch] + np.log(np.sum(np.exp(f), axis=1))) / len(X_batch)
            loss += reg_strength * np.sum(np.square(self.W))
            CT = self.predict_proba(X_batch)
            CT[np.arange(X_batch.shape[0]), y_batch] -= 1
            dW = X_batch.T.dot(CT) / X_batch.shape[0] + 2 * reg_strength * self.W
            # end TODO your code here
            # compute the loss and dW
            # perform a parameter update
            self.W -= lr * dW
            # append the training loss, accuracy on the training set and accuracy on the test set to the history dict
            history.append(loss)

        return history


    def get_weights(self, img_shape):
        W = None
        # TODO your code here
        # 0. ignore the bias term
        # 1. reshape the weights to (*image_shape, num_classes)
        W = np.reshape(self.W[:-1, :], (*img_shape, self.num_classes))
        return W

    def load(self, path: str) -> bool:
        # TODO your code here
        # load the input shape, the number of classes and the weight matrix from a file
        self.W = np.load(path)
        return True

    def save(self, path: str) -> bool:
        # TODO your code here
        # save the input shape, the number of classes and the weight matrix to a file
        # you might find np.save useful for this
        # TODO your code here
        np.save(path, self.W)
        return True