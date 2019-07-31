import numpy as np
import sklearn.datasets


def softmax(X):
    exps = np.exp(X)

    return exps / np.sum(exps, axis=1, keepdims=True)


def stable_softmax(X):
    exps = np.exp(X - np.max(X))

    return exps / np.sum(exps, axis=1, keepdims=True)


def negative_log_likelihood(X, y):
    m = y.shape[0]
    p = softmax(X)

    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m

    return loss


def calculate_loss(model, X, y):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)

    z2 = a1.dot(W2) + b2

    return negative_log_likelihood(z2, y)


def build_model(X, y, nn_hdim, nn_input_dim, nn_output_dim, lr=0.01, num_passes=3000, print_loss=False):
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    model = {}

    for i in range(0, num_passes):

        # forward
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2

        probs = softmax(z2)

        # bp
        delta3 = probs
        delta3[range(len(X)), y] -= 1

        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # update weights
        W1 += -lr * dW1
        b1 += -lr * db1

        W2 += -lr * dW2
        b2 += -lr * db2

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss and i % 1000 == 0:
            print('Loss after iteration %i: %f' % (i, calculate_loss(model, X, y)))

    return model


def generate_data():
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.2)

    return X, y


if __name__ == '__main__':
    X, y = generate_data()
    nn_input_dim = 2
    nn_output_dim = 2

    reg_lambda = 0.01
    lr = 0.01

    build_model(X, y, 2, nn_input_dim, nn_output_dim, lr=lr, print_loss=True)
