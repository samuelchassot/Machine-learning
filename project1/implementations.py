import numpy as np


def standardize(x):
    mean = np.mean(x)
    std = np.std(x)

    return (x-mean)/std, mean, std

def get_batches(y, tx, num_batches):
    seed = np.random.randint(0,1000000)
    np.random.seed(seed)
    np.random.shuffle(tx)
    np.random.shuffle(y)
    for i in range(num_batches):
        end_indx = min(i+1, len(y))
        if i != end_indx:
            yield y[i: end_indx], tx[i: end_indx]

def remove_wrong_columns(tx):
    for c in np.flip(np.where(np.any(tx == -999.0, axis = 0))):
        tx = np.delete(tx, c, axis = 1)
    return tx
    

def mse(e):
    return 1/2*np.mean(e**2)

def mae(e):
    return 1/2*np.mean(np.abs(e))

def compute_loss(f_loss, y, tx, w):
    return f_loss(y - tx@w)

def compute_rmse_loss(y, tx, w):
    return np.sqrt(2*compute_loss(mse, y, tx, w))

def compute_gradient(y, tx, w):
    e = y - tx@w
    gradient = -1/len(e) * tx.T@e
    
    return gradient, e

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient, error = compute_gradient(y, tx, w)
        loss = mse(error)
        w = w - gamma * gradient

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for y_batch, tx_batch in get_batches(y, tx, max_iters):
        gradient, error = compute_gradient(y_batch, tx_batch, w)
        loss = mse(error)
        w = w - gamma * gradient

    return w, loss

def least_squares(y, tx):
    w = np.linalg.inv(tx.T @ tx) @ (tx.T @ y)
    return w, compute_loss(mse, y, tx, w)

def ridge_regression(y, tx, lambda_):
    w = np.linalg.inv(tx.T @ tx +
            lambda_*2*len(y)*np.eye(tx.shape[1]) @ (tx.T @ y))
    return w, compute_loss(y , tx, w)

#Functions for logistic regression
def sigmoid(t):
    """apply sigmoid function on t."""
    t_exp = np.exp(t)
    return t_exp /(1 + t_exp)

def calculate_loss_sigmoid(y, tx, w):
    """compute the cost by negative log likelihood."""
    sigm_tx_w = sigmoid(tx @ w)
    return - np.sum(y.T @ np.log(sigm_tx_w) \
                    + (1 - y).T @ np.log(1 - sigm_tx_w))

def calculate_gradient_sigmoid(y, tx, w):
    """compute the gradient of loss."""
    return tx.T @ (sigmoid(tx@w) - y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the updated w and loss.
    """
    loss = calculate_loss_sigmoid(y, tx, w)
    gradient = calculate_gradient_sigmoid(y, tx, w)
    w -= gamma * gradient
    return w, loss

def reg_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient"""
    loss = calculate_loss_sigmoid(y, tx, w) + lambda_* w.T @ w
    gradient = calculate_gradient_sigmoid(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def learning_by_reg_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the regularized logistic regression.
    Return the updated w and loss.
    """
    loss, gradient = reg_logistic_regression(y, tx, w, lambda_)
    w -= gamma * gradient
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    threshold = 1e-8
    previous_loss = 0
    w = initial_w

    for iter in range(max_iters):
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        # converge criterion
        if np.abs(loss - previous_loss) < threshold:
            break

        previous_loss = loss

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w,
        max_iters, gamma):
    threshold = 1e-8
    previous_loss = 0
    w = initial_w

    for iter in range(max_iters):
        w, loss = learning_by_reg_gradient(y, tx, w, gamma, lambda_)
        # converge criterion
        if np.abs(loss - previous_loss) < threshold:
            break

        previous_loss = loss

    return w, loss


################################################################
def expand_features_polynomial(x, degree):
    result = np.zeros(x.shape)
    for i in range(0, degree):
        result = np.hstack((result, x**i))
    return result
