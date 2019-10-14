import numpy as np

def standardize(x):
    mean = np.mean(x)
    std = np.std(x)

    return (x-mean)/std, mean, std

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
    for y_batch, tx_batch in batch_iter(y, tx, 1, max_iters, True):
        gradient, error = compute_gradient(y_batch, tx_batch, w)
        loss = mse(error)
        w = w - gamma * gradient

   return w, loss

def least_squares(y, tx):
    w = np.linalg.inv(tx.T @ tx) @ (tx.T @ y)
    return w, compute_loss(mse, y, tx, w)

def ridge_regression(y, tx, lambda_):
    w = np.linalg.inv(tx.T @ tx +
            lambda_*2*len(y)*np.eye(tx.shape[1]) @ (tx.T @ y)
    return w, compute_loss(y , tx, w)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return NotImplementedError

def reg_logistic_regression(y, tx, lambda_, initial_w,
        max_iters, gamma):
    return NotImplementedError
