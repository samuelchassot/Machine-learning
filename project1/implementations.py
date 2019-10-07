#mean square error
def mse(e):
    return 1/2*np.mean(e**2)

#mean absolute error
def mae(e):
    return 1/2*np.mean(np.abs(e))

#compute the loss according to loss function
def compute_loss(f_loss, y, tx, w):
    return f_loss(y - tx@w)

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
    return NotImplementedError

def ridge_regression(y, tx, lambda_):
    return NotImplementedError

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return NotImplementedError

def reg_logistic_regression(y, tx, lambda_, initial_w,
        max_iters, gamma):
    return NotImplementedError
