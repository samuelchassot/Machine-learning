import numpy as np
import matplotlib.pyplot as plt

THRESHOLD = 1e-8

def get_batches(y, tx, num_batches):
    """ Generate a minibatch iterator for a dataset, with batch size = 1. """
    np.random.seed(np.random.randint(0,1000000))

    indices = list(range(len(y)))
    np.random.shuffle(indices)

    tx = tx[indices]
    y = y[indices]
    
    for i in range(num_batches):
        end_indx = min(i+1, len(y))
        if i != end_indx:
            yield y[i: end_indx], tx[i: end_indx]
    
def mse(e):
    """ Given an error, computer the mean square of it"""
    return 1/2*np.mean(e**2)

def mae(e):
    """ Given an error, computer the mean absolute of it"""
    return 1/2*np.mean(np.abs(e))

def compute_loss(f_loss, y, tx, w):
    """ Compute the loss, given a certain loss function, 'f_loss'. """
    return f_loss(y - tx@w)

def compute_rmse_loss(y, tx, w):
    """ Compute the root mean square error. """
    return np.sqrt(2*compute_loss(mse, y, tx, w))

def sigmoid(t):
    """ Apply sigmoid function on t. """
    t_exp = np.exp(t)

    return t_exp/(1 + t_exp)

def compute_loss_sigmoid(y, tx, w):
    """ Compute the cost by negative log likelihood. """
    sigm_tx_w = sigmoid(tx@w)

    return -np.sum(y.T @ np.log(sigm_tx_w) + (1 - y).T @ np.log(1 - sigm_tx_w))

def compute_gradient(y, tx, w):
    """ Returns the gradient and the error for least squares """
    e = y - tx@w
    gradient = -1/len(e) * tx.T@e
    
    return gradient, e

def compute_gradient_sigmoid(y, tx, w):
    """ Compute the gradient of the loss (see 'compute_loss_sigmoid') """
    return tx.T @ (sigmoid(tx@w) - y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """ Returns the weights and the loss after one step of gradient descent """
    loss = compute_loss_sigmoid(y, tx, w)
    gradient = compute_gradient_sigmoid(y, tx, w)
    w -= gamma * gradient

    return w, loss

def loss_grad_reg_logistic_regression(y, tx, w, lambda_):
    """ Returns the loss and the gradient for regularized logistic regression """
    loss = compute_loss_sigmoid(y, tx, w) + lambda_ * w.T @ w
    gradient = compute_gradient_sigmoid(y, tx, w) + 2 * lambda_ * w

    return loss, gradient

def learning_by_reg_gradient(y, tx, w, gamma, lambda_):
    """ 
    Returns the weights and the loss after one step of gradient descent
    for regularized logistic regression.
    """
    loss, gradient = loss_grad_reg_logistic_regression(y, tx, w, lambda_)
    w -= gamma * gradient

    return w, loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent """
    w = initial_w
    
    for n_iter in range(max_iters):
        gradient, error = compute_gradient(y, tx, w)
        loss = mse(error)
        w = w - gamma * gradient

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using stochastic gradient descent """
    w = initial_w

    for y_batch, tx_batch in get_batches(y, tx, max_iters):
        gradient, error = compute_gradient(y_batch, tx_batch, w)
        loss = mse(error)
        w = w - gamma * gradient

    return w, loss

def least_squares(y, tx):
    """ Least squares regression using normal equations """
    w = np.linalg.inv(tx.T @ tx) @ (tx.T @ y)
    return w, compute_loss(mse, y, tx, w)

def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations """
    w = np.linalg.inv(tx.T @ tx + lambda_*2*len(y)*np.eye(tx.shape[1])) @ (tx.T @ y)
    return w, compute_loss(mse, y , tx, w)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent """
    previous_loss = 0
    w = initial_w

    for _ in range(max_iters):
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        if np.abs(loss - previous_loss) < THRESHOLD:
            break
        previous_loss = loss

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent """
    previous_loss = 0
    w = initial_w

    for i in range(max_iters):
        w, loss = learning_by_reg_gradient(y, tx, w, gamma, lambda_)
        if i%1000 == 0:
            print("At iteration {i}, loss = {l}".format(i=i, l=loss))
        if np.abs(loss - previous_loss) < THRESHOLD:
            break

        previous_loss = loss

    return w, loss


#Helpers to improve data
def standardize(x):
    """ Standardize the original data set. """
    mean = np.mean(x)
    std = np.std(x)

    return (x-mean)/std, mean, std

def standardize_train_and_test(tX, tX_test):
    """
    Standardize the training data set and 
    standardize the test set with the mean and std from the training set
    """
    tX_stdzed, tX_mean, tX_std = standardize(tX)
    tX_test_stdzed = (tX_test-tX_mean)/tX_std

    return tX_stdzed, tX_test_stdzed

def remove_wrong_columns(tx):
    """ Remove the columns of 'tx' containing '-999.0' """
    for c in np.flip(np.where(np.any(tx == -999.0, axis = 0))):
        tx = np.delete(tx, c, axis = 1)
    
    return tx

def expand_features_polynomial(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    return np.vander(x, degree+1, True)


#Cross-validation

#Cross-validation helpers
def build_k_indices(y, k_fold, seed):
    """ Build k indices for k-fold. """
    np.random.seed(seed)

    num_row = len(y)
    interval = int(num_row/k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    
    return np.array(k_indices)

def divide_train_test(y, tx, k_indices, k):
    """ Divide the samples into a training set and a testing set depending on k """
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)
    test_tx = tx[k_indices[k]]
    test_y = y[k_indices[k]]
    train_tx = tx[tr_indice]
    train_y = y[tr_indice]
    
    return test_tx, test_y, train_tx, train_y

def map_label_01(y):
    """ Map the labels from -1/1 to 0/1 """
    y_logistic = []
    for elem in y:
        if elem == -1:
            y_logistic.append(0)
        else:
            y_logistic.append(1)
            
    return np.asarray(y_logistic)

#Cross-validation function
def cross_validation_least_squares_GD(y, tx, initial_w, max_iters, gammas, k_fold, seed):
    """Does cross-validation to find the best gamma to use with least_squares_GD"""
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    mse_tr = []
    mse_te = []
    
    weights = initial_w
    
    for gamma in gammas:
        tr_tmp = []
        te_tmp = []
        for k in range(k_fold):
            # divide the data into training set and testing set depending on k
            test_tx, test_y, train_tx, train_y = divide_train_test(y, tx, k_indices, k)
            
            #Train the set and computes the losses
            weights, loss_tr = least_squares_GD(train_y, train_tx, initial_w, max_iters, gamma)
            loss_te = compute_loss(mse, test_y, test_tx, weights)
            
            tr_tmp.append(loss_tr)
            te_tmp.append(loss_te)
        mse_tr.append(np.mean(tr_tmp))
        mse_te.append(np.mean(te_tmp))

    gamma = gammas[np.argmin(mse_te)]
    weights_final, loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)
        
    return mse_tr, mse_te, gamma, weights_final, loss

def ridge_cross_validation(y, x, k_indices, k, lambda_, degree):
    """ 
    Does cross-validation to find de loss depending on 'lambda_' and 'degree'
    for ridge regression 
    """
    train_y, train_x, test_y, test_x = np.array([]), np.array([]), np.array([]), np.array([])
    
    for k_ in range(len(k_indices)):
        temp_y = y.take(k_indices[k_])
        temp_x = x.take(k_indices[k_])
        
        if k_ != k:
            train_y = np.concatenate((train_y, temp_y))
            train_x = np.concatenate((train_x, temp_x))
        else:
            test_y = np.concatenate((test_y, temp_y))
            test_x = np.concatenate((test_x, temp_x))
    
    train_poly = expand_features_polynomial(train_x, degree)
    test_poly = expand_features_polynomial(test_x, degree)
    
    w, loss_tr = ridge_regression(train_y, train_poly, lambda_)
    loss_te = compute_loss(mse, test_y, test_poly, w)
    
    return loss_tr, loss_te

def cross_validation_logistic_regression(y, tx, initial_w, max_iters, gammas, k_fold, seed):
    """Does cross-validation to find the best gamma to use with logistic regression"""
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    loss_sigmoid_tr = []
    loss_sigmoid_te = []
    
    weights = initial_w
    
    for gamma in gammas:
        tr_tmp = []
        te_tmp = []
        for k in range(k_fold):
            # divide the data into training set and testing set depending on k
            test_tx, test_y, train_tx, train_y = divide_train_test(y, tx, k_indices, k)
            
            #Train the set and computes the losses
            weights, loss_tr = logistic_regression(train_y, train_tx, initial_w, max_iters, gamma)
            loss_te = compute_loss_sigmoid(test_y, test_tx, weights)
            
            tr_tmp.append(loss_tr)
            te_tmp.append(loss_te)
        loss_sigmoid_tr.append(np.mean(tr_tmp))
        loss_sigmoid_te.append(np.mean(te_tmp))
        
    gamma = gammas[np.argmin(loss_sigmoid_te)]
    weights_final, loss_sigmoid = logistic_regression(y, tx, initial_w, max_iters, gamma)
        
    return loss_sigmoid_tr, loss_sigmoid_te, gamma, weights_final, loss_sigmoid

def cross_validation_reg_log_regr(y, tx, w_initial, max_iters, gammas, lambdas_, k_fold, seed):
    """
    Does cross-validation to find the best gamma and lambda_ 
    to use with regularized logistic regression
    """
    k_indices = build_k_indices(y, k_fold, seed)
    tr_losses = np.zeros((len(gammas), len(lambdas_)))
    te_losses = np.zeros((len(gammas), len(lambdas_)))

    for gamma_index,gamma in zip(range(len(gammas)), gammas):
        for lambda_index, lambda_ in zip(range(len(lambdas_)),lambdas_):
            tr_k_losses = np.zeros((k_fold))
            te_k_losses = np.zeros((k_fold))

            for k in range(k_fold):
                te_tx_k, te_y_k, tr_tx_k, tr_y_k = divide_train_test(y, tx, k_indices, k)
                
                w_k, tr_loss_k = reg_logistic_regression(tr_y_k, tr_tx_k, lambda_, w_initial, max_iters, gamma)
                
                te_loss_k = compute_loss_sigmoid(te_y_k, te_tx_k, w_k)
                
                tr_k_losses[k] = tr_loss_k
                te_k_losses[k] = te_loss_k
                
            tr_loss = np.mean(tr_k_losses)
            te_loss = np.mean(te_k_losses)
            
            tr_losses[gamma_index][lambda_index] = tr_loss
            te_losses[gamma_index][lambda_index] = te_loss
            
            argmin = np.argmin(te_losses)
            gamma_idx = argmin//len(lambdas_)
            lambda_idx = argmin%len(lambdas_)
            gamma = gammas[gamma_idx]
            lambda_ = lambdas_[lambda_idx]

    return tr_losses, te_losses, gamma, lambda_
