import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix

def softmax_function(W, x, k):
    """
    Return value of softmax function
    """
    d, K = W.shape
    S = np.exp(np.dot(W.T[k], x))/np.sum(np.exp(np.dot(W.T, x)))

    return S

def loss_function(X, y, W):
    """
    Calculate the cross-entropy loss function
    """
    #X, y = data_train
    #X = data_train[:, 1:]
    #y = data_train[:, 0]

    loss = 0
    for i in range(X.shape[0]):
        loss += (np.log(np.sum(np.exp(np.dot(W.T, X[i])))) - np.dot(W.T[int(y[i])], X[i]))

    return loss

def cross_entropy_gradient(W, x, y_label, k):
    """
    Calculate the gradient of the cross-entropy loss function
    """
    if k==y_label:
        gradient = (-1 + softmax_function(W, x, k))*x
    else:
        gradient = softmax_function(W, x, k)*x

    return gradient

def mini_batch_gradient(W, X_mini_train, y_mini_train, k):
    """
    Return the average gradient of the given the mini batch
    """
    N, d = X_mini_train.shape
    gradient = np.zeros(d)
    for i in range(N):
        gradient = np.add(gradient, cross_entropy_gradient(W, X_mini_train[i], y_mini_train[i], k))

    gradient = gradient/N

    return gradient

def create_mini_batches(data_train, batch_size):
    """
    Split the data set into into batches according to the batch size
    """
    mini_batches = []
    np.random.shuffle(data_train)
    N = data_train.shape[0] // batch_size

    for i in range(N):
        mini_batch = data_train[i*batch_size : (i + 1)*batch_size, :]
        X_mini_train = mini_batch[:, 1:]
        Y_mini_train = mini_batch[:, 0]
        mini_batches.append((X_mini_train, Y_mini_train))
    # Additional batch consists of the remaining data
    if data_train.shape[0] % batch_size != 0:
        mini_batch = data_train[(i + 1) * batch_size:data_train.shape[0]]
        X_mini_train = mini_batch[:, 1:]
        Y_mini_train = mini_batch[:, 0]
        mini_batches.append((X_mini_train, Y_mini_train))

    return mini_batches

def multi_logistic_train(data_train, learning_rate, gd_itrs):
    """
    Train the model using mini-batch gradient descent and return the model weight
    """
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    d = X_train.shape[1]
    W = 0.01*np.random.rand(d, 10)

    loss_list = []
    num_of_batches = [0]
    loss = loss_function(X_train, y_train, W)
    loss_list.append(loss)
    print(loss)
    for itr in range(gd_itrs):
        print(itr)
        mini_batches = create_mini_batches(data_train, 32)
        batch_num = 0
        for mini_batch in mini_batches:
            batch_num += 1
            num_of_batches.append(batch_num)
            X_mini_train, y_mini_train = mini_batch
            for k in range(10):
                W.T[k] = np.add(W.T[k], -learning_rate*mini_batch_gradient(W, X_mini_train, y_mini_train, k)) # weight update with each mini batch
            loss = loss_function(X_train, y_train, W)
            loss_list.append(loss)
            print(loss)

    return W, loss_list, num_of_batches

def multi_logistic_predict(W, X, y):
    """
    Predict the label of the given data set with the trained weight
    """
    N = X.shape[0]
    confusion_matrix = np.zeros((10, 10), dtype=int) # Create 10-by-10 confusion matrix

    accuracy = 0
    for i in range(N):
        predicted_label = int(np.argmax(np.dot(W.T, X[i]))) # Select the label with the highest probability to be the predicted label
        if y[i]==predicted_label:
            accuracy += 1
            confusion_matrix[int(y[i])][predicted_label] += 1
        else:
            confusion_matrix[int(y[i])][predicted_label] += 1
    accuracy /= N

    return accuracy, confusion_matrix

if __name__=='__main__':
    data_train = np.genfromtxt('mnist_train.csv', delimiter=',')
    data_train = np.insert(data_train, 1, 1, axis=1)
    data_test = np.genfromtxt('mnist_test.csv', delimiter=',')
    data_test = np.insert(data_test, 1, 1, axis=1)
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]
    W, loss_list, num_of_batches = multi_logistic_train(data_train, 0.00001, 1)
    accuracy_train, confusion_matrix_train = multi_logistic_predict(W, X_train, y_train)
    accuracy_test, confusion_matrix_test = multi_logistic_predict(W, X_test, y_test)
    
    print('train accuracy:', accuracy_train)
    print('test accuracy:', accuracy_test)
    print('train data confusion matrix:')
    print(confusion_matrix_train)
    print('test data confusion matrix:')
    print(confusion_matrix_test)

    np.save('trained_weights.npy', W)

    plt.figure()   
    # loss function value plot
    plt.plot(num_of_batches, loss_list)
    plt.title('Loss Function Value')
    plt.xlabel('number of mini batches')
    plt.ylabel('Loss')
    plt.savefig('multiclass_loss.png')
    


