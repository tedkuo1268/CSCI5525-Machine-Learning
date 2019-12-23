import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix

def read_data(path):
    data = np.genfromtxt(path, delimiter=',')
    return data
    
def partition(data):
    np.random.shuffle(data)
    X = data[:, 0:2]
    y = data[:, 2:3]
    data_size = len(y)
    train_size = int(0.8*data_size)

    X_for_train = X[:train_size]
    y_for_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    X_partitioned = {}
    y_partitioned = {}
    for i in range(10):
        X_partitioned[i] = X_for_train[i*160 : (i + 1)*160]
        y_partitioned[i] = y_for_train[i*160 : (i + 1)*160]
    return X_partitioned, y_partitioned, X_test, y_test

def get_next_train_valid(X_partitioned, y_partitioned, itr):
    X_valid = X_partitioned[itr]
    y_valid = y_partitioned[itr]

    init = 0
    for j in range(10):
        if j!=itr:
            if not init:
                X_train = X_partitioned[j]
                y_train = y_partitioned[j]
                init = 1
            else:
                X_train = np.concatenate((X_train, X_partitioned[j]), axis=0)
                y_train = np.concatenate((y_train, y_partitioned[j]), axis=0)
    return X_train, y_train, X_valid, y_valid

def rbf_kernel(X1, X2, gamma):
    """
    Calculate and return the kernel matrix
    """
    m = len(X1)
    n = len(X2)
    if m==n:
        K = np.zeros((m, n)) # kernel matrix
        for i in range(m):
            for j in range(i + 1):
                K_ij = np.exp(-gamma * (np.linalg.norm(X1[i] - X2[j]))**2) # Gaussian kernel
                K[i][j] = K_ij
                K[j][i] = K_ij
        return K
    else:
        return [[np.exp(-gamma * (np.linalg.norm(X1[i] - X2[j]))**2) for j in range(n)] for i in range(m)]

def rbf_svm_train(X_train, y_train, C, gamma):
    """
    Train the model and return support vectors and bias for prediction function to predict labels
    """
    data_size, feature_size = X_train.shape
    
    # Find the matrices to do quadratic programming optimization to solve for dual SVM problem
    P = matrix(np.outer(y_train, y_train.T) * rbf_kernel(X_train, X_train, gamma), tc='d')
    q = matrix(-np.ones((data_size, 1)), tc='d')
    G = matrix(np.concatenate((-np.identity(data_size), np.identity(data_size)), axis=0), tc='d')
    h = matrix(np.concatenate((np.zeros((data_size, 1)), C*np.ones((data_size, 1))), axis=0), tc='d')
    A = matrix(y_train.T, tc='d')
    b = matrix(0.0)
    sol = solvers.qp(P, q, G, h, A, b)
    lambdas = sol['x']
    lambdas_max = max(lambdas)

    # Find support vectors
    X_sv, y_sv, lambda_sv = [], [], []
    for i in range(len(lambdas)):
        lambda_i = lambdas[i]
        if lambda_i>10**-6:
            lambda_sv.append(lambda_i)
            X_sv.append(X_train[i])
            y_sv.append(y_train[i])
    print(len(y_sv))
    lambda_sv = np.array(lambda_sv).reshape(-1, 1)
    y_sv = np.array(y_sv).reshape(-1, 1)
    X_sv = np.array(X_sv)

    count = 0
    for i in range(len(lambda_sv)):
        lambda_i = lambda_sv[i]
        # Find unbounded support vectors to calculate b
        if (lambda_i>0.01*lambdas_max) and (lambda_i<0.999*lambdas_max): # Find the lambda which is not close to the margin, which will give more accurate b value
            count += 1
            X_free_sv = X_sv[i]
            y_free_sv = y_sv[i]
            bias = y_free_sv - np.sum(lambda_sv*y_sv*rbf_kernel(X_sv, X_sv, gamma), axis=0)[i]
            print(bias)
            break

    # If unbounded support vectors was not found from previous range, use the minimum support vector to calculate b
    if count==0:
        min_lambda_index = np.argmin(lambda_sv)
        X_free_sv = X_sv[min_lambda_index]
        y_free_sv = y_sv[min_lambda_index]
        bias = y_free_sv - np.sum(lambda_sv*y_sv*rbf_kernel(X_sv, X_sv, gamma), axis=0)[min_lambda_index]

    return lambda_sv, X_sv, y_sv, bias

def rbf_svm_predict(X, X_sv, y_sv, lambda_sv, bias, gamma):
    data_size = X.shape[0]
    sv_size = X_sv.shape[0]
    
    # predict label
    predictor = np.sum(lambda_sv*y_sv*rbf_kernel(X_sv, X, gamma), axis=0)
    y_predict_class = -np.ones_like(predictor)
    y_predict_class[(predictor + bias)>0] = 1

    return y_predict_class

def accuracy(X_partitioned, y_partitioned, X_test, y_test, C, gamma):
    # Create a list to store different C and accuracy corresponds to it
    # First item is C and the second one is the accuracy
    accuracy_train = [[0, 0] for i in range(len(C))] 
    accuracy_valid = [[0, 0] for i in range(len(C))]
    accuracy_test = [[0, 0] for i in range(len(C))]

    for i in range(len(C)):
        accuracy_train[i][0] = C[i]
        accuracy_valid[i][0] = C[i]
        accuracy_test[i][0] = C[i]
        accuracy_train[i][1] = 0
        accuracy_valid[i][1] = 0 
        accuracy_test[i][1] = 0 

        # Loop through 10 times to do cross-validation
        for itr in range(10):
            X_train, y_train, X_valid, y_valid = get_next_train_valid(X_partitioned, y_partitioned, itr)
            lambda_sv, X_sv, y_sv, bias = rbf_svm_train(X_train, y_train, C[i], gamma)
            
            
            # Compute the train accuracy in each iteration
            total_errors_train = 0
            y_train_predict = rbf_svm_predict(X_train, X_sv, y_sv, lambda_sv, bias, gamma)
            for k in range(len(y_train)):
                total_errors_train += y_train_predict[k]==y_train[k][0] # If the predicted label is different form the data label, error += 1
            accuracy_train[i][1] += (total_errors_train/len(y_train))
            
            # Compute the validation accuracy in each iteration
            total_errors_valid = 0
            y_valid_predict = rbf_svm_predict(X_valid, X_sv, y_sv, lambda_sv, bias, gamma)
            print(y_valid_predict)
            for k in range(len(y_valid)):
                total_errors_valid += y_valid_predict[k]==y_valid[k][0] # If the predicted label is different form the data label, error += 1
            accuracy_valid[i][1] += (total_errors_valid/len(y_valid))

            # Test performance
            total_errors_test = 0
            y_test_predict = rbf_svm_predict(X_test, X_sv, y_sv, lambda_sv, bias, gamma)
            for k in range(len(y_test)):
                total_errors_test += y_test_predict[k]==y_test[k][0] # If the predicted label is different form the data label, error += 1
            accuracy_test[i][1] += (total_errors_test/len(y_test))

            print(C[i], itr)
        
        # Compute the average accuracy of training, validation, and test
        accuracy_train[i][1] /= 10
        accuracy_valid[i][1] /= 10
        accuracy_test[i][1] /= 10

    return accuracy_train, accuracy_valid, accuracy_test

if __name__ == '__main__':
    path = "hw2data.csv"
    data = read_data(path)
    X_partitioned, y_partitioned, X_test, y_test = partition(data)
    C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gamma = 1
    accuracy_train, accuracy_valid, accuracy_test = accuracy(X_partitioned, y_partitioned, X_test, y_test, C, gamma)

    # Save the output
    pd.DataFrame(accuracy_train, columns=['C', 'accuracy']).to_csv('kernel_svm_train.csv', index=False, header=True)
    pd.DataFrame(accuracy_valid, columns=['C', 'accuracy']).to_csv('kernel_svm_valid.csv', index=False, header=True)
    pd.DataFrame(accuracy_test, columns=['C', 'accuracy']).to_csv('kernel_svm_test.csv', index=False, header=True)

    plt.figure()   
    # accuracy plot for train set
    accuracy_train_list = [accuracy_train[i][1] for i in range(len(C))]
    plt.plot(C, accuracy_train_list, label='train')
    # accuracy plot for validation set
    accuracy_valid_list = [accuracy_valid[i][1] for i in range(len(C))]
    plt.plot(C, accuracy_valid_list, label='validation')
    # accuracy plot for test set
    accuracy_test_list = [accuracy_test[i][1] for i in range(len(C))] 
    plt.plot(C, accuracy_test_list,label='test')
    plt.xscale('log')
    plt.title('RBF Kernel SVM Accuracy')
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.savefig('rbf_kernel_svm_accuracy.png')
