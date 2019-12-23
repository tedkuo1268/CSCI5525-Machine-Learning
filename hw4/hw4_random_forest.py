import sklearn.tree as tree
import numpy as np
import matplotlib.pyplot as plt

def read_data(path):
    data = np.genfromtxt(path, delimiter=',')
    return data

def partition(data):
    """
    Partition the data to train and test set
    """
    #np.random.shuffle(data)
    X = data[:, 1:23]
    y = data[:, 0:1]
    data_size = len(y)
    train_size = 6000

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    return X_train, y_train, X_test, y_test

def bootstrap(sample_size, num_of_samples, X_train, y_train):
    """
    Subsample the train data set with replacement according to number of samples and sample size
    """
    subsamples_X = {}
    subsamples_y = {}

    for i in range(num_of_samples):
        subsample_index = np.random.randint(0, 6000, size=sample_size) # Generate random sample index
        subsample_X = []
        subsample_y = []

        for index in subsample_index:
            subsample_X.append(X_train[index]) 
            subsample_y.append(y_train[index]) 

        subsamples_X[i] = subsample_X
        subsamples_y[i] = subsample_y

    return subsamples_X, subsamples_y

def train(subsamples_X, subsamples_y, feature_size):
    """
    Train a decision tree with subsamples and the selected feature size
    """
    models = []

    for i in range(len(subsamples_X)):
        decision_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=2, max_features=feature_size)
        decision_tree.fit(subsamples_X[i], subsamples_y[i])
        models.append(decision_tree) # Append all trained decision trees in models

    return models

def random_forest_predict(models, X_test):
    """
    Combine all the decision trees to create random forest for voting the majority of labels
    """
    num_of_trees = len(models)
    y_predict = []

    for i in range(num_of_trees):
        y_predict.append(models[i].predict(X_test)) # Find the prediction for each decision tree

    vote = np.sum(y_predict, axis=0)
    final_predict = -np.ones_like(vote) # Initialize the prediction array with all -1
    final_predict[vote>=0] = 1 # If +1 votes is more than -1 vote, switch prediction to +1

    return final_predict

def accuracy(y_predict, y_test):
    """
    Return the accuracy of random forest prediction
    """
    test_size = len(y_test)

    accuracy = 0
    for i in range(test_size):
        if y_predict[i]==y_test[i]:
            accuracy += 1

    accuracy /= test_size
    #accuracy *= 100

    return accuracy

if __name__=='__main__':
    data = read_data('Mushroom.csv')
    X_train, y_train, X_test, y_test = partition(data)
    subsamples_X, subsamples_y = bootstrap(4000, 100, X_train, y_train)
    
    train_accuracy_random_feature = []
    test_accuracy_random_feature = []

    # Train decision trees with different feature sizes and combine to random forest
    for feature_size in [5, 10, 15, 20]:
        models_random_feature = train(subsamples_X, subsamples_y, feature_size) 
        y_predict_test = random_forest_predict(models_random_feature, X_test)
        y_predict_train = random_forest_predict(models_random_feature, X_train)
        test_accuracy_random_feature.append(accuracy(y_predict_test, y_test))
        train_accuracy_random_feature.append(accuracy(y_predict_train, y_train))
        print('feature size:', feature_size)
        print('train accuracy:', 100*accuracy(y_predict_train, y_train), '%')
        print('test accuracy:', 100*accuracy(y_predict_test, y_test), '%')
    
    plt.figure()
    plt.plot([5, 10, 15, 20], train_accuracy_random_feature, label='train accuracy')
    plt.plot([5, 10, 15, 20], test_accuracy_random_feature, label='test accuracy')
    plt.title('Accuracy for Different Random Feature Sizes')
    plt.xlabel('size')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.savefig('random_feature_size.png')    

    train_accuracy_num_of_trees = []
    test_accuracy_num_of_trees = []

    # Train decision trees with different number of trees and combine to random forest
    for num_of_tree in [10, 20, 40, 80, 100]:
        subsamples_X, subsamples_y = bootstrap(4000, num_of_tree, X_train, y_train)
        models_num_of_trees = train(subsamples_X, subsamples_y, 20)
        y_predict_train = random_forest_predict(models_num_of_trees, X_train)
        y_predict_test = random_forest_predict(models_num_of_trees, X_test)
        train_accuracy_num_of_trees.append(accuracy(y_predict_train, y_train))
        test_accuracy_num_of_trees.append(accuracy(y_predict_test, y_test))
        print('number of trees:', num_of_tree)
        print('train accuracy:', 100*accuracy(y_predict_train, y_train), '%')
        print('test accuracy:', 100*accuracy(y_predict_test, y_test), '%')
    
    plt.figure()
    plt.plot([10, 20, 40, 80, 100], train_accuracy_num_of_trees, label='train accuracy')
    plt.plot([10, 20, 40, 80, 100], test_accuracy_num_of_trees, label='test accuracy')
    plt.title('Accuracy for Different Number of Trees')
    plt.xlabel('number of trees')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.savefig('num_of_trees.png')
















