import pandas as pd
import numpy as np
import math

def load_data(file_name, trainning_percentage = 0.9):
    """
    Loads the data from the file_name.
    Prints the dataset analysis.
    Divides it between the trainning set input / output and test set input / output based on the tranning percentage default at 90%.
    """
    
    credit_card_data = pd.read_csv(file_name)
    m, n = credit_card_data.shape
    positive_examples_full_count = sum(credit_card_data['Class'] == 0)
    negative_examples_full_count = sum(credit_card_data['Class'] == 1)
    print('There are ' + str(m) + ' examples and ' + str(positive_examples_full_count) +' are valid credit card transactions and '+str(negative_examples_full_count)+' are fraudulent ones.')
    m_trainning = math.ceil( m * trainning_percentage)
    print('The trainning set is '+ str(trainning_percentage * 100) +'% from the full set, so it has '+ str(m_trainning) +' trainning examples.')
    X_trainning_data = credit_card_data.iloc[0:m_trainning,0:-1]
    y_trainning_data = credit_card_data.iloc[0:m_trainning,-1]
    X_test_data = credit_card_data.iloc[m_trainning:,0:-1]
    y_test_data = credit_card_data.iloc[m_trainning:,-1]
    positive_examples_trainning_count = sum(y_trainning_data == 0)
    negative_examples_trainning_count = sum(y_trainning_data == 1) 
    print('The trainning set has '+ str(positive_examples_trainning_count) +' positive examples and '+ str(negative_examples_trainning_count) +' negative ones.')
    positive_examples_test_count = sum(y_test_data == 0)
    negative_examples_test_count = sum(y_test_data == 1)
    print('The test set has '+ str(positive_examples_test_count) +' positive examples and '+ str(negative_examples_test_count) +' negative ones.')
    X_trainning = np.transpose(X_trainning_data.values)
    y_trainning = np.transpose(np.reshape(y_trainning_data.values,[m_trainning,1]))
    X_test = np.transpose(X_test_data.values)
    y_test = np.transpose(np.reshape(y_test_data.values,[m-m_trainning,1]))
    return (X_trainning, y_trainning, X_test, y_test)

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return parameters

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }
    return A2, cache

def compute_cost(A, Y):
    m = Y.shape[1]
    cost = -(1/m) * np.sum(np.multiply(np.log(A),Y) + np.multiply(np.log(1-A),1-Y))
    cost = np.squeeze(cost)
    return cost

def two_layer_model(X, Y, parameters, learning_rate = 0.001, num_iterations = 2):
    grads = {}
    costs = []
    m = X.shape[0]

    for i in range(0, num_iterations):

        A, cache = forward_propagation(X, parameters)
        cost = compute_cost(A, Y)
        
        if i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)

    return parameters, costs

def model():
    X_trainning, y_trainning, X_test, y_test = load_data('creditcard.csv')
    parameters = initialize_parameters(X_trainning.shape[0], 10, 1)
    parameters, cost_history = two_layer_model(X_trainning, y_trainning, parameters)

if __name__ == '__main__':
    model()