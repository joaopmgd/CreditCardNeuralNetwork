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

def model():
    X_training, y_training, X_test, y_test = load_data('creditcard.csv')

if __name__ == '__main__':
    model()