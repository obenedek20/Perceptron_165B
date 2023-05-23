# Starter code for CS 165B MP3
import math
import random
import numpy as np
import pandas as pd

from typing import List

def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: pd.DataFrame
        testing_data: the same as training_data with "label" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """
    max_epochs = 100
    learning_rate = .01
    #Basically need to run through training examples over and over again until you go through it and get all of them classified right
    weights = train(training_data, max_epochs, learning_rate)
    preds = test(weights, testing_data)

    return preds

    #TODO implement your model and return the prediction

def train(training: pd.DataFrame, max_epochs: int, learning_rate: float) -> List[float]:
    #from what I can tell, should take in the training data, iterate until it gets all correct, then spit out the weight vector?
    weights = np.array([0,0,0,0,0])
    epochs = 0
    num_false = 1
    #assuming no bias
    while epochs < max_epochs and num_false > 0:
        num_false = 0
        epochs += 1
        for index, datapoint in training.iterrows():
            #changing it to -1 and 1 then will change it back when testing, using sign instead of sigmoid
            if datapoint.iloc[-1] != np.sign(np.dot(np.array(datapoint.iloc[:-1]),weights)):
                if datapoint.iloc[-1] == 0:
                    label = -1
                else:
                    label = 1
                weights = weights + learning_rate * label * datapoint.iloc[:-1] #rate * label * data
                num_false += 1
            else:
                continue
    return weights

def test(weights: List[float], testing: pd.DataFrame) -> List[int]:
    #should run through, classify each point, then return a list of the classifications (0 or 1)
    preds = []
    for index, datapoint in testing.iterrows():
        label = np.dot(weights, np.array(datapoint))
        if label == -1:
            preds.append(0)
        else:
            preds.append(1)
    return preds

def sigmoid(x):
    return 1/(1 + np.e**(-x))

if __name__ == '__main__':
    # load data
    training = pd.read_csv('data/train.csv')
    testing = pd.read_csv('data/dev.csv')
    target_label = testing['target']
    testing.drop('target', axis=1, inplace=True)

    # run training and testing
    prediction = run_train_test(training, testing)

    # check accuracy
    target_label = target_label.values
    print("Dev Accuracy: ", np.sum(prediction == target_label) / len(target_label))
    


    


