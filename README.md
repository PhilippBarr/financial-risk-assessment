# Supervised Leaning

An excel file is provided which contains financial statements and some other information from a number of companies

Our data can be found in the "inputData.xlsx" file.
1) Fom columns A to H the performance index of the companies is presented
2) From columns I to K there are three binary activity indicators
3) In the L column it is the company status (1 all good, 2 bankrupt)
4) The M column is the year to which the above figures refer


After some thought, it was decided to write code in which:
1. We read the data from excel, normalize it, and divide it into training/test
sets (ignore column M with the year).
2. Train and evaluate the following supervised learning models on the sets
that we created, namely the following:

    • Linear Discriminant Analysis

    • Logistic Regression

    • Decision Trees

    • k-Nearest Neighbors
    
    • Naïve Bayes
    
    • Support Vector Machines
    
    • Neural Networks

After all, we pass the results of the experiments to an excel file where each line has the
following values:

Classifier Name | Training or test set| Number of training samples | Number of non-healthy companies in training sample | TP | TN | FP | FN | Precision | Recall | F1 score | Accuracy

-TP : True Positive 

-TN : True Negative

-FP : False Positive

-FN : False Negative
# Libraries and Specifications

Python version: 3.8

•Numpy 1.19.4

•Matplotlib.pyplot 3.3.3

•Xlrd 1.2.0

•Pandas 1.1.5

•Sklearn 0.23.2

•Tensorflow 2.4.0

•Sklearn 0.23.2

•Keras 2.4.3

•Xlsxwriter 1.3.7