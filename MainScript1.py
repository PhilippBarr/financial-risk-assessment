import numpy as np
import matplotlib.pyplot as plt
import xlrd
import pandas as pd  # excel reading

from sklearn import svm, datasets  # svm= support vector machines
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, datasets
from sklearn.svm import SVC
import tensorflow as tf
import sklearn
import keras
import xlsxwriter
import math
import seaborn as sns

def printvalues(tp, fp, fn, tn, trainORtest):
    if trainORtest == 0:
        print("TP1_train=", tp)  # geting TP FP FN TN for train set
        print("FP1_train=", fp)
        print("FN1_train=", fn)
        print("TN1=_train", tn)
        print("\n")

    if trainORtest == 1:  #
        print("TP1_test=", tp)  # geting TP FP FN TN for test set
        print("FP1_test=", fp)
        print("FN1_test=", fn)
        print("TN1=_test", tn)
        print("\n")


def numOfNonHealthy(y):
    counter = 0
    for x in range(len(y)):
        if y[x] == 1:
            counter = counter + 1
#    print(counter)
    return counter

def justRoundIt(number):
    g = float("{:.2f}".format(number))
    return g



# load the dataset
fileName = 'inputData.xlsx'  # you may add the full path here
sheetName = 'Dataset2Use_Assignment2'


try:
    # Confirm file exists.
    sheetValues = pd.read_excel(fileName)
    print(' .. successful parsing of file:', fileName)
    print("Column headings:")
    print(sheetValues.columns)
except FileNotFoundError:
    print(FileNotFoundError)


inputData = sheetValues[sheetValues.columns[:-2]].values
print('inputData')
print(inputData)


outputData = sheetValues[sheetValues.columns[-2]]
print('output data')
print(outputData)

outputData, levels = pd.factorize(outputData)


print(' .. we have', inputData.shape[0], 'available paradigms.')
print(' .. each paradigm has', inputData.shape[1], 'features')
print(' ... the distribution for the available class lebels is:')

for classIdx in range(0, len(np.unique(outputData))):
    tmpCount = sum(outputData == classIdx)
    tmpPercentage = tmpCount/len(outputData)
    print(' .. class', str(classIdx), 'has', str(tmpCount), 'instances', '(', '{:.2f}'.format(tmpPercentage), '%)')

# ola kala

# from sklearn.model_selection import train_test_split (it is on top)
# spame ta dedomena se test kai train
X_train, X_test, y_train, y_test = train_test_split(inputData, outputData, random_state=0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# now the models
logreg = LogisticRegression()
logreg.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = logreg.predict(X_train)
y_pred_test = logreg.predict(X_test)

# y_test , y_pred_test
disp = confusion_matrix(y_test, y_pred_test)
f, ax =plt.subplots(figsize = (5,5))

sns.heatmap(disp,annot = True, linewidths= 0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Logistic Regression")
plt.show()


measures_test = [[0 for x in range(4)] for y in range(7)]  # tp fp fn tn for test
measures_train = [[0 for x in range(4)] for y in range(7)]  # tp fp fn tn for train

trainSum = np.array([0 for x in range(7)])  # number of training samples train
testSum = np.array([0 for x in range(7)])  # number of test samples test

nonHealthy_train = np.array([0 for x in range(7)])  # non healthy counter on train
nonHealthy_test = np.array([0 for x in range(7)])  # non healthy counter on test

scores_train = [[0 for x in range(4)] for y in range(7)]  # acc pre rec f1  train
scores_test = [[0 for x in range(4)] for y in range(7)]  # acc pre rec f1 test


measures_test[0][0] = disp[0][0]  # true positive test
measures_test[0][1] = disp[0][1]  # false positive test
measures_test[0][2] = disp[1][0]  # false negative test
measures_test[0][3] = disp[1][1]  # true negative test

printvalues(measures_test[0][0], measures_test[0][1], measures_test[0][2], measures_test[0][3], 1)

# ans=confusion_matrix(y_test,y_pred_test) #idio me pano
ans = confusion_matrix(y_train, y_pred_train)  # confussion matrix for train set
measures_train[0][0] = ans[0][0]  # true positive train
measures_train[0][1] = ans[0][1]  # false positive train
measures_train[0][2] = ans[1][0]  # false negative train
measures_train[0][3] = ans[1][1]  # true negative train


printvalues(measures_train[0][0], measures_train[0][1], measures_train[0][2], measures_train[0][3], 0)

# number of training samples
trainSum[0] = measures_train[0][0]+measures_train[0][1]+measures_train[0][2]+measures_train[0][3]
# number of test samples
testSum[0] = measures_test[0][0]+measures_test[0][1]+measures_test[0][2]+measures_test[0][3]

nonHealthy_train[0] = numOfNonHealthy(y_train)
nonHealthy_test[0] = numOfNonHealthy(y_test)

# calculate the scores
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

pre_train = precision_score(y_train, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')  # vgazei kati


rec_train = recall_score(y_train, y_pred_train, average='macro')
rec_test = recall_score(y_test, y_pred_test, average='macro')

f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')


scores_train[0][0] = justRoundIt(acc_train)
scores_train[0][1] = justRoundIt(pre_train)
scores_train[0][2] = justRoundIt(rec_train)
scores_train[0][3] = justRoundIt(f1_train)


scores_test[0][0] = justRoundIt(acc_test)
scores_test[0][1] = justRoundIt(pre_test)
scores_test[0][2] = justRoundIt(rec_test)
scores_test[0][3] = justRoundIt(f1_test)

# print the scores
print('Accuracy scores of Logistic regression classifier are:',
      'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
print('Precision scores of Logistic regression classifier are:',
      'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
print('Recall scores of Logistic regression classifier are:',
      'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
print('F1 scores of Logistic regression classifier are:',
      'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))

print('\n\n')

# from sklearn.tree import DecisionTreeClassifier (it is on top)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# plotting
# y_test , y_pred_test
disp = confusion_matrix(y_test, y_pred_test)
f, ax =plt.subplots(figsize = (5,5))

sns.heatmap(disp,annot = True, linewidths= 0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Decision Trees")
plt.show()

measures_test[1][0] = disp[0][0]  # true positive test
measures_test[1][1] = disp[0][1]  # false positive test
measures_test[1][2] = disp[1][0]  # false negative test
measures_test[1][3] = disp[1][1]  # true negative test

printvalues(measures_test[1][0], measures_test[1][1], measures_test[1][2], measures_test[1][3], 1)

ans = confusion_matrix(y_train, y_pred_train)  # confussion matrix for train set
measures_train[1][0] = ans[0][0]  # true positive test
measures_train[1][1] = ans[0][1]  # false positive test
measures_train[1][2] = ans[1][0]  # false negative test
measures_train[1][3] = ans[1][1]  # true negative test

printvalues(measures_train[1][0], measures_train[1][1], measures_train[1][2], measures_train[1][3], 0)

# number of training samples
trainSum[1] = measures_train[1][0]+measures_train[1][1]+measures_train[1][2]+measures_train[1][3]
# number of test samples
testSum[1] = measures_test[1][0]+measures_test[1][1]+measures_test[1][2]+measures_test[1][3]

nonHealthy_train[1] = numOfNonHealthy(y_train)
nonHealthy_test[1] = numOfNonHealthy(y_test)

# calculate the scores
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
pre_train = precision_score(y_train, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')
rec_train = recall_score(y_train, y_pred_train, average='macro')
rec_test = recall_score(y_test, y_pred_test, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

scores_train[1][0] = justRoundIt(acc_train)
scores_train[1][1] = justRoundIt(pre_train)
scores_train[1][2] = justRoundIt(rec_train)
scores_train[1][3] = justRoundIt(f1_train)


scores_test[1][0] = justRoundIt(acc_test)
scores_test[1][1] = justRoundIt(pre_test)
scores_test[1][2] = justRoundIt(rec_test)
scores_test[1][3] = justRoundIt(f1_test)

# print the scores
print('Accuracy scores of Decision Tree classifier are:',
      'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
print('Precision scores of Decision Tree classifier are:',
      'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
print('Recall scores of Decision Tree classifier are:',
      'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
print('F1 scores of Decision Tree classifier are:',
      'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))
print('\n\n')

# from sklearn.neighbors import KNeighborsClassifier (it is on top)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

# plotting
# y_test, y_pred_test
disp = confusion_matrix(y_test, y_pred_test)
f, ax =plt.subplots(figsize = (5,5))

sns.heatmap(disp,annot = True, linewidths= 0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("K-Nearest Neighbors")
plt.show()

measures_test[2][0] = disp[0][0]  # true positive test
measures_test[2][1] = disp[0][1]  # false positive test
measures_test[2][2] = disp[1][0]  # false negative test
measures_test[2][3] = disp[1][1]  # true negative test

printvalues(measures_test[2][0], measures_test[2][1], measures_test[2][2], measures_test[2][3], 1)

ans = confusion_matrix(y_train, y_pred_train)  # confussion matrix for train set
measures_train[2][0] = ans[0][0]  # true positive test
measures_train[2][1] = ans[0][1]  # false positive test
measures_train[2][2] = ans[1][0]  # false negative test
measures_train[2][3] = ans[1][1]  # true negative test

printvalues(measures_train[2][0], measures_train[2][1], measures_train[2][2], measures_train[2][3], 0)

# number of training samples
trainSum[2] = measures_train[2][0]+measures_train[2][1]+measures_train[2][2]+measures_train[2][3]
# number of test samples
testSum[2] = measures_test[2][0]+measures_test[2][1]+measures_test[2][2]+measures_test[2][3]

nonHealthy_train[2] = numOfNonHealthy(y_train)
nonHealthy_test[2] = numOfNonHealthy(y_test)

# calculate the scores
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
pre_train = precision_score(y_train, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')
rec_train = recall_score(y_train, y_pred_train, average='macro')
rec_test = recall_score(y_test, y_pred_test, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

scores_train[2][0] = justRoundIt(acc_train)
scores_train[2][1] = justRoundIt(pre_train)
scores_train[2][2] = justRoundIt(rec_train)
scores_train[2][3] = justRoundIt(f1_train)


scores_test[2][0] = justRoundIt(acc_test)
scores_test[2][1] = justRoundIt(pre_test)
scores_test[2][2] = justRoundIt(rec_test)
scores_test[2][3] = justRoundIt(f1_test)

# print the scores
print('Accuracy scores of K-NN classifier are:',
      'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
print('Precision scores of Logistic regression classifier are:',
      'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
print('Recall scores of K-NN classifier are:',
      'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
print('F1 scores of K-NN classifier are:',
      'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))

print('\n\n')


# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis (it is on top)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = lda.predict(X_train)
y_pred_test = lda.predict(X_test)

# plot
disp = confusion_matrix(y_test, y_pred_test)
f, ax =plt.subplots(figsize = (5,5))

sns.heatmap(disp,annot = True, linewidths= 0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Linear Discriminant Analysis")
plt.show()

measures_test[3][0] = disp[0][0]  # true positive test
measures_test[3][1] = disp[0][1]  # false positive test
measures_test[3][2] = disp[1][0]  # false negative test
measures_test[3][3] = disp[1][1]  # true negative test

printvalues(measures_test[3][0], measures_test[3][1], measures_test[3][2], measures_test[3][3], 1)

ans = confusion_matrix(y_train, y_pred_train)  # confussion matrix for train set
measures_train[3][0] = ans[0][0]  # true positive test
measures_train[3][1] = ans[0][1]  # false positive test
measures_train[3][2] = ans[1][0]  # false negative test
measures_train[3][3] = ans[1][1]  # true negative test

printvalues(measures_train[3][0], measures_train[3][1], measures_train[3][2], measures_train[3][3], 0)

# number of training samples
trainSum[3] = measures_train[3][0]+measures_train[3][1]+measures_train[3][2]+measures_train[3][3]
testSum[3] = measures_test[3][0]+measures_test[3][1]+measures_test[3][2]+measures_test[3][3]  # number of test samples

nonHealthy_train[3] = numOfNonHealthy(y_train)
nonHealthy_test[3] = numOfNonHealthy(y_test)


# calculate the scores
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
pre_train = precision_score(y_train, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')
rec_train = recall_score(y_train, y_pred_train, average='macro')
rec_test = recall_score(y_test, y_pred_test, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

scores_train[3][0] = justRoundIt(acc_train)
scores_train[3][1] = justRoundIt(pre_train)
scores_train[3][2] = justRoundIt(rec_train)
scores_train[3][3] = justRoundIt(f1_train)


scores_test[3][0] = justRoundIt(acc_test)
scores_test[3][1] = justRoundIt(pre_test)
scores_test[3][2] = justRoundIt(rec_test)
scores_test[3][3] = justRoundIt(f1_test)


# print the scores
print('Accuracy scores of LDA classifier are:',
      'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
print('Precision scores of LDA classifier are:',
      'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
print('Recall scores of LDA classifier are:',
      'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
print('F1 scores of LDA classifier are:',
      'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))

print('\n\n')

# from sklearn.naive_bayes import GaussianNB (it is on top)
gnb = GaussianNB()
gnb.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)

disp = confusion_matrix(y_test, y_pred_test)
f, ax =plt.subplots(figsize = (5,5))

sns.heatmap(disp,annot = True, linewidths= 0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Naive Bayes")
plt.show()

measures_test[4][0] = disp[0][0]  # true positive test
measures_test[4][1] = disp[0][1]  # false positive test
measures_test[4][2] = disp[1][0]  # false negative test
measures_test[4][3] = disp[1][1]  # true negative test

printvalues(measures_test[4][0], measures_test[4][1], measures_test[4][2], measures_test[4][3], 1)

ans = confusion_matrix(y_train, y_pred_train)  # confussion matrix for train set
measures_train[4][0] = ans[0][0]  # true positive test
measures_train[4][1] = ans[0][1]  # false positive test
measures_train[4][2] = ans[1][0]  # false negative test
measures_train[4][3] = ans[1][1]  # true negative test

printvalues(measures_train[4][0], measures_train[4][1], measures_train[4][2], measures_train[4][3], 0)

# number of training samples
trainSum[4] = measures_train[4][0]+measures_train[4][1]+measures_train[4][2]+measures_train[4][3]
testSum[4] = measures_test[4][0]+measures_test[4][1]+measures_test[4][2]+measures_test[4][3]

nonHealthy_train[4] = numOfNonHealthy(y_train)
nonHealthy_test[4] = numOfNonHealthy(y_test)

# calculate the scores
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
pre_train = precision_score(y_train, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')
rec_train = recall_score(y_train, y_pred_train, average='macro')
rec_test = recall_score(y_test, y_pred_test, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

scores_train[4][0] = justRoundIt(acc_train)
scores_train[4][1] = justRoundIt(pre_train)
scores_train[4][2] = justRoundIt(rec_train)
scores_train[4][3] = justRoundIt(f1_train)


scores_test[4][0] = justRoundIt(acc_test)
scores_test[4][1] = justRoundIt(pre_test)
scores_test[4][2] = justRoundIt(rec_test)
scores_test[4][3] = justRoundIt(f1_test)

# print the scores
print('Accuracy scores of GNB classifier are:',
      'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
print('Precision scores of GBN classifier are:',
      'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
print('Recall scores of GNB classifier are:',
      'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
print('F1 scores of GNB classifier are:',
      'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))

print('\n\n')

# from sklearn.svm import SVC (it is on top)
svm = SVC()
svm.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)


# plot
# tp tn y_test
disp = confusion_matrix(y_test, y_pred_test)
f, ax =plt.subplots(figsize = (5,5))

sns.heatmap(disp,annot = True, linewidths= 0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Support Vector Machines")
plt.show()


measures_test[5][0] = disp[0][0]  # true positive test
measures_test[5][1] = disp[0][1]  # false positive test
measures_test[5][2] = disp[1][0]  # false negative test
measures_test[5][3] = disp[1][1]  # true negative test

printvalues(measures_test[5][0], measures_test[5][1], measures_test[5][2], measures_test[5][3], 1)

ans = confusion_matrix(y_train, y_pred_train)  # confussion matrix for train set
measures_train[5][0] = ans[0][0]  # true positive test
measures_train[5][1] = ans[0][1]  # false positive test
measures_train[5][2] = ans[1][0]  # false negative test
measures_train[5][3] = ans[1][1]  # true negative test

printvalues(measures_train[5][0], measures_train[5][1], measures_train[5][2], measures_train[5][3], 0)

# number of training samples
trainSum[5] = measures_train[5][0]+measures_train[5][1]+measures_train[5][2]+measures_train[5][3]
testSum[5] = measures_test[5][0]+measures_test[5][1]+measures_test[5][2]+measures_test[5][3]

nonHealthy_train[5] = numOfNonHealthy(y_train)
nonHealthy_test[5] = numOfNonHealthy(y_test)

# calculate the scores
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
pre_train = precision_score(y_train, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')
rec_train = recall_score(y_train, y_pred_train, average='macro')
rec_test = recall_score(y_test, y_pred_test, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

scores_train[5][0] = justRoundIt(acc_train)
scores_train[5][1] = justRoundIt(pre_train)
scores_train[5][2] = justRoundIt(rec_train)
scores_train[5][3] = justRoundIt(f1_train)


scores_test[5][0] = justRoundIt(acc_test)
scores_test[5][1] = justRoundIt(pre_test)
scores_test[5][2] = justRoundIt(rec_test)
scores_test[5][3] = justRoundIt(f1_test)

# print the scores
print('Accuracy scores of SVM classifier are:',
      'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
print('Precision scores of SVM classifier are:',
      'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
print('Recall scores of SVM classifier are:',
      'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
print('F1 scores of SVM classifier are:',
      'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))

print('\n\n')

# neural
# import  tensorflow as tf (it is on top)
print("Running Neural, Please Wait...")
CustomModel = tf.keras.models.Sequential()
CustomModel.add(tf.keras.layers.Dense(16, input_dim=X_train.shape[1], activation=tf.nn.relu))
CustomModel.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))


CustomModel.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

CustomModel.fit(X_train, keras.utils.to_categorical(y_train), epochs=100, verbose=False)
y_pred_train = CustomModel.predict_classes(X_train)
y_pred_test = CustomModel.predict_classes(X_test)
# batch_size

# confussion
ans = confusion_matrix(y_test, y_pred_test)

measures_test[6][0] = disp.confusion_matrix[0][0]  # true positive
measures_test[6][1] = disp.confusion_matrix[0][1]  # false positive
measures_test[6][2] = disp.confusion_matrix[1][0]  # false negative
measures_test[6][3] = disp.confusion_matrix[1][1]  # true negative

printvalues(measures_test[6][0], measures_test[6][1], measures_test[6][2], measures_test[6][3], 1)

ans = confusion_matrix(y_train, y_pred_train)  # confussion matrix for train set
measures_train[6][0] = ans[0][0]  # true positive test
measures_train[6][1] = ans[0][1]  # false positive test
measures_train[6][2] = ans[1][0]  # false negative test
measures_train[6][3] = ans[1][1]  # true negative test

printvalues(measures_train[6][0], measures_train[6][1], measures_train[6][2], measures_train[6][3], 0)

# number of training samples
trainSum[6] = measures_train[6][0]+measures_train[6][1]+measures_train[6][2]+measures_train[6][3]
testSum[6] = measures_test[6][0]+measures_test[6][1]+measures_test[6][2]+measures_test[6][3]

nonHealthy_train[6] = numOfNonHealthy(y_train)
nonHealthy_test[6] = numOfNonHealthy(y_test)

# now check for both train and test data, how well the model learned the patterns
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
pre_train = precision_score(y_train, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')
rec_train = recall_score(y_train, y_pred_train, average='macro')
rec_test = recall_score(y_test, y_pred_test, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

scores_train[6][0] = justRoundIt(acc_train)
scores_train[6][1] = justRoundIt(pre_train)
scores_train[6][2] = justRoundIt(rec_train)
scores_train[6][3] = justRoundIt(f1_train)


scores_test[6][0] = justRoundIt(acc_test)
scores_test[6][1] = justRoundIt(pre_test)
scores_test[6][2] = justRoundIt(rec_test)
scores_test[6][3] = justRoundIt(f1_test)


# print the scores
print('Accuracy scores of ANN classifier are:',
      'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
print('Precision scores of ANN classifier are:',
      'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
print('Recall scores of ANN classifier are:',
      'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
print('F1 scores of ANN classifier are:',
      'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))

print('\n \n')

#  until now everything works great

# store data to xlsx file
# import xlsxwriter
outWorkbook = xlsxwriter.Workbook("OutputData1.xlsx")  # seting up file name to store the data
outSheet = outWorkbook.add_worksheet()  # seting up sheet name of the xslx file to store the data

headings = ["Class_Name", "Train_Test", "Num_Of_train_Samp", "Num_Of_Non_Healthy", "TP", "TN", "FP", "FN",
            "Precision", "Recall", "F1_Score", "Accuracy"]

className = ["Logistic_Regression", "Decision_Trees", "kNN", "LDA", "Naive_Bayes", "SVM", "Neural"]
# train

for i in range(12):
    outSheet.write(0, i, headings[i])

counter = 0
for i in range(1, 15, 2):
    outSheet.write(i, 0, className[counter])
    outSheet.write(i, 1, "Train")
    outSheet.write(i, 2, trainSum[counter])
    outSheet.write(i, 3, nonHealthy_train[counter])
    outSheet.write(i, 4, measures_train[counter][0])  # tp train
    outSheet.write(i, 5, measures_train[counter][3])  # tn train
    outSheet.write(i, 6, measures_train[counter][1])  # fp train
    outSheet.write(i, 7, measures_train[counter][2])  # fn train
    outSheet.write(i, 8, scores_train[counter][1])  # Precision
    outSheet.write(i, 9, scores_train[counter][2])  # Recall
    outSheet.write(i, 10, scores_train[counter][3])  # F1 score
    outSheet.write(i, 11, scores_train[counter][0])  # Precision
    counter = counter+1
# 0 14 2
counter = 0
for i in range(2, 16, 2):
    outSheet.write(i, 0, className[counter])
    outSheet.write(i, 1, "Test")
    outSheet.write(i, 2, testSum[counter])
    outSheet.write(i, 3, nonHealthy_test[counter])
    outSheet.write(i, 4, measures_test[counter][0])  # tp train
    outSheet.write(i, 5, measures_test[counter][3])  # tn train
    outSheet.write(i, 6, measures_test[counter][1])  # fp train
    outSheet.write(i, 7, measures_test[counter][2])  # fn train
    outSheet.write(i, 8, scores_test[counter][1])  # Precision
    outSheet.write(i, 9, scores_test[counter][2])  # Recall
    outSheet.write(i, 10, scores_test[counter][3])  # F1 score
    outSheet.write(i, 11, scores_test[counter][0])  # Precision
    counter = counter+1

# lets go

outWorkbook.close()  # close the file
