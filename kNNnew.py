import os
import re
import csv
import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from collections import Counter
from sklearn.model_selection import LeaveOneOut


####################################################### Question 1 ##################################################################

#Importing the heart train set, heart train label and test data from the text files.
X_train= np.loadtxt('heart_trainSet.txt', delimiter=',')
Y_train= np.loadtxt('heart_trainLabels.txt', delimiter=',',dtype=int)
X_test= np.loadtxt('heart_testSet.txt', delimiter=',')

#leave out one CV

loo = LeaveOneOut()
X_predict_arr_err =[]
k_vals = list(range(1,11))
for k in k_vals:
	X_test_predict = []
	y_test_CV = []
	for train_index, test_index in loo.split(X_train): #this splits the training set using leave one out 

		X_trainCV, X_testCV = X_train[train_index], X_train[test_index]
		y_trainCV , y_testCV = Y_train[train_index], Y_train[test_index]

		y_test_CV.append(y_testCV)

		Norm_data = []
		for i in range(len(X_trainCV)):
			Norm_data.append(np.linalg.norm(X_testCV-X_trainCV[i]))

		Norm_data_arr = np.array(Norm_data)

		#Sorting the Norm data array and getting the indices of the maximum k value
		Indi= np.argpartition(Norm_data_arr, k)[:k]

		#Comparing the y values of the corresponding maximum values
		Max_y = []
		for i in Indi:
			Max_y.append(y_trainCV[i])

		#Prediction for X_testCV
		if np.count_nonzero(np.array(Max_y) == 1) > np.count_nonzero(np.array(Max_y) == -1):
			X_test_predict.append(1)
		else:
			X_test_predict.append(-1)

	X_test_predict_arr = np.array(X_test_predict)
	y_test_CV_arr = np.array(y_test_CV)

	#Reshaping the X_test_predict_arr to compare with y_test_CV_arr

	X_test_predict_arr_re = np.reshape(X_test_predict_arr,(len(y_test_CV_arr),1))
	X_predict_arr_error = (len(X_train)-np.count_nonzero(X_test_predict_arr_re==y_test_CV_arr))/len(X_train)

	# X_predict_arr_err.append(X_predict_arr_error)

	print("Number of elements correctly predicted with k= "+str(k)+"----->"+str(np.count_nonzero(X_test_predict_arr_re==y_test_CV_arr)))
	print("\t")
	print(" Leave out one error for  k= "+str(k)+"----->"+str(X_predict_arr_error))
	print("\t")
	X_predict_arr_err.append(X_predict_arr_error)

print("Best k value is = "+str(np.argmin(np.array(X_predict_arr_err))+1))
print("\t")

#Predicted labels for the testSet with k =6

#Calculating the 2-norm between x test data and x train
X_test_predict = []
k = np.argmin(np.array(X_predict_arr_err))+1
for j in range(len(X_test)):
	Norm_data = []
	for i in range(len(X_train)):
		Norm_data.append(np.linalg.norm(X_test[j]-X_train[i]))

	Norm_data_arr = np.array(Norm_data)

	#Sorting the Norm data array and getting the indices of the maximum k value
	Indi= np.argpartition(Norm_data_arr, k)[:k]

	#Comparing the y values of the corresponding maximum values
	Max_y = []
	for i in Indi:
		Max_y.append(Y_train[i])

	#Prediction for X_testif np.count_nonzero(np.array(Max_y) == 1) > np.count_nonzero(np.array(Max_y) == -1):
	if np.count_nonzero(np.array(Max_y) == 1) > np.count_nonzero(np.array(Max_y) == -1):
		X_test_predict.append(1)
	else:
		X_test_predict.append(-1)

print("this is the predicted outputs for k = 6 --->" +str(np.array(X_test_predict)))







