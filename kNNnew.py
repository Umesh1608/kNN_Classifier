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

# #Printing the data sets
# print(X_train)
# print(X_train.shape)
# print(Y_train)
# print(X_test)

#Printing the shape of the data sets
print("trainSet data = "+str(X_train.shape))
print("trainLabels data = "+str(Y_train.shape))
print("trainSet data = "+str(X_test.shape))

# #Calculating the 2-norm between x test data and x train
# X_test_predict = []
# k = 3
# for j in range(len(X_test)):
# 	Norm_data = []
# 	for i in range(len(X_train)):
# 		Norm_data.append(np.linalg.norm(X_test[j]-X_train[i]))

# 	Norm_data_arr = np.array(Norm_data)
# 	print(Norm_data_arr.shape)

# 	#Sorting the Norm data array and getting the indices of the maximum k value

# 	Indi_Norm_data_arr = np.argsort(Norm_data_arr)
# 	Indi= np.argpartition(Norm_data_arr, -k)[-k:]

# 	#Comparing the y values of the corresponding maximum values
# 	Max_y = []
# 	for i in Indi:
# 		Max_y.append(Y_train[i])

# 	#Prediction for X_test
# 	X_test_predict.append(max(Counter(Max_y)))

# # print("this is the predicted outputs for k = 3 --->" +str(np.array(X_test_predict)))

#leave out one CV

loo = LeaveOneOut()

k_vals = list(range(1,11))
for k in k_vals:
	X_test_predict = []
	y_test_CV = []
	print(loo.get_n_splits(X_train))
	for train_index, test_index in loo.split(X_train): #this splits the training set using leave one out 

		X_trainCV, X_testCV = X_train[train_index], X_train[test_index]
		y_trainCV , y_testCV = Y_train[train_index], Y_train[test_index]

		y_test_CV.append(y_testCV)


		Norm_data = []
		for i in range(len(X_trainCV)):
			Norm_data.append(np.linalg.norm(X_testCV-X_trainCV[i]))


		Norm_data_arr = np.array(Norm_data)

		#Sorting the Norm data array and getting the indices of the maximum k value

		Indi_Norm_data_arr = np.argsort(Norm_data_arr)
		Indi= np.argpartition(Norm_data_arr, -k)[-k:]

		#Comparing the y values of the corresponding maximum values
		Max_y = []
		for i in Indi:
			Max_y.append(y_trainCV[i])

		#Prediction for X_testCV
		X_test_predict.append(max(Counter(Max_y)))

	X_test_predict_arr = np.array(X_test_predict)
	y_test_CV_arr = np.array(y_test_CV)

	#Reshaping the X_test_predict_arr to compare with y_test_CV_arr
	X_test_predict_arr_re = np.reshape(X_test_predict_arr,(len(y_test_CV_arr),1))

	print(np.count_nonzero(X_test_predict_arr_re==y_test_CV_arr))

	print("Number of elements correctly predicted with k= "+str(k)+"----->"+str(np.count_nonzero(X_test_predict_arr_re==y_test_CV_arr)))



