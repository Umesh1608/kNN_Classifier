import os
import re
import csv
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv


#Importing the heart train set data from the text file.
heart_trainSet_data = []
f_heart_trainSet = open("heart_trainSet.txt", "r")
for line in f_heart_trainSet:
	heart_trainSet_data.append(re.findall(r"[-+]?\d*\.\d+|\d+|-\d", line))

print("heart train set data")
print(np.array(heart_trainSet_data))
print(np.array(heart_trainSet_data).shape)

#Importing the heart train label data from the text file.

heart_trainLabels_data = []
f_heart_trainLabels = open("heart_trainLabels.txt", "r")
for line in f_heart_trainLabels:
	heart_trainLabels_data.append(re.findall(r"[-+]?\d*\.\d+|\d+|-\d", line))

print("heart train label data")
print(np.array(heart_trainLabels_data))
print(np.array(heart_trainLabels_data).shape)

#Importing the heart test set data from the text file.

heart_testSet_data = []
f_heart_testSet = open("heart_testSet.txt", "r")
for line in f_heart_testSet:
	heart_testSet_data.append(re.findall(r"[-+]?\d*\.\d+|\d+|-\d", line))

print("heart test set data")
print(np.array(heart_testSet_data))
print(np.array(heart_testSet_data).shape)


