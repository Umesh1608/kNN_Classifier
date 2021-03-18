import os
import re
import csv
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv


#Importing the heart train set, heart train label and test data from the text files.
heart_trainSet_data = []
heart_trainLabels_data = []
heart_testSet_data = []

f_heart_trainSet = open("heart_trainSet.txt", "r")
f_heart_trainLabels = open("heart_trainLabels.txt", "r")
f_heart_testSet = open("heart_testSet.txt", "r")

for line in f_heart_trainSet:
	heart_trainSet_data.append(re.findall(r"[-+]?\d*\.\d+|\d+|-\d", line))
for line in f_heart_trainLabels:
	heart_trainLabels_data.append(re.findall(r"[-+]?\d*\.\d+|\d+|-\d", line))
for line in f_heart_testSet:
	heart_testSet_data.append(re.findall(r"[-+]?\d*\.\d+|\d+|-\d", line))


print("heart train set data")
print(np.array(heart_trainSet_data))
print(np.array(heart_trainSet_data).shape)

print("heart train label data")
print(np.array(heart_trainLabels_data))
print(np.array(heart_trainLabels_data).shape)

print("heart test set data")
print(np.array(heart_testSet_data))
print(np.array(heart_testSet_data).shape)


