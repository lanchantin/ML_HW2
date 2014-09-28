#!/usr/bin/python
import sys
import numpy as np
import matplotlib.pyplot as pl
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
dataFile = 'adult.data'

def removeNan(x):
	for i in range(0,x.shape[0]):
		if(i < len(x)):
			for j in range(0,x.shape[1]):
				if (x[i,j] == '?'):
					x = np.delete(x, i, 0)
					y = np.delete(y, i, 0)
					break
	for i in range(0,x.shape[0]):
		if(i < len(x)):
			for j in range(0,x.shape[1]):
				if (x[i,j] == '?'):
					x = np.delete(x, i, 0)
					y = np.delete(y, i, 0)
					break
	return x

def	processDataSet(dataFile):
	fileMatrix = np.loadtxt('adult.data', dtype = 'string', delimiter = ", ",unpack=True).transpose()
	x = np.delete(fileMatrix,((fileMatrix.shape)[1] -1), 1)
	y = fileMatrix[:,((fileMatrix.shape)[1] -1)]

	workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
	education =  ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
	maritalStatus = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
	occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'] 
	relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'] 
	race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
	sex = ['Female', 'Male'] 
	nativeCountry = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
	salary = ['>50K', '<=50K']


	x = removeNan(x)

	s = (x.shape[0], 105)
	z = np.zeros(s)

	for i in range(0,x.shape[0]):
		z[i,0] = x[i,0]
		z[i,workclass.index(x[i,1])+1] = 1
		z[i,9] = x[i,2]
		z[i,education.index(x[i,3])+10] = 1
		z[i,26] = x[i,4]
		z[i,maritalStatus.index(x[i,5])+27] = 1
		z[i,occupation.index(x[i,6])+34] = 1
		z[i,relationship.index(x[i,7])+48] = 1
		z[i,race.index(x[i,8])+53] = 1
		z[i,sex.index(x[i,9])+58] = 1
		z[i,59] = x[i,10]
		z[i,60] = x[i,11]
		z[i,61] = x[i,12]
		z[i,nativeCountry.index(x[i,13])+62] = 1

	

	scaler = preprocessing.StandardScaler().fit(z[:,0])
	z[:,0] = scaler.transform(z[:,0])                               
	scaler = preprocessing.StandardScaler().fit(z[:,2])
	z[:,2] = scaler.transform(z[:,2])      
	scaler = preprocessing.StandardScaler().fit(z[:,4])
	z[:,4] = scaler.transform(z[:,4]) 
	scaler = preprocessing.StandardScaler().fit(z[:,10])
	z[:,10] = scaler.transform(z[:,10])   
	scaler = preprocessing.StandardScaler().fit(z[:,11])
	z[:,11] = scaler.transform(z[:,11])  
	scaler = preprocessing.StandardScaler().fit(z[:,12])
	z[:,12] = scaler.transform(z[:,12])      

	for i in range(0,y.shape[0]):
		y[i] = int(salary.index(y[i]))
	
	clf = SVC()
	clf.set_params(C=0.1, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
	clf.fit(z, y) 


	fileMatrix = np.loadtxt('adult.test', dtype = 'string', delimiter = ", ",unpack=True).transpose()
	x = np.delete(fileMatrix,((fileMatrix.shape)[1] -1), 1)
	y = fileMatrix[:,((fileMatrix.shape)[1] -1)]
	salary = ['>50K.', '<=50K.']

	k = 0
	for i in range(0,y.shape[0]):
		if (clf.predict(z[i]) == y[i]):
			k = k+1
	print k


	for i in range(0,500):
		print(clf.predict(z[i]))

	for i in range(0,y.shape[0]):
		if (clf.predict(z[i]) != '1'):
			print 7







































enc = OneHotEncoder(n_values='auto', categorical_features=[1,3,5,6,7,8,9,13], dtype=<type 'str'>, sparse=True)
len(workclass) + len(education) + len(maritalStatus) + len(occupation) + len(relationship) + len(race) + len(sex) + len(nativeCountry) =  99
for i in range(0,x.shape[0]):
	x[i,0] = int(x[i,0])
	x[i,1] = int(workclass.index(x[i,1]))
	x[i,2] = int(x[i,2])
	x[i,3] = int(education.index(x[i,3]))
	x[i,4] = int(x[i,4])
	x[i,5] = int(maritalStatus.index(x[i,5]))
	x[i,6] = int(occupation.index(x[i,6]))
	x[i,7] = int(relationship.index(x[i,7]))
	x[i,8] = int(race.index(x[i,8]))
	x[i,9] = int(sex.index(x[i,9]))
	x[i,10] = int(x[i,10])
	x[i,11] = int(x[i,11])
	x[i,12] = int(x[i,12])
	x[i,13] = int(nativeCountry.index(x[i,13]))



#0 0age: continuous.
#1 1-8workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
#2 9fnlwgt: continuous. 
#3 10-25-education =  ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
#4 26education-num: continuous. 
#5 27-33maritalStatus = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
#6 34-47occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'] 
#7 48-52relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'] 
#8 53-57race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
#9 58sex = ['Female', 'Male'] 
#10 59capital-gain: continuous. 
#11 60capital-loss: continuous. 
#12 61hours-per-week: continuous. 
#13 62-102nativeCountry = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']

# for i in range(0,z.shape[0]):
# 	z[i,0] = int(z[i,0])
# 	z[i,1] = int(z[i,1])
# 	z[i,2] = int(z[i,2])
# 	z[i,3] = int(z[i,3])
# 	z[i,4] = int(z[i,4])
# 	z[i,5] = int(z[i,5])
# 	z[i,6] = int(z[i,6])
# 	z[i,7] = int(z[i,7])
# 	z[i,8] = int(z[i,8])
# 	z[i,9] = int(z[i,9])
# 	z[i,10] = int(z[i,10])
# 	z[i,11] = int(z[i,11])
# 	z[i,12] = int(z[i,12])
# 	z[i,13] = int(z[i,13])



 