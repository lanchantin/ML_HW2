#!/usr/bin/python
import sys
import numpy as np
import matplotlib.pyplot as pl
from sklearn.svm import SVC
from sklearn import preprocessing
dataFile = 'adult.test'



workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
education =  ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
maritalStatus = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'] 
relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'] 
race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
sex = ['Female', 'Male'] 
nativeCountry = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
salary = ['>50K', '<=50K']

def removeNan(x,y):
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
	return (x,y)

  

def processDataSet(dataFile):
	fileMatrix = np.loadtxt('adult.data', dtype = 'S26', delimiter = ", ")
	x = np.delete(fileMatrix,((fileMatrix.shape)[1] -1), 1)
	y = fileMatrix[:,((fileMatrix.shape)[1] -1)]
	salary = ['>50K', '<=50K']

	print 'preprocessing training data...'
	x,y = removeNan(x,y)

	for i in range(0,y.shape[0]):
		y[i] = int(salary.index(y[i]))

	s = (x.shape[0], 107)
	z = np.zeros(s)

	for i in range(0,x.shape[0]):
		z[i,0] = x[i,0]
		z[i,workclass.index(x[i,1])+1] = int(1)
		z[i,9] = x[i,2]
		z[i,education.index(x[i,3])+10] = 1
		z[i,26] = x[i,4]
		z[i,maritalStatus.index(x[i,5])+27] = 1
		z[i,occupation.index(x[i,6])+34] = 1
		z[i,relationship.index(x[i,7])+48] = 1
		z[i,race.index(x[i,8])+54] = 1
		z[i,sex.index(x[i,9])+59] = 1
		z[i,61] = x[i,10]
		z[i,62] = x[i,11]
		z[i,63] = x[i,12]
		z[i,nativeCountry.index(x[i,13])+64] = 1

	scaler1 = preprocessing.StandardScaler().fit(z[:,0])
	z[:,0] = scaler1.transform(z[:,0])                               
	scaler2 = preprocessing.StandardScaler().fit(z[:,9])
	z[:,9] = scaler2.transform(z[:,9])      
	scaler3 = preprocessing.StandardScaler().fit(z[:,26])
	z[:,26] = scaler3.transform(z[:,26]) 
	scaler4 = preprocessing.StandardScaler().fit(z[:,61])
	z[:,61] = scaler4.transform(z[:,61])   
	scaler5 = preprocessing.StandardScaler().fit(z[:,62])
	z[:,62] = scaler5.transform(z[:,62])  
	scaler6 = preprocessing.StandardScaler().fit(z[:,63])
	z[:,63] = scaler6.transform(z[:,63]) 

#################################################################################
#################################################################################
	print 'training SVM...'
	clf = SVC(C=1, kernel='rbf', degree=3, gamma=0, 
		coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, 
		class_weight=None, verbose=False, max_iter=-1, random_state=None)
	clf.fit(z, y) 

##################################################################################
#################################################################################

	fileMatrix = np.loadtxt(dataFile, dtype = 'S26', delimiter = ", ")
	x2 = np.delete(fileMatrix,((fileMatrix.shape)[1] -1), 1)
	y2 = fileMatrix[:,((fileMatrix.shape)[1] -1)]

	print 'preprocessing test data...'
	x2,y2 = removeNan(x2,y2)

	for i in range(0,y2.shape[0]):
		y2[i] = int(salary.index(y2[i]))

	s = (x2.shape[0], 107)
	z2 = np.zeros(s)

	for i in range(0,x2.shape[0]):
		z2[i,0] = x2[i,0]
		z2[i,workclass.index(x2[i,1])+1] = int(1)
		z2[i,9] = x2[i,2]
		z2[i,education.index(x2[i,3])+10] = 1
		z2[i,26] = x2[i,4]
		z2[i,maritalStatus.index(x2[i,5])+27] = 1
		z2[i,occupation.index(x2[i,6])+34] = 1
		z2[i,relationship.index(x2[i,7])+48] = 1
		z2[i,race.index(x2[i,8])+54] = 1
		z2[i,sex.index(x2[i,9])+59] = 1
		z2[i,61] = x2[i,10]
		z2[i,62] = x2[i,11]
		z2[i,63] = x2[i,12]
		z2[i,nativeCountry.index(x2[i,13])+64] = 1

	z2[:,0] = scaler1.transform(z2[:,0])                               
	z2[:,9] = scaler2.transform(z2[:,9])      
	z2[:,26] = scaler3.transform(z2[:,26]) 
	z2[:,61] = scaler4.transform(z2[:,61])   
	z2[:,62] = scaler5.transform(z2[:,62])  
	z2[:,63] = scaler6.transform(z2[:,63]) 


##################################################################################

	print 'testing SVM...'
	predictions = []


	k1 = 0
	for i in range(0,y.shape[0]):
		if (clf.predict(z[i]) == y[i]):
			k1 = k1+1
	acc1 = float(k1)/float(len(y))



	k2 = 0
	for i in range(0,y2.shape[0]):
		if (clf.predict(z2[i]) == '1'):
			predictions.append('>50K')
		else:
			predictions.append('<=50K')

		if (clf.predict(z2[i]) == y2[i]):
			k2 = k2+1
	acc2 = float(k2)/float(len(y2))

	print 'Correctly predicted ' + str(k1) + ' out of: ' + str(len(y)) + ' test samples'
	print 'Train Accuracy: ' + str(100*acc1) + '%'
	print 'Correctly predicted ' + str(k2) + ' out of: ' + str(len(y2)) +' train samples'
	print 'Test Accuracy: ' + str(100*acc2) + '%'

	# print 'Test Accuracy: ' + str(100*(clf.score(z,y))) + '%'
	# print 'Train Accuracy: ' + str(100*(clf.score(z2,y2))) + '%'

	return predictions