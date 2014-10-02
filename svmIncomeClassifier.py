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
	clf = SVC(C=50, kernel='poly', degree=2, gamma=0.0, 
		coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, 
		class_weight=None, verbose=False, max_iter=-1, random_state=None)
	clf.fit(z, y) 

	# C = 1, kernel = poly, degree = 3 --> 78.2669322709%
	# C = 50, kernel = poly, degree = 2 --> 76.1354581673%  
	# C = 50, kernel = poly, degree = 3 --> 84.5816733068%
	# C = 100, kernel = poly, degree = 3 --> 84.8605577689%

	# C = 1, kernel = rbf, degree = 3 --> 84.7343957503%
	# C = 50, kernel = rbf, degree = 3 --> 85.2988047809%
	# C = 100, kernel = rbf, degree = 3 --> 

	# C = 1, kernel = linear, degree = 3 --> 84.6414342629%


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

	print 'testing SVM...'
	predictions = []
	k = 0
	for i in range(0,y2.shape[0]):
		if (clf.predict(z2[i]) == '1'):
			predictions.append('>50K')
		else:
			predictions.append('<=50K')

		if (clf.predict(z2[i]) == y2[i]):
			k = k+1

	length = len(y2)
	acc = float(k)/float(length)
	print 'Correctly predicted ' + str(k) + ' out of: ' + str(len(y2))
	print 'Accuracy: ' + str(100*acc) + '%'

	#print(clf.score(z2,y2))

	return predictions
	# k = 0
	# for i in range(0,y.shape[0]):
	# 	if (clf.predict(z[i]) == y[i]):
	# 		k = k+1
	# print k





















# enc = OneHotEncoder(n_values='auto', categorical_features=[1,3,5,6,7,8,9,13], dtype=<type 'str'>, sparse=True)
# len(workclass) + len(education) + len(maritalStatus) + len(occupation) + len(relationship) + len(race) + len(sex) + len(nativeCountry) =  99
# for i in range(0,x.shape[0]):
# 	x[i,0] = int(x[i,0])
# 	x[i,1] = int(workclass.index(x[i,1]))
# 	x[i,2] = int(x[i,2])
# 	x[i,3] = int(education.index(x[i,3]))
# 	x[i,4] = int(x[i,4])
# 	x[i,5] = int(maritalStatus.index(x[i,5]))
# 	x[i,6] = int(occupation.index(x[i,6]))
# 	x[i,7] = int(relationship.index(x[i,7]))
# 	x[i,8] = int(race.index(x[i,8]))
# 	x[i,9] = int(sex.index(x[i,9]))
# 	x[i,10] = int(x[i,10])
# 	x[i,11] = int(x[i,11])
# 	x[i,12] = int(x[i,12])
# 	x[i,13] = int(nativeCountry.index(x[i,13]))



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