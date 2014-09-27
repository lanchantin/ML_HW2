#!/usr/bin/python
import sys
import numpy as np
import matplotlib.pyplot as pl
from sklearn.svm import SVC





def	processDataSet(dataFile):
	fileMatrix = np.asmatrix(np.loadtxt(dataFile, dtype = 'string', delimiter = ", ",unpack=True)).transpose()
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


	for i in range(0,x.shape[0]):
		try:
			x[i,1] = workclass.index(x[i,1])
		except ValueError:
			x[i,1] = ''
		try:
			x[i,3] = education.index(x[i,3])
		except ValueError:
			x[i,3] = ''
		try:
			x[i,5] = maritalStatus.index(x[i,5])
		except ValueError:
			x[i,5] = ''
		try:
			x[i,6] = occupation.index(x[i,6])
		except ValueError:
			x[i,6] = ''
		try:
			x[i,7] = relationship.index(x[i,7])
		except ValueError:
			x[i,7] = ''
		try:
			x[i,8] = race.index(x[i,8])
		except ValueError:
			x[i,8] = ''
		try:
			x[i,9] = sex.index(x[i,9])
		except ValueError:
			x[i,9] = ''
		try:
			x[i,13] = nativeCountry.index(x[i,13])
		except ValueError:
			x[i,13] = ''

	return (x,y)