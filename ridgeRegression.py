#!/usr/bin/python
import sys
import numpy as np
import matplotlib.pyplot as pl


def loadDataSet(dataFile):
	xVal = np.asmatrix(np.loadtxt(dataFile, delimiter = " ", usecols=(0,1),unpack=True)).transpose()
	yVal = np.asmatrix(np.loadtxt(dataFile, delimiter = " ", usecols=(2,),unpack=True)).transpose()

	return (xVal,yVal)


def ridgeRegress(x,y,L):
	I = np.identity(x[0].size) 
	theta = ((x.T*x)+L*I)*(x.T*y)
	return theta



# (Hint1: you should implement a function to split the data into ten folds; 
# 	then loop over the folds;use one as test, the rest train )
# (Hint2: for each fold, on the train part, perform ridgeRegress to learn βk; 
# 	Then use this βk on all samples in the test fold to get predicted yˆ; 
# 	Then calculate the error (difference) between true y and yˆ, sum over 
# 	all testing points in the current fold k. )

def cv(x,y):
	kFold = len(x)/10
	L = 0.2
	for i in range(0,10):
		currVal = i*kFold
		a = x[0:currVal]
		b = x[currVal:len(x)-1]
		xC = np.concatenate((a,b),axis = 0)

		a = y[0:currVal]
		b = y[currVal:(y.size - 1)]
		yC = np.concatenate((a,b),axis = 0)

		Bk = ridgeRegress(xC,yC,L)

		y_hat = x[currVal:(currVal+kFold)]*Bk


		for j in range(0,kFold):
			y[currVal+j] - y_hat[j]

	#use Bk on test 