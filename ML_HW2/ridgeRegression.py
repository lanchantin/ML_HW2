#!/usr/bin/python
import sys
import numpy as np
import matplotlib.pyplot as pl


def loadDataSet(dataFile):
	xVal = np.loadtxt(dataFile, delimiter = " ", usecols=(0,1),unpack=True)
	yVal = np.loadtxt(dataFile, delimiter = " ", usecols=(2,),unpack=True)
	# pl.plot(xVal[1],yVal,'ro')
	# pl.xlabel(r'$X_1$')
	# pl.ylabel('Y')
	# pl.title('Question 4 - Part A')
	# pl.legend('Y',loc=2)
	# pl.show()
	return (xVal,yVal)


def normalEquations(x,y,theta):
	y1 = np.array(y)
	lambda = 0.2

	x1 = np.array(x) 
	x2 = np.array(x.transpose())

	z = np.dot(x1,x2)
	z = np.linalg.inv(z)

	v = np.ones(2)
	v[0] = np.dot(x1[0,:],y1)
	v[1] = np.dot(x1[1,:],y1)

	theta[0] = np.dot(z[:,0],v)
	theta[1] = np.dot(z[:,1],v)

	return theta