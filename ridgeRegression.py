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


def cv(x,y):
	kFold = len(x)/10
	L = 0.2
	for i in range(0, 10):
		Bk = ridgeRegress(x[currFold],y[currFold],L)


	#use Bk on test 