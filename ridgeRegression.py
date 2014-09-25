#!/usr/bin/python
import sys
import numpy as np
import matplotlib.pyplot as pl


def loadDataSet(dataFile):
	xval = np.asmatrix(np.loadtxt(dataFile, delimiter = " ", usecols=(0,1,2),unpack=True)).transpose()
	yval = np.asmatrix(np.loadtxt(dataFile, delimiter = " ", usecols=(3,),unpack=True)).transpose()

	return (xval,yval)


def ridgeRegress(x,y,L):
	I = np.identity(x[0].size) 
	theta = (np.linalg.inv((x.T*x)+L*I))*(x.T*y)
	return theta



def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

def cv(x,y):
	kFold = len(x)/10
	lowSum = 0
	bestL = 0

	L = 0
	#for L in drange(0,1.02,0.02):
	sum = 0
	for i in range(0,10):
		testFold = i*kFold
		a = x[0:testFold]
		b = x[testFold+kFold:len(x)]
		xC = np.concatenate((a,b),axis = 0)

		a = y[0:testFold]
		b = y[testFold+kFold:(y.size)]
		yC = np.concatenate((a,b),axis = 0)

		Bk = ridgeRegress(xC,yC,L)

		y_hat = x[testFold:(testFold+kFold)]*Bk

		for j in range(0,kFold):
			sum = sum + abs(y[testFold+j] - y_hat[j])	
			print sum

		if (L == 0):
			lowsum = sum
		else:
			if sum < lowSum:
				lowSum = sum
				bestL = L

	print bestL




