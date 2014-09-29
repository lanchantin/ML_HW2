#!/usr/bin/python
import sys
import numpy as np
import math
import random
dataFile = 'RRdata.txt'


def loadDataSet(dataFile):
	xval = np.loadtxt(dataFile, delimiter = " ", usecols=(0,1,2),unpack=True).transpose()
	yval = np.loadtxt(dataFile, delimiter = " ", usecols=(3,),unpack=True).transpose()

	return (xval,yval)


def ridgeRegress(x,y,L):
	I = np.identity(x[0].size) 
	theta = np.dot((np.linalg.inv(np.dot(x.T,x)+L*I)),np.dot(x.T,y))
	return theta

# Bk = ridgeRegress(xval,yval,0)
# Z = np.dot(xval,Bk)
# X, Y = np.meshgrid(x[:,1], x[:,2])
# ax.plot_surface(X, Y, yval)



def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

def cv(x,y):
	kFold = len(x)/10
	lowSum = 0
	bestL = 0

	random.seed(30)
	random.shuffle(x)
	random.shuffle(y)

	for L in drange(0,1.02,0.02):
		sum = 0
		for i in range(0,10):
			testFold = i*kFold
			a = x[0:testFold]
			b = x[testFold+kFold:x.shape[0]]
			xTrain = np.concatenate((a,b),axis = 0)

			a = y[0:testFold]
			b = y[testFold+kFold:y.shape[0]]
			yTrain = np.concatenate((a,b),axis = 0)

			Bk = ridgeRegress(xTrain,yTrain,L)

			y_hat = np.dot(x[testFold:(testFold+kFold)],Bk)

			for j in range(0,kFold):
				sum = sum + math.pow((y[testFold+j] - y_hat[j]),2)
		print sum

		if (L == 0):
			lowSum = sum
			#print 'lowsum: ' + str(lowAvg)
		else:
			#print 'if ' + str(sum) + '<' + str(lowAvg)
			if sum < lowSum:
				lowSum = sum
				bestL = L
		print sum

	return bestL


def run():
	x,y = loadDataSet('RRdata.txt')
	lambdaBest = cv(x,y)
	betaRR = ridgeRegress(x,y,lambdaBest)
	return lambdaBest,betaRR
