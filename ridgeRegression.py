#!/usr/bin/python
import sys
import numpy as np
dataFile = 'RRdata.txt'

def loadDataSet(dataFile):
	xval = np.loadtxt(dataFile, delimiter = " ", usecols=(0,1,2),unpack=True).transpose()
	yval = np.loadtxt(dataFile, delimiter = " ", usecols=(3,),unpack=True).transpose()

	return (xval,yval)


def ridgeRegress(x,y,L):
	I = np.identity(x[0].size) 
	theta = np.dot((np.linalg.inv(np.dot(x.T,x)+L*I)),np.dot(x.T,y))
	return theta



def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

def cv(x,y):
	kFold = len(x)/10
	lowAvg = 0
	bestL = 0

	#np.random.shuffle(x)

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
				sum = sum + abs(y[testFold+j] - y_hat[j])
		print sum

		if (L == 0):
			lowAvg = sum/10
			#print 'lowsum: ' + str(lowAvg)
		else:
			#print 'if ' + str(sum) + '<' + str(lowAvg)
			if sum/10 < lowAvg:
				lowAvg = sum/10
				bestL = L
		print sum/10

	return bestL


def run():
	x,y = loadDataSet('RRdata.txt')
	lambdaBest = cv(x,y)
	betaRR = ridgeRegress(x,y,lambdaBest)
	return lambdaBest,betaRR
