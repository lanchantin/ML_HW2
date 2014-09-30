#!/usr/bin/python
import sys
import numpy as np
import math
import random
dataFile = 'RRdata.txt'


def loadDataSet(dataFile):
	xval = np.loadtxt(dataFile, delimiter = " ", usecols=(0,1,2))
	yval = np.loadtxt(dataFile, delimiter = " ", usecols=(3,))
	return (xval,yval)


def ridgeRegress(x,y,L):
	I = np.identity(x[0].size) 
	theta = np.dot((np.linalg.inv(np.dot(x.T,x)+(L*I))),np.dot(x.T,y))
	return theta

x = xval
y = yval
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x[:,1], x[:,2])
Z = y

ax.plot_surface(X, Y, Z)
plt.show()



def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

def cv(x,y):
	kFolds = len(x)/10
	lowLoss = 0
	bestL = 0

	random.seed(37)
	random.shuffle(x)
	random.shuffle(y)

	for L in drange(0.2,1.02,0.02):
		loss = 0
		for i in range(0,10):
			currFold = i*kFolds

			a = x[0:currFold]
			b = x[currFold+kFolds:x.shape[0]]
			xTrain = np.concatenate((a,b),axis = 0)

			a = y[0:currFold]
			b = y[currFold+kFolds:y.shape[0]]
			yTrain = np.concatenate((a,b),axis = 0)

			xTest = x[currFold:(currFold+kFolds)]
			yTest = y[currFold:(currFold+kFolds)]

			Bk = ridgeRegress(xTrain,yTrain,L)

			y_hat = np.dot(xTest,Bk)

			for j in range(0,kFolds):
				loss = loss + math.pow((yTest[j] - y_hat[j]),2)
				# print loss

		loss = loss/10
		print loss
		if (L == 0.2):
			lowLoss = loss
			bestL = L
		else:
			if loss < lowLoss:
				lowLoss = loss
				bestL = L

	return bestL


def run():
	x,y = loadDataSet('RRdata.txt')
	lambdaBest = cv(x,y)
	betaRR = ridgeRegress(x,y,lambdaBest)
	return lambdaBest,betaRR