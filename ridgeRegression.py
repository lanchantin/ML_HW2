#!/usr/bin/python
import sys
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def loadDataSet(dataFile):
	xVal = np.loadtxt(dataFile, delimiter = " ", usecols=(0,1,2))
	yVal = np.loadtxt(dataFile, delimiter = " ", usecols=(3,))
	return (xVal,yVal)


def ridgeRegress(x,y,L):
	I = np.identity(x[0].size) 
	theta = np.dot((np.linalg.inv(np.dot(x.T,x)+(L*I))),np.dot(x.T,y))
	return theta



def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

def cv(xIN,yIN):
	xTemp = xIN.tolist()
	yTemp = yIN.tolist()
	x = xTemp[:]
	y = yTemp[:]
	random.seed(37)
	random.shuffle(x)
	random.shuffle(y)

	x = np.asarray(x)
	y = np.asarray(y)

	foldElements = len(x)/10
	lowLoss = 0
	bestL = 0

	lossArr = []
	lArr = []
	for L in drange(0.02,1.02,0.02):
		lArr.append(L)
		loss = 0
		NUMFOLDS = 10
		for fold in range(0,NUMFOLDS):
			currFold = fold*foldElements

			a = x[0:currFold]
			b = x[currFold+foldElements:x.shape[0]]
			xTrain = np.concatenate((a,b),axis = 0)

			a = y[0:currFold]
			b = y[currFold+foldElements:len(y)]
			yTrain = np.concatenate((a,b),axis = 0)

			xTest = x[currFold:(currFold+foldElements)]
			yTest = y[currFold:(currFold+foldElements)]

			Bk = ridgeRegress(xTrain,yTrain,L)

			y_hat = np.dot(xTest,Bk)

			for j in range(0,len(yTest)):
				loss = loss + math.pow((yTest[j] - y_hat[j]),2)
				#print loss

		loss = loss/10
		#print loss
		lossArr.append(loss)
		if (L == 0.02):
			lowLoss = loss
			bestL = L
		else:
			if loss < lowLoss:
				lowLoss = loss
				bestL = L

	#plt.plot(lArr,lossArr)
	#plt.show()
	return bestL


def test():
	x,y = loadDataSet('RRdata.txt')
	betaLR = ridgeRegress(x,y,0)
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(x[:,1], x[:,2],y, color='g')
	X, Y = np.meshgrid(x[:,1], x[:,2])
	Z = np.dot(x,betaLR)
	ax.plot_surface(X, Y, Z)

	plt.show()

	lambdaBest = cv(x,y)
	betaRR = ridgeRegress(x,y,lambdaBest)

	fig2 = plt.figure()
	ax2 = Axes3D(fig2)
	Z2 = np.dot(x,betaRR)
	ax2.plot_surface(X, Y, Z2)
	ax2.scatter(x[:,1],x[:,2],y, color='m')
	plt.show()

	betaCorr = ridgeRegress(x[:,1], x[:,2],lambdaBest)
	return lambdaBest,betaRR
