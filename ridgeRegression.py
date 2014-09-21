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