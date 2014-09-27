#!/usr/bin/python
import sys
import numpy as np
import matplotlib.pyplot as pl
from sklearn.svm import SVC





def	processDataSet(dataFile)
	fileMatrix = np.asmatrix(np.loadtxt(dataFile, dtype = 'string', delimiter = ", ",unpack=True)).transpose()
	x = np.delete(fileMatrix,((fileMatrix.shape)[1] -1), 1)
	y = fileMatrix[:,((fileMatrix.shape)[1] -1)]


	



