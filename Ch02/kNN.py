'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharrin

Modified on Sep 11, 2018 by leon-zheng
'''
import numpy as np
import operator
from os import listdir

def classify(data, dataSet, labels, k):
    diff = data - dataSet
    distances = np.linalg.norm(diff, axis=1)
    sortedDistances = distances.argsort()
    classCount={}
    for i in range(k):
        voteLabel = labels[sortedDistances[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClass = sorted(classCount.iteritems(),key=lambda x:x[1],reverse=True)
    return sortedClass[0][0]

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines
    returnMat = np.zeros((numberOfLines,3))     #prepare matrix to return
    classLabel = []                             #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabel.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabel

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(dataSet.shape)
    normData = dataSet - minVals
    normData = normData / ranges
    return normData, ranges, minVals

def datingClassTest():
    holdOut = 0.50      #hold out 10%
    datingData,datingLabels = file2matrix('datingTestSet2.txt')
    normData, ranges, minVals = autoNorm(datingData)
    m = normData.shape[0]
    numTestVecs = int(m*holdOut)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normData[i,:],normData[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error count is: %d" % errorCount
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))

def classifyPerson():
    result = ['not at all', 'in small doses', 'in large doses']
    miles = float(raw_input('flier miles per year: '))
    percent = float(raw_input('percent of time spent playing video games: '))
    icecream = float(raw_input('liters of icecream consumed per year: '))
    datingData, datingLabels = file2matrix('datingTestSet2.txt')
    normData, ranges, minVals = autoNorm(datingData)
    data = np.array([miles, percent, icecream])
    class_ = classify((data - minVals) / ranges, normData, datingLabels, 3)
    print "You will probably like this person: ", result[class_ - 1]

def img2vector(filename):
    vector = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            vector[0,32*i+j] = int(lineStr[j])
    return vector

def handwritingClassTest():
    labels = []
    trainingList = listdir('trainingDigits')
    length = len(trainingList)
    trainingMat = np.zeros((length,1024))
    for i in range(length):
        filename = trainingList[i]
        filestr = filename.split('.')[0]        #take off .txt
        classNum = int(filestr.split('_')[0])
        labels.append(classNum)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % filename)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    lenTest = len(testFileList)
    for i in range(lenTest):
        filename = testFileList[i]
        filestr = filename.split('.')[0]
        classNum = int(filestr.split('_')[0])
        testData = img2vector('testDigits/%s' % filename)
        result = classify(testData, trainingMat, labels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (result, classNum)
        if (result != classNum): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(lenTest))