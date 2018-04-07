#! -*- coding:utf-8 -*-

import sys
import random
import numpy as np
from numpy import mat, shape, zeros, multiply, nonzero, exp, sign
# import matplotlib.pyplot as plt
from sklearn import preprocessing
import string

# 装载数据
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        # print lineArr[0] + "\n" + lineArr[1] + "\n" + lineArr[2] + "\n" + lineArr[3] + "\n" + lineArr[4]
        dataMat.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3]), float(lineArr[4])])
        labelMat.append(float(lineArr[5]))

    # 预处理  正则化
    dataMat = mat(dataMat)
    # print dataMat, type(dataMat)
    dataMat_norm = mat(preprocessing.normalize(dataMat, norm='l2'))
    # print mat(dataMat_norm), type(mat(dataMat_norm))
    return dataMat_norm, labelMat

# 装载俄罗斯数据集
def loadRussiaDataSet(fileName, percent=0.3):
    dataMat = []; labelMat = []
    fr = open(fileName)
    lines = fr.readlines()
    lineLen = len(lines)
    cc = 0
    print lineLen
    while cc < lineLen * percent:
        line = lines[cc]
        lineArr = line.strip().split(' ')
        feaDic = {}
        for item in lineArr:
            if string.find(item, ':') != -1:
                arr = item.strip().split(':')
                feaDic[arr[0]] = float(arr[1])
        dataItem = []
        for i in range(246):
            if str(i) not in feaDic.keys():
                dataItem.append(0)
            else:
                dataItem.append(feaDic[str(i)])
        dataMat.append(dataItem)
        labelMat.append(lineArr[0])
        cc += 1

    dataMat = mat(dataMat)
    return dataMat, labelMat


# 选择不等于i的随机数
def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

# 将aj限制在L和H之间
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while iter < maxIter:
        alphaPairChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if (labelMat[i] * Ei < -toler and alphas[i] < C) or (labelMat[i] * Ei > toler and alphas[i] > 0):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L==H: print "L==H"; continue

                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i,:] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold < 0.00001): print "j not moving enough"; continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if 0 < alphas[i] and C > alphas[i]:
                    b = b1
                elif 0 < alphas[j] and C > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairChanged)

        if alphaPairChanged == 0:
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" % iter

    return b, alphas


# 核函数
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0] == 'lin': K = X *A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else: raise NameError("Houston we have a problem -- That kernel is not recognized")
    return K

# 数据结构
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i, :], kTup)

def calcEk(oS, k):
    # fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 随机选择j， 和 selecJrand不同
def selectJ(i, oS, Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i : continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

# 更新Ek
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

# 内层循环
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or (oS.labelMat[i]*Ei > oS.tol and oS.alphas[i] > 0):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j] :
            L = max(0, oS.alphas[j] - oS.alphas[j])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else :
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        # eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        eta = 2.0 * oS.K[i, j] - oS.K[i,i] - oS.K[j, j]
        if eta >= 0: print "eta >= 0"; return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        # b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        # b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold) * oS.K[i,i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i] - alphaIold) * oS.K[i,j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j,j]
        if 0 < oS.alphas[i] and oS.alphas[i] < oS.C :
            oS.b = b1
        elif 0 < oS.alphas[j] and oS.alphas[j] < oS.C :
            oS.b = b2
        else: oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

# 完整版的SMO算法
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(dataMatIn, mat(classLabels).transpose(), C, toler, kTup)
    print "init complete..."
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while iter < maxIter and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print "fullset, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        else :
            nonBoundIs = nonzero((oS.alphas.A > 0)*(oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False
        elif alphaPairsChanged == 0: entireSet = True
        print "iteration number: %d" % iter
    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
    X = dataArr; labelMat = classLabels.transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w
'''
def plot(dataArr, labelArr, alphas = [], b = []):
    posData = []; negData = []
    m = len(labelArr)
    for i in range(m):
        if labelArr[i] > 0:
           posData.append(dataArr[i])
        else:
            negData.append(dataArr[i])
    posNp = np.array(posData)
    negNp = np.array(negData)

    # 支持向量
    # svData = []
    # for i in range(len(alphas)):
    #     if alphas[i] > 0.0:
    #         svData.append(dataArr[i])
    # svNp = np.array(svData)

    # 函数
    # w = calcWs(alphas, dataArr, labelArr)
    # x = np.linspace(0,12,1000)
    # y = multiply(w, x) + b
    # print w

    plt.scatter(posNp[:, 0], posNp[:, 1], marker = 's', color='m')
    plt.scatter(negNp[:, 0], negNp[:, 1], marker = 'o', color='b')
    # plt.scatter(svNp[:,0], svNp[:, 1], color='y')
    # plt.plot(x, y)
    plt.show()
'''

def testRbf(trainFile, testFile, k1 = 13):
    dataArr, labelArr = loadRussiaDataSet(trainFile)
    print "load complete..."
    b, alphas = smoP(dataArr, labelArr, 2, 0.0001, 100, ('rbf', k1))
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print "there are %d Support Vectors" % shape(sVs)[0]
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print "The training error rate is : %f" % (float(errorCount) / m)

    dataArr, labelArr = loadRussiaDataSet(testFile)
    print "load complete..."
    errorCount = 0
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print "The test error rate is : %f" % (float(errorCount) / m)

if __name__ == "__main__":
    # trainFile = './svm/CM1.train'
    trainFile = './imat2009-datasets/imat2009_learning.txt'
    # testFile = './svm/CM1.test'
    testFile = './imat2009-datasets/imat2009_test.txt'

    # if len(sys.argv) < 2:
    #     print "缺少参数k!"
    #     exit()
    # k1 = float(sys.argv[1])
    testRbf(trainFile, testFile)
    # loadRussiaDataSet('./imat2009-datasets/imat2009_learning.txt')