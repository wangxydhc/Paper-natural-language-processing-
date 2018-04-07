#! -*- coding:utf-8 -*-

from numpy import mat, shape, zeros, multiply, nonzero, exp, sign, array
import random
from MySQLHelper import *

class SVMProcess:
    def __init__(self):
        self.mysqlHelper = MySQLHelper("192.168.1.119", "root", "root")
        self.mysqlHelper.setDB("ocr")

    # 核函数 线性核 和 rbf核
    def kernelTrans(self, X, A, kTup):
        m, n = shape(X)
        K = mat(zeros((m,1)))
        if kTup[0] == 'lin': K = X *A.T
        elif kTup[0] == 'rbf':
            for j in range(m):
                deltaRow = X[j, :] - A
                K[j] = deltaRow * deltaRow.T
            K = exp(K / (-1 * kTup[1] ** 2))
        else: raise NameError("Can not recognize the kernel function!")
        return K

    # 选择不等于i的随机数
    def selectJrand(self, i, m):
        j = i
        while j == i:
            j = int(random.uniform(0, m))
        return j

    # 将aj限制在L和H之间 裁剪
    def clipAlpha(self, aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def calcEk(self, oS, k):
        # fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
        fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
        Ek = fXk - float(oS.labelMat[k])
        return Ek

    # 随机选择j， 和 selecJrand不同
    def selectJ(self, i, oS, Ei):
        maxK = -1;
        maxDeltaE = 0;
        Ej = 0
        oS.eCache[i] = [1, Ei]
        validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
        if len(validEcacheList) > 1:
            for k in validEcacheList:
                if k == i: continue
                Ek = self.calcEk(oS, k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k;
                    maxDeltaE = deltaE;
                    Ej = Ek
            return maxK, Ej
        else:
            j = self.selectJrand(i, oS.m)
            Ej = self.calcEk(oS, j)
        return j, Ej

    # 更新Ek
    def updateEk(self, oS, k):
        Ek = self.calcEk(oS, k)
        oS.eCache[k] = [1, Ek]

    # 内层循环
    def innerL(self, i, oS):
        Ei = self.calcEk(oS, i)
        # 优化权重
        currentC = oS.C * oS.queryNumMat[i] * oS.taoMat[i]
        if (oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < currentC) or (
                    oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > 0):
            j, Ej = self.selectJ(i, oS, Ei)
            alphaIold = oS.alphas[i].copy();
            alphaJold = oS.alphas[j].copy()
            if oS.labelMat[i] != oS.labelMat[j]:
                L = max(0, oS.alphas[j] - oS.alphas[j])
                H = min(currentC, currentC + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - currentC)
                H = min(currentC, oS.alphas[j] + oS.alphas[i])
            if L == H: print "L==H"; return 0
            # eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
            eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
            if eta >= 0: print "eta >= 0"; return 0
            oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
            oS.alphas[j] = self.clipAlpha(oS.alphas[j], H, L)
            self.updateEk(oS, j)
            if abs(oS.alphas[j] - alphaJold) < 0.00001:
                print "j not moving enough";
                return 0
            oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
            self.updateEk(oS, i)
            b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
            oS.alphas[j] - alphaJold) * oS.K[i, j]
            b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
            oS.alphas[j] - alphaJold) * oS.K[j, j]
            if 0 < oS.alphas[i] and oS.alphas[i] < currentC:
                oS.b = b1
            elif 0 < oS.alphas[j] and oS.alphas[j] < currentC:
                oS.b = b2
            else:
                oS.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    # 完整版的SMO算法
    def smoP(self, dataMatIn, classLabels, queryNumArr, taoArr, C, toler, maxIter, kTup=('lin', 0)):
        oS = optStruct(dataMatIn, mat(classLabels).transpose(), queryNumArr, taoArr, C, toler, kTup)
        iter = 0
        entireSet = True
        alphaPairsChanged = 0
        while iter < maxIter and ((alphaPairsChanged > 0) or (entireSet)):
            alphaPairsChanged = 0
            if entireSet:
                for i in range(oS.m):
                    alphaPairsChanged += self.innerL(i, oS)
                    # print "fullset, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
                iter += 1
            else:
                nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.innerL(i, oS)
                    # print "non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
                iter += 1
            if entireSet:
                entireSet = False
            elif alphaPairsChanged == 0:
                entireSet = True
            # print "iteration number: %d" % iter
        return oS.b, oS.alphas

    def calcWs(self, alphas, dataArr, classLabels):
        X = dataArr;
        labelMat = classLabels.transpose()
        m, n = shape(X)
        w = zeros((n, 1))
        for i in range(m):
            w += multiply(alphas[i] * labelMat[i], X[i, :].T)
        return w

    # 装载数据
    def loadDataSet(self):
        dataMat = [];
        labelMat = [];
        queryNumMat = [];
        taoMat = []

        featureRows = self.mysqlHelper.queryAll("select semantic_sim, jaccard_sim, idf_sum, keyword_num, text_len, label, query_num, tao_val from ir_feature")
        minArr = [100000, 100000, 100000, 100000, 100000]
        maxArr = [0, 0, 0, 0, 0]
        for row in featureRows:
            dataItem = [float(row['semantic_sim']),float(row['jaccard_sim']),float(row['idf_sum']),float(row['keyword_num']),float(row['text_len']) ]
            dataMat.append(dataItem)
            labelMat.append(float(row['label']))
            queryNumMat.append(float(row['query_num']))
            taoMat.append(float(row['tao_val']))
            for i in range(5):
                if minArr[i] > float(dataItem[i]):
                    minArr[i] = float(dataItem[i])
                if maxArr[i] < float(dataItem[i]):
                    maxArr[i] = float(dataItem[i])

        # 线性归一化
        for item in dataMat:
            for i in range(5):
                item[i] = (item[i] - minArr[i]) / float(maxArr[i] - minArr[i])

        # 预处理
        dataMat = mat(dataMat)
        return dataMat, labelMat, queryNumMat, taoMat

    def testRbf(self, trainFile, testFile, k1=13):
        dataArr, labelArr, queryNumArr, taoArr = self.loadDataSet(trainFile)
        # print "load complete..."
        b, alphas = self.smoP(dataArr, labelArr, queryNumArr, taoArr, 100000, 0.0001, 100, ('rbf', k1))
        dataMat = mat(dataArr);
        labelMat = mat(labelArr).transpose()
        svInd = nonzero(alphas.A > 0)[0]
        sVs = dataMat[svInd]
        labelSV = labelMat[svInd]
        print "there are %d Support Vectors" % shape(sVs)[0]
        m, n = shape(dataMat)
        errorCount = 0
        for i in range(m):
            kernelEval = self.kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
            predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
            if sign(predict) != sign(labelArr[i]):
                errorCount += 1
        print "The training error rate is : %f" % (float(errorCount) / m)

        dataArr, labelArr, queryNumArr, taoArr = self.loadDataSet(testFile)
        print "load complete..."
        errorCount = 0
        dataMat = mat(dataArr);
        labelMat = mat(labelArr).transpose()
        m, n = shape(dataMat)
        for i in range(m):
            kernelEval = self.kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
            predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
            if sign(predict) != sign(labelArr[i]):
                errorCount += 1
        print "The test error rate is : %f" % (float(errorCount) / m)

    def handleBeforeSave(self, vec):
        m, n = shape(vec)
        vecList = []
        for item in vec:
            for i in range(n):
                vecList.append(str(item[0, i]))
        return " ".join(vecList)

    # 根据两两相对顺序关系 整合列表
    def groupRankedList(self, respondKey, predictArr, sortedPrimaryResult):
        n = len(respondKey)
        keyList = []
        simList = []
        for item in sortedPrimaryResult:
            keyList.append(item[0])
            simList.append(item[1])

        i = 0
        while i < n:
            pVal = int(predictArr[i][0,0])
            key1 = respondKey[i][0]
            key2 = respondKey[i][1]
            if pVal == 1 :
                index1 = keyList.index(key1)
                index2 = keyList.index(key2)
                if index1 > index2:
                    keyList[index1], keyList[index2] = keyList[index2], keyList[index1]
                    simList[index1], simList[index2] = simList[index2], simList[index1]
            i += 1
        # result = []
        # for i in range(len(keyList)):
        #     result.append((keyList[i], simList[i]))
        return keyList, simList


    # 训练并保存支持向量、label[svInd]、alpha[svInd]、b
    def trainAndSaveSV(self, k1=13):
        # 读取ir_feature中的内容用来训练
        dataArr, labelArr, queryNumArr, taoArr = self.loadDataSet()
        b, alphas = self.smoP(dataArr, labelArr, queryNumArr, taoArr, 100000, 0.0001, 100, ('rbf', k1))
        labelMat = mat(labelArr).transpose()
        svInd = nonzero(alphas.A > 0)[0]
        sVs = dataArr[svInd]
        labelSV = labelMat[svInd]
        saveAlpha = alphas[svInd]
        m,n = shape(sVs)
        sVsStr = self.handleBeforeSave(sVs)
        labelStr = self.handleBeforeSave(labelSV)
        saveAlphaStr = self.handleBeforeSave(saveAlpha)
        bStr = self.handleBeforeSave(b)
        dataSource = {"dimension":str(n), "vectors":sVsStr}
        self.mysqlHelper.update("support_vector", dataSource, {"id":str(1)})
        dataSource = {"vectors":labelStr}
        self.mysqlHelper.update("support_vector", dataSource, {"id": str(2)})
        dataSource = {"vectors": saveAlphaStr}
        self.mysqlHelper.update("support_vector", dataSource, {"id": str(3)})
        dataSource = {"vectors": bStr}
        self.mysqlHelper.update("support_vector", dataSource, {"id": str(4)})

    # 预测相对顺序关系
    def predict(self, dataArr, k1=13):
        dataMat = mat(dataArr)
        m,n = shape(dataMat)

        # 读sVs
        sVsRow = self.mysqlHelper.queryOnlyRow("select vectors from support_vector where id=1")
        sVsList = sVsRow['vectors'].split(' ')
        sVsNum = len(sVsList) / 5
        sVsList2 = []
        i = 0
        while i < sVsNum:
            j = 0
            item = []
            while j < n:
                item.append(float(sVsList[i*n + j]))
                j += 1
            sVsList2.append(item)
            i += 1
        sVs = mat(sVsList2)

        # 读 label
        labelRow = self.mysqlHelper.queryOnlyRow("select vectors from support_vector where id=2")
        labelList = labelRow['vectors'].split(' ')
        labelSV = mat([float(item) for item in labelList]).T

        # 读alphas
        alphaRow = self.mysqlHelper.queryOnlyRow("select vectors from support_vector where id=3")
        alphaList = alphaRow['vectors'].split(' ')
        alphas = mat([float(item) for item in alphaList]).T

        # 读 b
        bRow = self.mysqlHelper.queryOnlyRow("select vectors from support_vector where id=4")
        bList = bRow['vectors'].split(' ')
        b = mat([float(item) for item in bList]).T

        # 预测
        predictResult = []
        for i in range(m):
            kernelEval = self.kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
            predictVal = kernelEval.T * multiply(labelSV, alphas) + b
            predictResult.append(sign(predictVal))

        return predictResult

    # 形成排序列表
    # sourceKey 表示查询项aid
    # featureDic 为键值对， 键为targetId，值为五个特征值列表
    def generateRankedList(self, keyArr, featureArr, sortedPrimaryResult):
        # 首先，生成两两文档对
        i = 0
        respondRelation = []
        featureResult = []
        while i < len(featureArr):
            j = i + 1
            while j < len(featureArr):
                first = array(featureArr[i])
                second = array(featureArr[j])
                respondRelation.append((keyArr[i],keyArr[j]))
                featureResult.append(first - second)
                j += 1
            i += 1
        # 使用self.predict函数预测以上的值
        predictResult = self.predict(featureResult)
        # 整合所有的顺序，得到最终列表
        keyList, simList = self.groupRankedList(respondRelation, predictResult, sortedPrimaryResult)
        return keyList, simList


# 公开的
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError("Can not recognize the kernel function!")
    return K


# 数据结构
class optStruct:
    def __init__(self, dataMatIn, classLabels, queryNumArr, taoArr, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.queryNumMat = queryNumArr
        self.taoMat = taoArr
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


