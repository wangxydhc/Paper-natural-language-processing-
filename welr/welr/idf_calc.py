#! -*- coding:utf-8 -*-

import math
import simplejson as json
from MySQLHelper import *

# 以json字符串的格式传入，计算结果存入数据库

class IDFCALC:

    def __init__(self, dataset, inputStr):
        self.dataset = dataset
        self.inputStr = inputStr

    # 计算idf,不是TFIDF
    def calcIDF(self):
        inputDic = json.loads(self.inputStr)
        doc_count = len(inputDic)
        dic = {}
        arrs = []  # 用来存每一行的单词
        voc = set("")  # 用来存所有的单词
        for key in inputDic:
            arrs.append(inputDic[key].split(' '))
            voc |= set(inputDic[key].split(' '))

        max_count = 0  # 用作归为[0,1]

        # 计算idf
        for word in voc:
            count = 0
            for arr in arrs:
                if word in arr:
                    count += 1
            tmp = math.log((float(doc_count) / count))  # 此处分母没有加1，原因是单词肯定至少有一个存在，count不会为0
            dic[word] = tmp
            max_count = tmp if tmp > max_count else max_count

        # 存数据库  现将之前的dataset数据删除
        mysqlHelper = MySQLHelper("localhost", "root", "root")
        mysqlHelper.setDB("ocr")
        whereData = {"dataset": self.dataset}
        mysqlHelper.delete("idf_true", whereData)
        for key in dic:
            dataSource = {"word": key, "idf_num": str(dic[key]), "dataset": self.dataset}
            mysqlHelper.insert("idf_true", dataSource)

    # 计算idf
    def handle(self):
        inputDic = json.loads(self.inputStr)
        doc_count = len(inputDic)
        dic = {}
        arrs = []  # 用来存每一行的单词
        voc = set("")  # 用来存所有的单词
        for key in inputDic:
            arrs.append(inputDic[key].split(' '))
            voc |= set(inputDic[key].split(' '))

        max_count = 0  # 用作归为[0,1]

        # 计算idf
        for word in voc:
            count = 0
            for arr in arrs:
                if word in arr:
                    count += 1
            tmp = math.log((float(doc_count) / count))  # 此处分母没有加1，原因是单词肯定至少有一个存在，count不会为0
            dic[word] = tmp
            max_count = tmp if tmp > max_count else max_count
        TFIDFArr = {}
        for key in inputDic:
            arr = inputDic[key].split(' ')
            lineLength = len(arr)
            if lineLength == 0:
                continue
            lineArr = {}
            for word in arr:
                if lineArr.has_key(word):
                    continue
                tmpCount = 0
                for w in arr:
                    if w == word:
                        tmpCount += 1
                # print key, word, str(float(tmpCount) / lineLength), str(dic[word])
                lineArr[word] = (float(tmpCount) / lineLength) * dic[word]
            TFIDFArr[key] = lineArr

        # 归一
        # for word in voc:
        #     dic[word] /= max_count

        # 存数据库  现将之前的dataset数据删除
        mysqlHelper = MySQLHelper("localhost", "root", "root")
        mysqlHelper.setDB("ocr")
        whereData = {"dataset":self.dataset}
        mysqlHelper.delete("idf", whereData)

        for key in TFIDFArr:
            lineArr = TFIDFArr[key]
            for word in lineArr:
                dataSource = {"word": word, "tfidf_num": str(lineArr[word]), "dataset": self.dataset, "aid":key}
                mysqlHelper.insert("idf", dataSource)


# s = '{"a":"I want to know this.","b":"He just took this away."}'
# i = IDFCALC("abc",s)
# i.handle()