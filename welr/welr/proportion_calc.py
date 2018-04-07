#! -*- coding:utf-8 -*-

import simplejson as json
from MySQLHelper import *
from common import Common

class ProportionCALC:
    def __init__(self, dataset, inputStr):
        self.proportion = 0.3      #topn%
        self.dataset = dataset
        self.inputStr = inputStr
        self.mysqlHelper = MySQLHelper(Common.mysqlServer, Common.mysqlUser, Common.mysqlPassword)
        self.mysqlHelper.setDB("ocr")

    # 计算idf
    def calcIDF(self):
        dic = {}
        inputDic = json.loads(self.inputStr)
        for key in inputDic:
            lineDic = {}
            rows = self.mysqlHelper.queryAll(
                "select word, tfidf_num from idf where dataset='" + self.dataset + "' and aid='" + key + "'")
            for item in rows:
                lineDic[item['word']] = float(item['tfidf_num'])
            dic[key] = lineDic

        return dic

    def handle(self):
        idf_dic = self.calcIDF()
        inputDic = json.loads(self.inputStr)

        whereData = {"dataset": self.dataset}
        self.mysqlHelper.delete("proportion", whereData)

        for key in inputDic:
            filename = key
            content = inputDic[key].strip().split(' ')
            needExpandArray = []
            for item in content:
                if idf_dic[key].has_key(item):
                    needExpandArray.append((item, idf_dic[key][item]))

            sortedNeedExpandArray = sorted(needExpandArray, key=lambda x: x[1], reverse=1)
            tmpArray = sortedNeedExpandArray[0:int(self.proportion * len(needExpandArray))]  # 只对proportion%的单词进行查询扩展
            arr = [w[0] for w in tmpArray]
            dataSource = {"filename": filename, "keywords": " ".join(arr), "dataset": self.dataset}
            self.mysqlHelper.insert("proportion", dataSource)