#! -*- coding:utf-8 -*-

import re
import simplejson as json
from MySQLHelper import *
from common import Common

# 以json字符串的格式传入，同样返回一个json字符串

class RemoveStopwords:
    def __init__(self, inputStr):
        self.inputStr = inputStr
        self.mysqlHelper = MySQLHelper(Common.mysqlServer, Common.mysqlUser, Common.mysqlPassword)
        self.mysqlHelper.setDB("ocr")
        self.stopwords = []
        stopwords_lines = open("/var/www/html/thinkphp/Public/python/welr/nltk_stopwords.txt", 'r').readlines()
        for line in stopwords_lines:
            self.stopwords.append(line.strip())
        self.isRemoveNumber = False  # 是否删除数字
        self.spliter = re.compile('[^\w(\-*)(_*)]')

    def handle(self):
        dic = json.loads(self.inputStr)
        for key in dic:
            if self.isRemoveNumber:
                after = [word for word in self.spliter.split(dic[key].lower())
                         if len(word) > 0 and word not in self.stopwords and not re.match(r'^\d+$', word)]
                dic[key] = " ".join(after)
            else:
                after = [word for word in self.spliter.split(dic[key].lower())
                         if len(word) > 0 and word not in self.stopwords]
                dic[key] = " ".join(after)

        # 更新数据库
        for key in dic:
            pData = {"words": dic[key]}
            whereData = {"aid": key}
            self.mysqlHelper.update("translation", pData, whereData)

        dicStr = json.dumps(dic)
        return dicStr
