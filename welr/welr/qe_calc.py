#! -*- coding:utf-8 -*-

from MySQLHelper import *
import gensim
from common import Common

class QECALC:
    def __init__(self, dataset):
        self.dataset = dataset
        self.sim_threshold = 0.7
        self.mysqlHelper = MySQLHelper(Common.mysqlServer, Common.mysqlUser, Common.mysqlPassword)
        self.mysqlHelper.setDB("ocr")
        self.w2vModel = gensim.models.KeyedVectors.load_word2vec_format('/var/www/html/thinkphp/Public/python/welr/wiki.en.vector', binary=False)

    def handle(self):
        voc = set("")
        rows = self.mysqlHelper.queryAll("select keywords from proportion where dataset='" + self.dataset + "'")
        for item in rows:
            line = item['keywords']
            voc |= set(line.split(' '))

        for item in voc:
            try:
                arr = self.w2vModel.most_similar(item, topn=20)
                res = [w for w in arr if w[1] > self.sim_threshold]
                for tup in res:
                    dataSource = {"word": item, "qe_word": str(tup[0]), "similarity": str(tup[1]),
                                  "dataset": self.dataset}
                    self.mysqlHelper.insert("qe", dataSource)
            except:
                pass