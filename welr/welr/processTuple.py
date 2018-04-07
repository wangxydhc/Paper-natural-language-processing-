#! -*- coding:utf-8 -*-

from MySQLHelper import *
import numpy as np
from common import Common


class ProcessTuple:
    def __init__(self, dataset):
        self.dataset = dataset
        self.mysqlHelper = MySQLHelper(Common.mysqlServer, Common.mysqlUser, Common.mysqlPassword)
        self.mysqlHelper.setDB("ocr")
        self.qe_dic = []
        self.idf_dic = []
        self.alpha = 0.8

    # 计算两个词向量的cos值
    def simWtWu(self, wf, wu):
        wf_np = np.array(wf)
        wu_np = np.array(wu)
        lwf_np = np.sqrt(wf_np.dot(wf_np))
        lwu_np = np.sqrt(wu_np.dot(wu_np))
        cos = wf_np.dot(wu_np) / (lwf_np * lwu_np)
        # 归一化一下  cos值【-1，1】，归一化后为【0，1】
        cos = 0.5 + 0.5 * cos
        return cos

    # 计算w（词，词向量）和T文档中所有向量的最大值
    def sim_w_T(self, w, T, needQE=False):
        maxCos = 0.0

        word = w[0]
        qe_arr = []
        if needQE:
            qe_arr = self.getQE(word)
        # print word
        for vec in T:
            cos = self.simWtWu(w[1], vec[1])

            ext = 0  # 扩展项
            if len(qe_arr) != 0:
                sum = 0
                for item in qe_arr:
                    item_vector = self.getWordVector(item[0])
                    # sum += float(item[1]) * self.simWtWu(item_vector, vec[1])
                    sum += self.simWtWu(item_vector, vec[1])
                ext = float(sum) / len(qe_arr)

            if len(qe_arr) != 0:
                cos2 = self.alpha * cos + (1 - self.alpha) * ext  # cos加上扩展项
            else:
                cos2 = cos

            maxCos = maxCos if maxCos > cos2 else cos2

        return maxCos

    # 计算两个文档 S 和 T 的相似度
    def sim_S_T(self, S, T, source_key, target_key):
        fenzi = 0.0
        fenmu = 0

        if len(S) < len(T):
            S, T = T, S
            source_key, target_key = target_key, source_key

        # 找到S中应该扩展的词 已经存到了proportion表中
        needExpandArray = []
        row_dic = self.mysqlHelper.queryOnlyRow(
            "select keywords from proportion where filename='" + source_key + "' and dataset='" + self.dataset + "'")
        if row_dic.has_key("keywords"):
            needExpandArray = row_dic["keywords"].split(" ")

        for item in S:  # 每个item是（词，词向量）的一个元组
            cos_vec_T = 0
            if item[0] in needExpandArray:
                # cos_vec_T = sim_w_T(item, T, True) * idf_dic[item[0]]
                cos_vec_T = self.sim_w_T(item, T, True)
            else:
                if self.idf_dic.has_key(source_key) and self.idf_dic[source_key].has_key(item[0]):
                    # cos_vec_T = sim_w_T(item, T, False) * idf_dic[item[0]]
                    cos_vec_T = self.sim_w_T(item, T, False)
            # if cos_vec_T > 0:
            fenzi += cos_vec_T
            fenmu += 1
                # if idf_dic.has_key(item[0]):
                # fenmu += idf_dic[item[0]]

        return float(fenzi / fenmu)

    # 从训练好的词vector中，获取某一个单词的vector
    # word是要查找的词， vector_file是训练的词向量文件
    def getWordVector(self, word):
        vec = []
        row = self.mysqlHelper.queryOnlyRow("select * from data_vector where word='" + word + "'")
        if row is not None and len(row) > 0:
            arr = row['vector'].split(' ')
            vec = map(float, arr)
        return vec

    # 通过向量找到word
    def getWordByVector(self, vector):
        word = ""
        tmp = map(str, vector)
        vector_str = " ".join(tmp).strip()
        # print vector_str
        row = self.mysqlHelper.queryOnlyRow("select * from data_vector where vector like '%" + vector_str + "%'")
        if row is not None and len(row) > 0:
            word = row['word']
        return word

    # 将一个文档表示成词向量的list, list中每个元素是一个元组，第一项为词，第二项为词向量
    def doc2VecList(self, sString):
        res = []
        arr = sString.strip('\n').split(' ')
        for word in arr:
            vec = self.getWordVector(word)
            if len(vec) > 0:
                res.append((word, vec))
        return res

    # 得到QE集合 word是待扩展的词，sim_threshold是相似度阈值，经验值，待定  返回值是QE list, 其中每个元素是一个元组"（word,similarity）"
    def getQE(self, word, sim_threshold=0.7):
        # if word in self.qe_dic:  # qe_dic保存每个词的qe列表 空间换时间
        #     return self.qe_dic[word]
        res = []
        try:
            rows = self.mysqlHelper.queryAll(
                "select qe_word,similarity from qe where word='" + word + "' and dataset='" + self.dataset + "'")
            for item in rows:
                res.append((item["qe_word"], item["similarity"]))
        except:
            pass
        # if len(res) > 0:
        #     self.qe_dic[word] = res
        return res

    # 计算idf
    def calcIDF(self):
        rows = self.mysqlHelper.queryAll("select word, tfidf_num, aid from idf where dataset='" + self.dataset + "'")
        dic = {}
        artifactArr = []
        for item in rows:
            if item["aid"] not in artifactArr:
                artifactArr.append(item["aid"])
        for item in rows:
            if item["aid"] not in dic:
                dic[item["aid"]] = {}
            arr = dic[item["aid"]]
            arr[item['word']] = float(item['tfidf_num'])
        return dic

    # 计算Jaccard 系数
    def calcJaccardCoefficient(self, S, T):
        # 用词向量的平均数表示text向量
        s_vec = []
        t_vec = []
        s_arr = []
        t_arr = []

        for item in S:
            s_arr.append(item[1])
        for item in T:
            t_arr.append(item[1])

        S_np = np.array(s_arr)
        T_np = np.array(t_arr)
        i = 0
        lengthS = len(s_arr[0])
        lengthT = len(t_arr[0])
        while i < lengthS:
            tmp_s_col = S_np[:, i]
            s_vec.append(sum(tmp_s_col))
            i += 1
        i = 0
        while i < lengthT:
            tmp_t_col = T_np[:, i]
            t_vec.append(sum(tmp_t_col))
            i += 1

        s_vec_np = np.array(s_vec)
        t_vec_np = np.array(t_vec)

        fenmu = s_vec_np.dot(s_vec_np) + t_vec_np.dot(t_vec_np) - s_vec_np.dot(t_vec_np)
        jaccard = 0
        if fenmu != 0:
            jaccard = s_vec_np.dot(t_vec_np) / fenmu

        return jaccard

    def init(self):
        self.idf_dic = self.calcIDF()

    def handle(self, sourceKey, sourceStr, targetKey, targetStr):

        sourceVec = self.doc2VecList(sourceStr)
        targetVec = self.doc2VecList(targetStr)
        sim = self.sim_S_T(sourceVec, targetVec, sourceKey, targetKey)  # 计算相似度最耗时
        return sim





