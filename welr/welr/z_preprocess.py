#! -*- coding:utf-8 -*-

from MySQLHelper import *
from common import Common
from process import Process
import random
import numpy as np

mysqlHelper = MySQLHelper(Common.mysqlServer, Common.mysqlUser, Common.mysqlPassword)
mysqlHelper.setDB("ocr")

'''
    将五个数据库中的内容存入DB
'''
def saveDB():
    print "代码已经完成其作用，不可重复执行！！！"
    return 0
    datasetDic = {
        "CM1":4,"eTOUR":5,"GANTT":6,"iTrust":7,"EasyClinic":8
    }
    classDic={
        "系统需求":1, "高级需求":2, "低级需求":3, "设计":4, "源代码":8,"用例":9,"测试用例":10,"UML交互":11,"类描述":12,"代码类":13
    }

    curSet = {
        "dataset":"EasyClinic",
        "class":"用例",
        "filePath":"EasyClinic/UC.total",
    }

    processObj = Process(datasetDic[curSet["dataset"]])
    docDic = processObj.doc2Dict(curSet["filePath"])
    for key in docDic.keys():
        dataSource = {'pid': str(datasetDic[curSet["dataset"]]), 'class': str(classDic[curSet["class"]]), 'name': str(key), 'content': str(docDic[key])}
        mysqlHelper.insert("artifact", dataSource)

    print "完成 " + curSet["dataset"] + " 数据集的 " + curSet['class'] + " ，文件路径为 " + curSet['filePath']

'''
    录入链接信息
'''
def saveLink():
    print "代码已经完成其作用，不可重复执行！！！"
    return 0
    datasetDic = {
        "CM1": 4, "eTOUR": 5, "GANTT": 6, "iTrust": 7, "EasyClinic": 8
    }
    classDic = {
        "系统需求": 1, "高级需求": 2, "低级需求": 3, "设计": 4, "源代码": 8, "用例": 9, "测试用例": 10, "UML交互": 11, "类描述": 12, "代码类": 13
    }

    curSet = {
        "dataset": "EasyClinic",
        "filePath": "EasyClinic/answers/UC_CC.txt.answer",
        "from_class": "用例",
        "to_class": "类描述"
    }

    processObj = Process(datasetDic[curSet["dataset"]])
    docDic = processObj.readAnswer(curSet["filePath"])
    pid = datasetDic[curSet["dataset"]]
    count = 0
    for key in docDic.keys():
        # 根据key，找到对应的artifact的id
        from_id = findArtifactId(key, pid, classDic[curSet["from_class"]])
        if from_id == 0:
            continue
        for target in docDic[key]:
            to_id = findArtifactId(target, pid, classDic[curSet["to_class"]])
            if to_id == 0:
                continue
            dataSource = {'pid': str(datasetDic[curSet["dataset"]]), 'from_class': str(classDic[curSet["from_class"]]),
                          'to_class': str(classDic[curSet["to_class"]]), 'from_aid': str(from_id), 'to_aid':str(to_id)}
            mysqlHelper.insert("answer", dataSource)
            count += 1

    print "共录入 " + str(count) + " 个链接"
    print "完成 " + curSet["dataset"] + " 数据集的 " + curSet['from_class'] + " 到 " + curSet['to_class'] + " 链接录入，文件路径为 " + curSet['filePath']

def findArtifactId(key, pid, curClass):
    row = mysqlHelper.queryOnlyRow("select id from artifact where name='" + str(key) + "' and pid='" + str(pid) + "' and class='" + str(curClass) + "'")
    if row != None and len(row) > 0:
        return row['id']
    return 0

'''
    设置反例数据集  正例和反例设为相同的数量, 反例随机选择  存入neg_answer
'''
def saveNegtiveAnswer():
    print "代码已经完成其作用，不可重复执行！！！"
    return 0
    datasetDic = {
        "CM1": 4, "eTOUR": 5, "GANTT": 6, "iTrust": 7, "EasyClinic": 8
    }
    classDic = {
        "系统需求": 1, "高级需求": 2, "低级需求": 3, "设计": 4, "源代码": 8, "用例": 9, "测试用例": 10, "UML交互": 11, "类描述": 12, "代码类": 13
    }

    curSet = {
        "dataset": "EasyClinic",
        "from_class": "用例",
        "to_class": "类描述"
    }

    answerRows = mysqlHelper.queryAll("select pid, from_aid, to_aid from answer where pid='"+str(datasetDic[curSet["dataset"]])+"' and from_class='"+str(classDic[curSet["from_class"]])+"' and to_class='"+str(classDic[curSet["to_class"]])+"'")
    fromClassRows = mysqlHelper.queryAll("select id from artifact where  pid='"+str(datasetDic[curSet["dataset"]])+"' and class='"+str(classDic[curSet["from_class"]])+"'")
    toClassRows = mysqlHelper.queryAll("select id from artifact where  pid='"+str(datasetDic[curSet["dataset"]])+"' and class='"+str(classDic[curSet["to_class"]])+"'")
    count = len(answerRows)
    posAnswer = []
    for row in answerRows:
        item = str(row["from_aid"]) + "_" + str(row["to_aid"])
        if item not in posAnswer:
            posAnswer.append(item)

    otherAnswer = []
    for fromRow in fromClassRows:
        for toRow in toClassRows:
            item = str(fromRow["id"]) + "_" + str(toRow["id"])
            if (item not in otherAnswer) and (item not in posAnswer):
                otherAnswer.append(item)

    negAnswer = random.sample(otherAnswer, count)

    # 删掉neg_answer表中已有的该数据集的内容
    whereData = {"pid": str(datasetDic[curSet["dataset"]]), "from_class": str(classDic[curSet["from_class"]]), "to_class": str(classDic[curSet["to_class"]])}
    mysqlHelper.delete("neg_answer", whereData)

    for item in negAnswer:
        fromAid = item.split("_")[0]
        toAid = item.split("_")[1]
        dataSource = {'pid': str(datasetDic[curSet["dataset"]]),
                      "from_class": str(classDic[curSet["from_class"]]), "to_class": str(classDic[curSet["to_class"]]),
                      "from_aid":str(fromAid), "to_aid": str(toAid) }
        mysqlHelper.insert("neg_answer", dataSource)

    print "共录入 " + str(len(negAnswer)) + " 个反例"
    print "完成 " + curSet["dataset"] + " 数据集的 " + curSet['from_class'] + " 到 " + curSet['to_class']


'''
    计算所有特征，存入数据表feature中
'''
def saveFeatures():
    print "代码已经完成其作用，不可重复执行！！！"
    return 0
    datasetDic = {
        "CM1": 4, "eTOUR": 5, "GANTT": 6, "iTrust": 7, "EasyClinic": 8
    }
    classDic = {
        "系统需求": 1, "高级需求": 2, "低级需求": 3, "设计": 4, "源代码": 8, "用例": 9, "测试用例": 10, "UML交互": 11, "类描述": 12, "代码类": 13
    }

    curSet = {
        "dataset": "CM1",
        "from_class": "高级需求",
        "to_class": "低级需求"
    }

    processObj = Process(datasetDic[curSet["dataset"]])
    posAnswers = mysqlHelper.queryAll("select from_aid, to_aid from answer where pid='"+str(datasetDic[curSet["dataset"]])+"' and from_class='"+str(classDic[curSet["from_class"]])+"' and to_class='"+str(classDic[curSet["to_class"]])+"'")
    negAnswers = mysqlHelper.queryAll("select from_aid, to_aid from neg_answer where pid='"+str(datasetDic[curSet["dataset"]])+"' and from_class='"+str(classDic[curSet["from_class"]])+"' and to_class='"+str(classDic[curSet["to_class"]])+"'")

    # 删掉之前计算的结果
    whereData = {"pid": str(datasetDic[curSet["dataset"]]),
                "from_class": str(classDic[curSet["from_class"]]), "to_class": str(classDic[curSet["to_class"]])}
    mysqlHelper.delete("feature", whereData)

    idfRows = mysqlHelper.queryAll(
        "select word, tfidf_num from idf where dataset='" + str(datasetDic[curSet["dataset"]]) + "'")
    idfDic = {}
    for row in idfRows:
        idfDic[row['word']] = row['tfidf_num']

    for posRow in posAnswers:
        fromAid = posRow['from_aid']
        toAid = posRow['to_aid']
        fromArtRow = mysqlHelper.queryOnlyRow("select name, content from artifact where id='" + str(fromAid) + "'")
        toArtRow = mysqlHelper.queryOnlyRow("select name, content from artifact where id='" + str(toAid) + "'")
        fromVec = processObj.doc2VecList(fromArtRow["content"])
        toVec = processObj.doc2VecList(toArtRow["content"])
        semantic_sim = processObj.sim_S_T(fromVec, toVec, str(fromAid), str(toAid))

        jaccard_sim = processObj.calcJaccardCoefficient(fromVec, toVec)

        idf_sum = 0
        for vec in toVec:
            if idfDic.has_key(vec[0]):
                idf_sum += float(idfDic[vec[0]])

        keyword_num = 0
        proportionRow = mysqlHelper.queryOnlyRow("select keywords from proportion where filename='" + str(toAid) + "'")
        if (proportionRow is not None) and (len(proportionRow) > 0):
            keywordsArr = proportionRow["keywords"].split()
            keyword_num = len(keywordsArr)

        text_len = len(toVec)

        dataSource = {"pid": str(datasetDic[curSet["dataset"]]),
                      "from_class": str(classDic[curSet["from_class"]]), "to_class": str(classDic[curSet["to_class"]]),
                      "from_aid": str(fromAid), "to_aid": str(toAid),
                      "semantic_sim": str(semantic_sim), "jaccard_sim": str(jaccard_sim), "idf_sum": str(idf_sum),
                      "keyword_num": str(keyword_num), "text_len": str(text_len),
                      "label": str(1)}
        mysqlHelper.insert("feature", dataSource)

    for negRow in negAnswers:
        fromAid = negRow['from_aid']
        toAid = negRow['to_aid']
        fromArtRow = mysqlHelper.queryOnlyRow("select name, content from artifact where id='" + str(fromAid) + "'")
        toArtRow = mysqlHelper.queryOnlyRow("select name, content from artifact where id='" + str(toAid) + "'")
        fromVec = processObj.doc2VecList(fromArtRow["content"])
        toVec = processObj.doc2VecList(toArtRow["content"])
        semantic_sim = processObj.sim_S_T(fromVec, toVec, fromAid, toAid)

        jaccard_sim = processObj.calcJaccardCoefficient(fromVec, toVec)

        idf_sum = 0
        for vec in toVec:
            if idfDic.has_key(vec[0]):
                idf_sum += float(idfDic[vec[0]])

        keyword_num = 0
        proportionRow = mysqlHelper.queryOnlyRow("select keywords from proportion where filename='" + str(toAid) + "'")
        if (proportionRow is not None) and (len(proportionRow) > 0):
            keywordsArr = proportionRow["keywords"].split()
            keyword_num = len(keywordsArr)

        text_len = len(toVec)

        dataSource = {"pid": str(datasetDic[curSet["dataset"]]),
                      "from_class": str(classDic[curSet["from_class"]]), "to_class": str(classDic[curSet["to_class"]]),
                      "from_aid": str(fromAid), "to_aid": str(toAid),
                      "semantic_sim": str(semantic_sim), "jaccard_sim": str(jaccard_sim), "idf_sum": str(idf_sum),
                      "keyword_num": str(keyword_num), "text_len": str(text_len),
                      "label": str(-1)}
        mysqlHelper.insert("feature", dataSource)

    print "共录入 " + str(len(posAnswers) + len(negAnswers)) + " 个链接"
    print "完成 " + curSet["dataset"] + " 数据集的 " + curSet['from_class'] + " 到 " + curSet['to_class']


'''
    将feature改成svm输入的格式
'''
def modifyForSVMInput():
    print "代码已经完成其作用，不可重复执行！！！"
    return 0
    datasetDic = {
        "CM1": 4, "eTOUR": 5, "GANTT": 6, "iTrust": 7, "EasyClinic": 8
    }
    classDic = {
        "系统需求": 1, "高级需求": 2, "低级需求": 3, "设计": 4, "源代码": 8, "用例": 9, "测试用例": 10, "UML交互": 11, "类描述": 12, "代码类": 13
    }

    curSet = {
        "dataset": "CM1",
        "from_class": "高级需求",
        "to_class": "低级需求",
        "filepath": "./svm/CM1.train"
    }

    featureRows = mysqlHelper.queryAll("select semantic_sim, jaccard_sim, idf_sum, keyword_num, text_len, label from feature where pid='"+str(datasetDic[curSet["dataset"]])+"' and from_class='"+str(classDic[curSet["from_class"]])+"' and to_class='"+str(classDic[curSet["to_class"]])+"'")

    with open(curSet["filepath"], 'w') as fp:
        for row in featureRows:
            fp.write(str(row['semantic_sim']) + "\t" + str(row['jaccard_sim']) + "\t" + str(row['idf_sum']) + "\t" + str(
                row['keyword_num']) + "\t" + str(row['text_len']) + "\t" + str(row['label']) + "\n")

    print "完成 " + curSet["dataset"] + " 数据集的 " + curSet['from_class'] + " 到 " + curSet['to_class']


def decimalHandle(strNum):
    return ("%.10f" % float(strNum))

'''
    将ir_feature改成svm输入的格式
'''
def modifyForIRSVMInput():
    print "代码已经完成其作用，不可重复执行！！！"
    return 0
    datasetDic = {
        "CM1": 4, "eTOUR": 5, "GANTT": 6, "iTrust": 7, "EasyClinic": 8
    }
    classDic = {
        "系统需求": 1, "高级需求": 2, "低级需求": 3, "设计": 4, "源代码": 8, "用例": 9, "测试用例": 10, "UML交互": 11, "类描述": 12, "代码类": 13
    }

    curSet = {
        "dataset": "CM1",
        "from_class": "高级需求",
        "to_class": "低级需求",
        "filepath": "./svm/CM1.train"
    }

    featureRows = mysqlHelper.queryAll("select semantic_sim, jaccard_sim, idf_sum, keyword_num, text_len, label, query_num, tao_val from ir_feature where pid='"+str(datasetDic[curSet["dataset"]])+"' and from_class='"+str(classDic[curSet["from_class"]])+"' and to_class='"+str(classDic[curSet["to_class"]])+"'")

    with open(curSet["filepath"], 'w') as fp:
        for row in featureRows:
            fp.write(decimalHandle(row['semantic_sim']) + "\t" + decimalHandle(row['jaccard_sim']) + "\t" + decimalHandle(row['idf_sum']) + "\t" + decimalHandle(
                row['keyword_num']) + "\t" + decimalHandle(row['text_len']) + "\t" + decimalHandle(row['label']) +  "\t" + decimalHandle(row['query_num']) +  "\t" + decimalHandle(row['tao_val']) +  "\n")

    print "完成 " + curSet["dataset"] + " 数据集的 " + curSet['from_class'] + " 到 " + curSet['to_class']



'''
    自定义sign函数
'''
def sign(i):
    if (i>=0):
        return 1
    else:
        return -1

'''
    生成IR SVM输入
'''
def generateIRSVMInput():
    datasetDic = {
        "CM1": 4, "eTOUR": 5, "GANTT": 6, "iTrust": 7, "EasyClinic": 8
    }
    classDic = {
        "系统需求": 1, "高级需求": 2, "低级需求": 3, "设计": 4, "源代码": 8, "用例": 9, "测试用例": 10, "UML交互": 11, "类描述": 12, "代码类": 13
    }

    curSet = {
        "dataset": "CM1",
        "from_class": "高级需求",
        "to_class": "低级需求",
        "filepath": "./svm/CM1.train"
    }
    featureRows = mysqlHelper.queryAll(
        "select from_aid, to_aid, semantic_sim, jaccard_sim, idf_sum, keyword_num, text_len from feature where pid='" + str(
            datasetDic[curSet["dataset"]]) + "' and from_class='" + str(
            classDic[curSet["from_class"]]) + "' and to_class='" + str(classDic[curSet["to_class"]]) + "' and label=1 order by semantic_sim desc")

    # 设置以from_aid为键的字典
    arrBasedFromAid = {}
    for row in featureRows:
        from_aid = row["from_aid"]
        if not arrBasedFromAid.has_key(from_aid):
            arrBasedFromAid[from_aid] = []
        arrBasedFromAid[from_aid].append(row)

    resultArr = []
    for key in arrBasedFromAid.keys():
        listItem = arrBasedFromAid[key]
        if len(listItem) < 2:
            continue
        i = 0
        while i < len(listItem):
            j = i + 1
            while j < len(listItem):
                first = listItem[i]
                second = listItem[j]
                xi = [float(first["semantic_sim"])-float(second["semantic_sim"]),float(first["jaccard_sim"])-float(second["jaccard_sim"]),float(first["idf_sum"])-float(second["idf_sum"]),float(first["keyword_num"])-float(second["keyword_num"]),float(first["text_len"])-float(second["text_len"])]
                xj = [float(second["semantic_sim"])-float(first["semantic_sim"]),float(second["jaccard_sim"])-float(first["jaccard_sim"]),float(second["idf_sum"])-float(first["idf_sum"]),float(second["keyword_num"])-float(first["keyword_num"]),float(second["text_len"])-float(first["text_len"])]

                tao_val = max(float(first["semantic_sim"]), float(second["semantic_sim"])) * abs(float(first["semantic_sim"])-float(second["semantic_sim"]))

                dataSource =  {"pid": str(datasetDic[curSet["dataset"]]),
                      "from_class": str(classDic[curSet["from_class"]]), "to_class": str(classDic[curSet["to_class"]]),
                      "query_aid":str(key),
                      "first_aid": str(first['to_aid']), "second_aid": str(second['to_aid']),
                      "semantic_sim": str(xi[0]), "jaccard_sim": str(xi[1]), "idf_sum": str(xi[2]),
                      "keyword_num": str(xi[3]), "text_len": str(xi[4]),
                      "query_num": str(len(listItem)), "tao_val":str(tao_val),
                      "label": str(sign(xi[0]))}
                mysqlHelper.insert("ir_feature", dataSource)
                dataSource2 = {"pid": str(datasetDic[curSet["dataset"]]),
                              "from_class": str(classDic[curSet["from_class"]]),
                              "query_aid": str(key),
                              "to_class": str(classDic[curSet["to_class"]]),
                              "first_aid": str(second['to_aid']), "second_aid": str(first['to_aid']),
                              "semantic_sim": str(xj[0]), "jaccard_sim": str(xj[1]), "idf_sum": str(xj[2]),
                              "keyword_num": str(xj[3]), "text_len": str(xj[4]),
                              "query_num": str(len(listItem)), "tao_val": str(tao_val),
                              "label": str(sign(xj[0]))}
                mysqlHelper.insert("ir_feature", dataSource2)
                j += 1
            i += 1



    print "完成 " + curSet["dataset"] + " 数据集的 " + curSet['from_class'] + " 到 " + curSet['to_class']



if __name__=='__main__':
    print "aaa"
    # modifyForIRSVMInput()
