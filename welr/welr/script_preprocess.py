#! -*- coding:utf-8 -*-

# 数据导入后，进行数据预处理

import sys

from removeStopwords import *
from idf_calc import *
from proportion_calc import *
from qe_calc import *
from process import *

if len(sys.argv) < 3:
    print "缺少参数: 应该带两个参数 dataset 和 json"
    exit(1)

dataset = sys.argv[1]
inputStr = sys.argv[2]

# 1. 去停用词
rmStopwords = RemoveStopwords(inputStr)
newInputStr = rmStopwords.handle()

# print newInputStr
#
# exit()

# 2. 计算idf和tfidf
idfCalc = IDFCALC(dataset, newInputStr)
idfCalc.calcIDF()
idfCalc.handle()

# 3. 抽取关键词
proCalc = ProportionCALC(dataset, newInputStr)
proCalc.handle()

# 4. 通过w2vModel获取相近的词
qeCalc = QECALC(dataset)
qeCalc.handle()

# 5. 计算其他特征（关键字数量， 文本长度）

