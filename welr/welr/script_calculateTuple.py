#! -*- coding:utf-8 -*-

# 计算

import sys
from processTuple import *
import simplejson as json

if len(sys.argv) < 5:
    print "缺少参数: 应该带五个参数 dataset, sourceKey, sourceStr, targetKey, targetStr"
    exit(1)

dataset = sys.argv[1]
sourceKey = sys.argv[2]
sourceStr = sys.argv[3]
targetKey = sys.argv[4]
targetStr = sys.argv[5]

pro = ProcessTuple(dataset)
pro.init()
sim = pro.handle(sourceKey, sourceStr, targetKey, targetStr)
print sim
