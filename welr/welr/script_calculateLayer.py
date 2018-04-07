#! -*- coding:utf-8 -*-

# 计算

import sys
from processLayer import *
import simplejson as json

if len(sys.argv) < 4:
    print "缺少参数: 应该带四个参数 dataset, sourceKey, sourceStr, targetData"
    exit(1)

dataset = sys.argv[1]
sourceKey = sys.argv[2]
sourceStr = sys.argv[3]
targetData = sys.argv[4]

targetDic = json.loads(targetData)
pro = ProcessLayer(dataset)
pro.init()
keyList, simList = pro.handle(sourceKey, sourceStr, targetDic)
print keyList, simList
