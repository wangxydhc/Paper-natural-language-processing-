#! -*- coding:utf-8 -*-

# 计算

import sys
from process import *
import simplejson as json

if len(sys.argv) < 4:
    print "缺少参数: 应该带四个参数 dataset, sourceKey, sourceStr, targetDicStr"
    exit(1)

dataset = sys.argv[1]
sourceKey = sys.argv[2]
sourceStr = sys.argv[3]
targetDicStr = sys.argv[4]

targetDic = json.loads(targetDicStr)
pro = Process(dataset)
pro.init()
pro.handle(sourceKey, sourceStr, targetDic)
