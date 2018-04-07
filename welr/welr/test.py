#! -*- coding:utf-8 -*-

from MySQLHelper import *
from common import Common
from process import Process
import random
import numpy as np
from svmProcess import *

process = SVMProcess()
# dataArr = [[1,1,1,1,1]]
# process.predict(dataArr)
process.trainAndSaveSV()