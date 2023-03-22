import pandas as pd
import numpy as np
import os
import inspect
import re
import time


#   模式	  可做操作   若文件不存在   是否覆盖
#     r	       读	      报错	      -
#    r+	     读写	      报错	      是
#     w        写	      创建	      是
#    w+	     读写	      创建	      是
#     a　　	   写	      创建	   追加写
#    a+      读写	      创建	   追加写

class Log_Recordor:
    def __init__(self, filename, mode='w', type='txt'):
        self.filename = filename
        self.mode = mode
        self.type = type
        self.file = open(self.filename, self.mode)
        self.file.close()

    def variate_to_file(self, data):
        if os.path.exists(self.filename):
            self.file = open(self.filename, 'a')
        else:
            self.file = open(self.filename, self.mode)
        # 输出到csv文件
        if self.type == 'csv':
            if isinstance(data, dict):
                record = pd.DataFrame(data)
                record.to_csv(self.filename)
        # 输出到txt文件
        if self.type == 'txt':
            if isinstance(data, str):
                self.file.write('logtime: %5s   :' % time.strftime('%Y-%m-%d %H:%M:%S'))
                self.file.write('%s\n' % data)
            else:
                self.file.write('**** ---------------------------------------- ****\n')
                self.file.write('logtime: %5s\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
                self.file.write('   type: %5s\n' % str(type(data)))
                self.file.write('   type: %5s\n' % str(varname(data)))
            # 如果数据类型为dict
            if isinstance(data, dict):
                for key in data.keys():
                    self.file.write('            ')
                    self.file.write('%5s: ' % key)
                    for v in data[key]:
                        self.file.write('%5s,' % str(v))
                    self.file.write('\n')
            # 如果数据类型为list
            if isinstance(data, list):
                for v in data:
                    self.file.write('%10s,' % str(v))
                self.file.write('\n')
            # 如果数据类型为narray
            if isinstance(data, np.ndarray):
                for v in data:
                    self.file.write('%10s,' % str(v))
                self.file.write('\n')
        self.file.write('\n')
        # 结束输出
        self.file.close()

    def log_to_file(self, *word):
        if os.path.exists(self.filename):
            self.file = open(self.filename, 'a')
        else:
            self.file = open(self.filename, self.mode)
        # 结束输出
        self.file.write('%5s:        ' % time.strftime('%Y-%m-%d %H:%M:%S'))
        for i in range(0,len(word)):
            self.file.write(str(word[i]))
        self.file.write('\n')
        # 结束输出
        self.file.close()


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)
