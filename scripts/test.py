
import numpy as np
import pandas as pd
import re

# with open('url35.txt', 'r') as f1:
#     list = f1.readlines()
# bmi = 0
# for i in range(0, len(list)):
#     list[i] = list[i].rstrip('\n')
#     r1 = re.search('http://192.168.3.200:5020/bmi/(.*)',list[i])
#     list[i] = r1.group(1)
#     r2 = re.search('(\d+).*',list[i])
#     bmi = r2.group(1)
#     r3 = re.search('/(.*)-face.jpg',list[i])
#     md5 = r3.group(1)
#     list[i] = md5
#     # print(bmi)
#     # print(md5)with open
# dic = {bmi:list}
# df = pd.DataFrame(dic)
# df.to_csv(bmi+'md5.csv',index=False)

# for tag in range(35,41):
#     df = pd.read_csv(str(tag)+'md5.csv')
#     print(df.count())