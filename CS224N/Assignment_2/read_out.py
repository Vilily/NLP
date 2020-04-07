import numpy as np
import matplotlib.pyplot as plt


filename = 'nohup.out'
fp = None
try:
    fp = open(filename, 'r')
    print("%s 文件成功打开" % filename)
except IOError:
    print("error")

content = fp.read()
content = content.split('\n')
inter2num = np.zeros([len(content), 2])
for i in range(len(content) - 1):
    piece = content[i].split(':')
    inter2num[i, 0] = int(piece[0].split(' ')[1])
    inter2num[i, 1] = float(piece[1].split(' ')[1])
plt.scatter(inter2num[:, 0], inter2num[:, 1])
plt.show()
fp.close()