import re
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def getData(file, patern):
    with open(file, encoding='utf-8') as file:
        data = file.read()
    data = re.findall(patern, data, re.S)
    data = [float(num) for num in data]
    return data

def plot(data, title, xlabel, ylabel, file):
    """ 训练过程图
    """
    font={'family':'SimHei',
          'style':'italic',
          'weight':'normal',
          'color':'black',
          'size':20
          }
    plt.title(title, font)
    plt.ylabel(ylabel, font)
    plt.xlabel(xlabel, fontdict=font)
    plt.plot(data)
    plt.grid()
    plt.show()
    # plt.savefig(file, dpi=500, bbox_inches = 'tight')

if __name__ == "__main__":
    # data = getData('nohup.out', r'总数:10000, 测试结果accu: ([0-9\.]*)\n')
    # plot(data, 'ESIM训练准确率', 'epoch', '准确率')
    data = getData('nohup.out', r'loss: ([0-9\.]*)\n')
    plot(data, 'ESIM训练loss', 'itor', 'loss', 'readme_pic/train_loss.png')
