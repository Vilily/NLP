""" matplotlib绘制直方图
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/12/16
@github: https://github.com/baowj-678
"""
from typing import Optional
import matplotlib.pyplot as plt

def draw_hist(x, bins: int, title: Optional[str], color: Optional[str] = 'red'):
    """ 绘制直方图
    @param:\n
    :x [list]原始数据\n
    :bins [int]划分的块数\n
    """
    if title is not None:
        plt.title(title)
    plt.hist(x=x, 
             bins=bins,
             color='r')
    plt.show()


if __name__ == "__main__":
    data = [1, 1, 2, 2, 2, 3, 3, 4]
    draw_hist(x=data,
              bins=2,
              title=None)