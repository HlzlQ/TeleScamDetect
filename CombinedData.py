#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import sklearn
# 设置Pandas显示选项，确保数据框中的所有列都能显示出来
pd.options.display.max_columns = None
pd.options.display.max_rows = 20

# 设置Matplotlib字体，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用于正确显示负号

# 读取原始数据文件
df = pd.read_csv('output_data.csv')

# 过滤数据，只保留标签为1的数据行
df = df[df["label"] == 1]

# 读取含有缺失值的数据文件
df_nan = pd.read_csv('data_na.csv')

# 将两个数据集合并，创建一个新的DataFrame
df_combined = pd.concat([df, df_nan], ignore_index=True)

# 对合并后的数据集进行随机打乱处理，以保证数据的随机性
df_shuffled = df_combined.sample(frac=1).reset_index(drop=True)

# 将处理后的数据保存到新的CSV文件中
df_shuffled.to_csv('all_data_with_nan_one.csv', index=False)