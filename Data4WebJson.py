#!/usr/bin/env python
# coding: utf-8

# 导入必要的库
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 设置matplotlib以支持中文显示
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import xgboost as xgb
import sklearn
# 设置pandas显示选项
pd.options.display.max_columns = None
pd.options.display.max_rows = 20

# 读取CSV文件内容，筛选出label为1的数据
df = pd.read_csv('output_data.csv')
df = df[df["label"]==1]

# 读取JSON文件
import json
json_path = r"./数据大屏/big_screen/corp.json"
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 获取JSON文件中用于Echart1的数据
echart1 = data["echart1_data"]

# 计算df中"type_1_name"字段前9个最常出现的值及其计数
value_counts = df["type_1_name"].value_counts()[:9]
# 将计数结果转换为Echart1所需的格式
echart1['data'] = [{'name': index, 'value': count} for index, count in value_counts.items()]

# 获取JSON文件中用于Echart2的数据
echart2 = data["echart2_data"]

# 计算df中"county_id"字段前10个最常出现的值及其计数
value_counts = df["county_id"].value_counts()[:10]
# 将计数结果转换为Echart2所需的格式
echart2['data'] = [{'name': index, 'value': count} for index, count in value_counts.items()]

# 获取JSON文件中用于Echart3_1的数据
echart3_1 = data["echarts3_1_data"]

# 计算df中"brand_name"字段所有值的计数
value_counts = df["brand_name"].value_counts()
# 将计数结果转换为Echart3_1所需的格式
echart3_1['data'] = [{'name': index, 'value': count} for index, count in value_counts.items()]

# 获取JSON文件中用于Echart3_2的数据
echart3_2 = data["echarts3_2_data"]

# 计算df中"old_flag"字段所有值的计数
value_counts = df["old_flag"].value_counts()
# 映射0和1到"老人机"和"智能机"
name_mapping = {0: "老人机", 1: "智能机"}
value_counts.index = value_counts.index.map(name_mapping)
# 将映射后的计数结果转换为Echart3_2所需的格式
echart3_2['data'] = [{'name': index, 'value': count} for index, count in value_counts.items()]

# 获取JSON文件中用于Echart3_3的数据
echart3_3 = data["echarts3_3_data"]

# 计算df中"user_flag"字段所有值的计数
value_counts = df["user_flag"].value_counts()
# 映射1, 2, 3到"个人", "政企", "校园"
name_mapping = {1: "个人", 2: "政企", 3: "校园"}
value_counts.index = value_counts.index.map(name_mapping)
# 将映射后的计数结果转换为Echart3_3所需的格式
echart3_3['data'] = [{'name': index, 'value': count} for index, count in value_counts.items()]

# 获取JSON文件中用于Echart4的数据
echart4 = data["echart4_data"]

# 计算工作日每小时平均驻留时长
work_averages = [df[f'work_time_{i}'].mean() for i in range(24)]
# 计算休息日每小时平均驻留时长
stay_averages = [df[f'stay_time_{i}'].mean() for i in range(24)]

# 更新Echart4的标题和数据
echart4["title"] = "平均驻留时长"
echart4["data"][0]["name"] = "工作日"
echart4["data"][0]["value"] = work_averages
echart4["data"][1]["name"] = "休息日"
echart4["data"][1]["value"] = stay_averages
echart4["xAxis"] = [str(i) for i in range(24)]

# 获取JSON文件中用于Echart5的数据
echart5 = data["echart5_data"]

# 计算df中"iden_city_id"字段所有值的计数
value_counts = df["iden_city_id"].value_counts()

# 将计数结果转换为Echart5所需的格式
echart5['data'] = [{'name': int(index), 'value': count} for index, count in value_counts.items()]

# 获取JSON文件中用于Echart6的数据
echart6 = data["echart6_data"]

# 将更新后的数据写回JSON文件
with open(json_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)