#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# 设置 Matplotlib 字体，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用于正确显示负号

# 设置 Pandas 显示选项
pd.options.display.max_columns = None
pd.options.display.max_rows = 20

# 将 JSON 文件转换为 CSV 文件
def json_to_csv(json_file, csv_file):
    """
    将 JSON 文件转换为 CSV 文件。

    :param json_file: 输入的 JSON 文件路径
    :param csv_file: 输出的 CSV 文件路径
    """
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    csv_content = data['content']
    with open(csv_file, 'w', encoding='utf-8') as file:
        file.write(csv_content)

# 读取字段对应表
def read_field_description(excel_file):
    """
    读取字段对应表，并去除多余的列。

    :param excel_file: 输入的 Excel 文件路径
    :return: 处理后的 DataFrame
    """
    tab_id_describe = pd.read_excel(excel_file)
    tab_id_describe = tab_id_describe.drop(tab_id_describe.columns[0], axis=1)
    return tab_id_describe

# 读取并处理 CSV 文件
def process_csv(csv_file):
    """
    读取并处理 CSV 文件，过滤出标签为1的样本。

    :param csv_file: 输入的 CSV 文件路径
    :return: 处理后的 DataFrame
    """
    df = pd.read_csv(csv_file)
    df = df[df["label"] == 1]
    return df

# 绘制年龄分布
def plot_age_distribution(df):
    """
    绘制年龄分布图。

    :param df: 数据 DataFrame
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['nat_age_year'], bins=100, kde=True, color='blue', kde_kws={'bw_adjust': 4})
    plt.title('年龄分布')
    plt.xlabel('年龄 (年)')
    plt.ylabel('频率')
    plt.show()

# 绘制性别分布
def plot_gender_distribution(df):
    """
    绘制性别分布图。

    :param df: 数据 DataFrame
    """
    sns.countplot(x='sex_id', data=df)
    plt.title('性别分布')
    plt.xlabel('类别')
    plt.ylabel('频率')
    plt.xticks(ticks=[0, 1, 2], labels=['未知', '男', '女'])
    plt.show()

# 绘制城市分布
def plot_city_distribution(df):
    """
    绘制城市分布图。

    :param df: 数据 DataFrame
    """
    sns.countplot(x='city_id', data=df)
    plt.show()

# 绘制区县分布
def plot_county_distribution(df):
    """
    绘制区县分布图。

    :param df: 数据 DataFrame
    """
    sns.countplot(x='county_id', data=df)
    plt.xticks(rotation=45)
    plt.show()

# 绘制单位分布
def plot_unit_distribution(df):
    """
    绘制单位分布图。

    :param df: 数据 DataFrame
    """
    sns.countplot(x='unit_id', data=df)
    plt.xticks(rotation=90)
    plt.show()

# 绘制区县和单位的联合分布
def plot_county_unit_distribution(df):
    """
    绘制区县和单位的联合分布图。

    :param df: 数据 DataFrame
    """
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df, x='county_id', hue='unit_id')
    plt.xticks(rotation=90)
    plt.title('区县和单位的数据分布')
    plt.xlabel('区县 ID')
    plt.ylabel('数量')
    plt.legend(title='单位 ID', loc='upper right')
    plt.show()

# 绘制身份证城市分布
def plot_iden_city_distribution(df):
    """
    绘制身份证城市分布图。

    :param df: 数据 DataFrame
    """
    df['iden_city_id'] = df['iden_city_id'].fillna(-1).astype(int)  # -1代表未知
    sns.countplot(x='iden_city_id', data=df)
    plt.xticks(rotation=45)
    plt.show()

# 绘制号码数量分布
def plot_num_distribution(df):
    """
    绘制号码数量分布图。

    :param df: 数据 DataFrame
    """
    sns.countplot(x='num', data=df)
    plt.yscale('log')  # 设置 y 轴为对数尺度
    plt.xticks(rotation=45)
    plt.show()

# 绘制老人机标志分布
def plot_old_flag_distribution(df):
    """
    绘制老人机标志分布图。

    :param df: 数据 DataFrame
    """
    sns.countplot(x='old_flag', data=df)  # 0代表不是老人机
    plt.xticks(rotation=45)
    plt.show()

# 绘制用户标志分布
def plot_user_flag_distribution(df):
    """
    绘制用户标志分布图。

    :param df: 数据 DataFrame
    """
    sns.countplot(x='user_flag', data=df)
    plt.xticks(rotation=45)
    plt.show()

# 绘制月费分布
def plot_month_fee_distribution(df):
    """
    绘制月费分布图。

    :param df: 数据 DataFrame
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['month_fee'], bins=20, kde=True, color='blue', kde_kws={'bw_adjust': 4})
    plt.show()

# 绘制品牌名称分布
def plot_brand_name_distribution(df):
    """
    绘制品牌名称分布图。

    :param df: 数据 DataFrame
    """
    sns.countplot(x='brand_name', data=df)
    plt.xticks(rotation=45)
    plt.show()

# 绘制通话时长分布
def plot_call_duration_distribution(df):
    """
    绘制通话时长分布图。

    :param df: 数据 DataFrame
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['call_duration_m'], bins=200, kde=True, color='blue', kde_kws={'bw_adjust': 4})
    plt.show()

# 绘制流量分布
def plot_flow_distribution(df):
    """
    绘制流量分布图。

    :param df: 数据 DataFrame
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['flow_all'], bins=200, kde=True, color='blue', kde_kws={'bw_adjust': 4})
    plt.show()

# 绘制短信数量分布
def plot_sms_count_distribution(df):
    """
    绘制短信数量分布图。

    :param df: 数据 DataFrame
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sms_count'], bins=10, kde=True, color='blue', kde_kws={'bw_adjust': 4})
    plt.show()

# 绘制区域类型分布
def plot_zonetype_distribution(df):
    """
    绘制区域类型分布图。

    :param df: 数据 DataFrame
    """
    sns.countplot(x='zonetype_id', data=df)
    plt.xticks(rotation=45)
    plt.show()

# 绘制电视类型分布
def plot_htv_type_distribution(df):
    """
    绘制电视类型分布图。

    :param df: 数据 DataFrame
    """
    sns.countplot(x='htv_type', data=df)
    plt.xticks(rotation=45)
    plt.show()

# 绘制 GPRS 时长分布
def plot_gprs_duration_distribution(df):
    """
    绘制 GPRS 时长分布图。

    :param df: 数据 DataFrame
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['gprs_duration'], bins=100, kde=True, color='blue', kde_kws={'bw_adjust': 4})
    plt.show()

# 绘制来电次数和时长分布
def plot_call_in_distribution(df):
    """
    绘制来电次数和时长分布图。

    :param df: 数据 DataFrame
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['call_in_counts'], bins=100, kde=True, color='blue', kde_kws={'bw_adjust': 4})
    sns.histplot(df['call_in_duration'], bins=100, kde=True, color='red', kde_kws={'bw_adjust': 4})
    plt.xlim(0, 200)
    plt.show()

# 绘制去电次数和时长分布
def plot_call_out_distribution(df):
    """
    绘制去电次数和时长分布图。

    :param df: 数据 DataFrame
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['call_out_counts'], bins=1000, kde=True, color='blue', kde_kws={'bw_adjust': 4})
    sns.histplot(df['call_out_duration'], bins=1000, kde=True, color='red', kde_kws={'bw_adjust': 4})
    plt.xlim(0, 50)
    plt.show()

# 绘制其他号码分布
def plot_other_num_distribution(df):
    """
    绘制其他号码分布图。

    :param df: 数据 DataFrame
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['other_num'], bins=100, kde=True, color='blue', kde_kws={'bw_adjust': 4})
    plt.xlim(0, 200)
    plt.show()

# 绘制工作时间序列图
def plot_work_time_series(df, num_samples=10):
    """
    绘制工作时间序列图。

    :param df: 数据 DataFrame
    :param num_samples: 要绘制的样本数量
    """
    columns = df.columns
    for index in range(num_samples):
        hours = list(range(24))
        lac_values = df.loc[index, [f'work_lac_{i}' for i in hours]]
        cell_values = df.loc[index, [f'work_cell_{i}' for i in hours]]
        time_values = df.loc[index, [f'work_time_{i}' for i in hours]]
        plt.figure(figsize=(12, 4))
        
        # 绘制 work_lac 折线图
        plt.subplot(1, 3, 1)
        plt.plot(hours, lac_values, marker='o')
        plt.title(f'样本 {index+1} - work_lac')
        plt.xlabel('小时')
        plt.ylabel('LAC 值')
        
        # 绘制 work_cell 折线图
        plt.subplot(1, 3, 2)
        plt.plot(hours, cell_values, marker='o')
        plt.title(f'样本 {index+1} - work_cell')
        plt.xlabel('小时')
        plt.ylabel('Cell 值')
        
        # 绘制 work_time 折线图
        plt.subplot(1, 3, 3)
        plt.plot(hours, time_values, marker='o')
        plt.title(f'样本 {index+1} - work_time')
        plt.xlabel('小时')
        plt.ylabel('停留时长')
        plt.tight_layout()
        plt.show()
        
# 计算所有时刻的差值
def calculate_all_differences(df):
    """
    计算所有时刻的差值，并绘制差值分布图。

    :param df: 数据 DataFrame
    """
    all_differences = []
    # 遍历每个样本计算所有时刻的差值
    for index, row in df.iterrows():
        # 遍历每个时刻 (0 到 22)，计算与下一个时刻的差值
        for t in range(23):  # 23是因为要和下一个时刻做差值
            diff = abs(row[f'work_time_{t}'] - row[f'work_time_{t+1}'])
            all_differences.append(diff)
    
    # 将所有差值转换为 DataFrame 以便分析
    diff_df = pd.DataFrame(all_differences, columns=['time_diff'])
    
    # 绘制差值分布图
    plt.figure(figsize=(8, 5))
    plt.hist(diff_df['time_diff'], bins=15, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Absolute Differences between Consecutive Hours')
    plt.xlabel('Absolute Difference')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# 计算每个时间对的差值并绘制分布图
def plot_hourly_differences(df):
    """
    计算每个时间对的差值并绘制分布图。

    :param df: 数据 DataFrame
    """
    for t in range(23):  # 共有23个时间对，从 (0,1) 到 (22,23)
        # 计算当前时间对的差值并收集
        differences = abs(df[f'work_time_{t}'] - df[f'work_time_{t+1}'])
        
        # 绘制当前时间对的差值分布图
        plt.figure(figsize=(8, 5))
        plt.hist(differences, bins=100, alpha=0.7, edgecolor='black')
        plt.title(f'Distribution of Absolute Differences between Hour {t} and Hour {t+1}')
        plt.xlabel('Absolute Difference')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

# 计算 stay_time 的差值并绘制分布图
def plot_stay_time_differences(df):
    """
    计算 stay_time 的差值并绘制分布图。

    :param df: 数据 DataFrame
    """
    for t in range(23):  # 共有23个时间对，从 (0,1) 到 (22,23)
        # 计算当前时间对的差值并收集
        differences = abs(df[f'stay_time_{t}'] - df[f'stay_time_{t+1}'])
        
        # 绘制当前时间对的差值分布图
        plt.figure(figsize=(8, 5))
        plt.hist(differences, bins=100, alpha=0.7, edgecolor='black')
        plt.title(f'Distribution of Absolute Differences between Hour {t} and Hour {t+1}')
        plt.xlabel('Absolute Difference')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

# 计算 work_time 和 stay_time 的差值并绘制分布图
def plot_work_stay_differences(df):
    """
    计算 work_time 和 stay_time 的差值并绘制分布图。

    :param df: 数据 DataFrame
    """
    plt.figure(figsize=(8, 5))
    for t in range(24):  # 共有24个时间对，从 (0,1) 到 (23,0)
        # 计算当前时间对的差值并收集
        differences = abs(df[f'stay_time_{t}'] - df[f'work_time_{t}'])
        
        # 绘制当前时间对的差值分布图
        plt.hist(differences, bins=100, alpha=0.7, label=f"hour {t}")
    plt.title(f'Distribution of Absolute Differences between work Hour and stay Hour')
    plt.xlabel('Absolute Difference')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right', ncol=2)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# 计算每个类别的出现次数，并按降序排序
def plot_category_counts(df):
    """
    计算每个类别的出现次数，并按降序排序。

    :param df: 数据 DataFrame
    """
    for type_id in range(1, 6):
        order = df[f'type_{type_id}_name'].value_counts().index[:20]
        # 绘制 countplot，并按照从高到低的顺序排列
        sns.countplot(y=f'type_{type_id}_name', data=df, order=order)
        plt.yticks(fontsize=8)
        plt.show()

# 绘制 acce_flow_15 的分布图
def plot_acce_flow_distribution(df):
    """
    绘制 acce_flow_15 的分布图。

    :param df: 数据 DataFrame
    """
    plt.figure(figsize=(10, 6))  # 设置图形的大小
    sns.histplot(df['acce_flow_15'], bins=1000, kde=True, color='blue', kde_kws={'bw_adjust': 4})
    plt.show()

# 对小时数据进行预处理，然后快速FFT变换
def preprocess_and_fft(df):
    """
    对小时数据进行预处理，然后快速FFT变换。

    :param df: 数据 DataFrame
    """
    # 获取字段分组
    work_lac_cols = [f'work_lac_{i}' for i in range(24)]
    work_cell_cols = [f'work_cell_{i}' for i in range(24)]
    work_time_cols = [f'work_time_{i}' for i in range(24)]
    stay_lac_cols = [f'stay_lac_{i}' for i in range(24)]
    stay_cell_cols = [f'stay_cell_{i}' for i in range(24)]
    stay_time_cols = [f'stay_time_{i}' for i in range(24)]

    # 定义逐行处理函数
    def fill_nan(row):
        for columns in [work_lac_cols, work_cell_cols, work_time_cols,
                        stay_lac_cols, stay_cell_cols, stay_time_cols]:
            row[columns] = row[columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
        return row

    # 初始化 tqdm 进度条
    tqdm.pandas(desc="Processing rows")

    # 使用 apply 并行处理
    df = df.progress_apply(fill_nan, axis=1)

    # 保存到文件
    df.to_csv('hours_fillna_data.csv', index=False)

    # 读取处理后的数据
    hours_fillna_df = pd.read_csv('hours_fillna_data.csv')

    # 计算绝对差
    def calculate_absolute_diff(df, columns_prefix):
        # 提取指定的列
        columns = [f"{columns_prefix}_{i}" for i in range(24)]
        data = df[columns].values
        
        # 添加首列到末尾，使得 23 小时与 0 小时相邻
        cyclic_data = np.hstack([data, data[:, [0]]])
        
        # 计算相邻小时的绝对差
        diffs = np.abs(np.diff(cyclic_data, axis=1))
        
        # 返回包含差分的 DataFrame
        return pd.DataFrame(diffs, columns=[f"{columns_prefix}_diff_{i}" for i in range(24)])

    # 处理 work 和 stay 的 lac、cell 和 time 部分
    work_lac_diff = calculate_absolute_diff(hours_fillna_df, "work_lac")
    work_cell_diff = calculate_absolute_diff(hours_fillna_df, "work_cell")
    work_time_diff = calculate_absolute_diff(hours_fillna_df, "work_time")
    stay_lac_diff = calculate_absolute_diff(hours_fillna_df, "stay_lac")
    stay_cell_diff = calculate_absolute_diff(hours_fillna_df, "stay_cell")
    stay_time_diff = calculate_absolute_diff(hours_fillna_df, "stay_time")

    # 合并所有差分结果到新的 DataFrame
    diff_df = pd.concat([work_lac_diff, work_cell_diff, work_time_diff, 
                         stay_lac_diff, stay_cell_diff, stay_time_diff], axis=1)

    # 归一化
    def normalize_row(row):
        scaler = MinMaxScaler()
        for prefix in ["work_lac", "work_cell", "work_time", "stay_lac", "stay_cell", "stay_time"]:
            columns = [f"{prefix}_diff_{i}" for i in range(24)]
            # 对指定的24列差分结果进行 min-max 归一化
            row[columns] = scaler.fit_transform(row[columns].values.reshape(-1, 1)).flatten()
        return row

    # 使用 apply 加速并结合 tqdm 显示进度
    tqdm.pandas(desc="Normalizing rows")
    min_max_diff_df = diff_df.progress_apply(normalize_row, axis=1)

    # 选取特定列
    work_lac_diff_cols = [f'work_lac_diff_{i}' for i in range(24)]
    stay_lac_diff_cols = [f'stay_lac_diff_{i}' for i in range(24)]
    one_type_df = min_max_diff_df[work_lac_diff_cols + stay_lac_diff_cols]

    # 引入时间窗口 w=6
    data_array = one_type_df.values
    reshaped_array = data_array.reshape(-1, 8, 6)
    fft_result = np.fft.fft(reshaped_array, axis=2)
    # 取前 4 个元素（保留一半加1），去掉对称部分
    fft_magnitude = np.abs(fft_result[:, :, :4])
    # 取对数
    processed_data = np.log(fft_magnitude + 1e-10)  # 防止 log(0)

    # 计算样本的特征间距：在 24 个维度上的 2 范数
    sample_norms = np.linalg.norm(processed_data, axis=2)
    # 标准化处理数据
    scaler = StandardScaler()
    sample_norms_scaled = scaler.fit_transform(sample_norms)

    # 使用 KMeans 进行聚类
    n_clusters = 5  # 可以根据具体需求调整
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(sample_norms_scaled)

    # 可视化聚类结果
    pca = PCA(n_components=2)
    sample_pca = pca.fit_transform(sample_norms_scaled)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(sample_pca[:, 0], sample_pca[:, 1], c=clusters, cmap='viridis', s=1, alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('2D Visualization of Clusters using PCA')
    plt.show()
# 主程序
if __name__ == "__main__":
    # 将 JSON 文件转换为 CSV 文件
    json_to_csv('final_data.json', 'output_data.csv')
    
    # 读取字段对应表
    tab_id_describe = read_field_description("电信诈骗领域-dw_fraud_model_work_dm_sex1_0919-dw_fraud_model_work_dm_sex1_0919_20241016152635.xlsx")
    
    # 读取并处理 CSV 文件
    df = process_csv('output_data.csv')
    
    # 绘制各种分布图
    plot_age_distribution(df)
    plot_gender_distribution(df)
    plot_city_distribution(df)
    plot_county_distribution(df)
    plot_unit_distribution(df)
    plot_county_unit_distribution(df)
    plot_iden_city_distribution(df)
    plot_num_distribution(df)
    plot_old_flag_distribution(df)
    plot_user_flag_distribution(df)
    plot_month_fee_distribution(df)
    plot_brand_name_distribution(df)
    plot_call_duration_distribution(df)
    plot_flow_distribution(df)
    plot_sms_count_distribution(df)
    plot_zonetype_distribution(df)
    plot_htv_type_distribution(df)
    plot_gprs_duration_distribution(df)
    plot_call_in_distribution(df)
    plot_call_out_distribution(df)
    plot_other_num_distribution(df)
    
    # 绘制工作时间序列图
    plot_work_time_series(df)
    # 读取并处理 CSV 文件
    df = process_csv('output_data.csv')
    
    # 计算所有时刻的差值
    calculate_all_differences(df)
    
    # 计算每个时间对的差值并绘制分布图
    plot_hourly_differences(df)
    
    # 计算 stay_time 的差值并绘制分布图
    plot_stay_time_differences(df)
    
    # 计算 work_time 和 stay_time 的差值并绘制分布图
    plot_work_stay_differences(df)
    
    # 计算每个类别的出现次数，并按降序排序
    plot_category_counts(df)
    
    # 绘制 acce_flow_15 的分布图
    plot_acce_flow_distribution(df)
    
    # 对小时数据进行预处理，然后快速FFT变换
    preprocess_and_fft(df)