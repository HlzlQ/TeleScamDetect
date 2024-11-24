
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import time
import warnings

# 忽略特定类型的警告
warnings.filterwarnings('ignore', category=UserWarning)

# 还可以指定忽略包含特定文本的警告
warnings.filterwarnings('ignore', 'Glyph .* missing from current font.')

pd.options.display.max_columns = None
pd.options.display.max_rows = 20

def data_process(data):
    # 特征处理

    # 删除user_id，product_no，city_id	
    data = data.drop('user_id', axis=1)
    data = data.drop('product_no', axis=1)
    data = data.drop('city_id', axis=1)

    # county_id统一减去24000，众数填充
    data["county_id"] -=24000
    mode = data['county_id'].mode()[0]
    data["county_id"] = data["county_id"].fillna(mode)

    # 删除unit_id
    data = data.drop('unit_id', axis=1)

    # nat_age_year使用均值填充
    mean_age = data["nat_age_year"].mean()
    data["nat_age_year"] = data["nat_age_year"].fillna(mean_age)

    # sex_id使用众数填充
    mode_sex_id = data['sex_id'].mode()[0]
    data['sex_id'] = data['sex_id'].fillna(mode_sex_id)

    # iden_city_id，240本省映射为0，其他均为1
    mapping = {240: 0}
    data['iden_city_id'] = data['iden_city_id'].map(mapping).fillna(1).astype(int)
    
    data = data.drop('num', axis=1)

    # brand_name映射到0，1，2，
    mapping = {
        '全球通': 0,
        '动感地带': 1,
        '神州行': 2
    }
    data['brand_name'] = data['brand_name'].map(mapping)

    # month_fee归一化
    data['month_fee'] = (data['month_fee'] - data['month_fee'].min()) / (data['month_fee'].max() - data['month_fee'].min())

    # call_duration_m归一化
    data['call_duration_m'] = (data['call_duration_m'] - data['call_duration_m'].min()) / (data['call_duration_m'].max() - data['call_duration_m'].min())

    # flow_all归一化
    data['flow_all'] = (data['flow_all'] - data['flow_all'].min()) / (data['flow_all'].max() - data['flow_all'].min())

    # sms_count归一化
    data['sms_count'] = (data['sms_count'] - data['sms_count'].min()) / (data['sms_count'].max() - data['sms_count'].min())

    # call_counts*call_duration，并归一化
    data["call_counts_duration"] = data['call_counts'] * data["call_duration"]
    data['call_counts_duration'] = (data['call_counts_duration'] - data['call_counts_duration'].min()) / (data['call_counts_duration'].max() - data['call_counts_duration'].min())
    mean_counts = data['call_counts_duration'].mean()
    data["call_counts_duration"] = data["call_counts_duration"].fillna(mean_counts)
    data = data.drop('call_counts', axis=1)
    data = data.drop('call_duration', axis=1)

    # gprs_duration归一化
    data['gprs_duration'] = (data['gprs_duration'] - data['gprs_duration'].min()) / (data['gprs_duration'].max() - data['gprs_duration'].min())
    mean_counts = data['gprs_duration'].mean()
    data["gprs_duration"] = data["gprs_duration"].fillna(mean_counts)

    # call_in_counts*call_in_duration
    data["call_in_counts_duration"] = data['call_in_counts'] * data["call_in_duration"]
    data['call_in_counts_duration'] = (data['call_in_counts_duration'] - data['call_in_counts_duration'].min()) / (data['call_in_counts_duration'].max() - data['call_in_counts_duration'].min())
    mean_counts = data['call_in_counts_duration'].mean()
    data["call_in_counts_duration"] = data["call_in_counts_duration"].fillna(mean_counts)
    data = data.drop('call_in_counts', axis=1)
    data = data.drop('call_in_duration', axis=1)

    # call_out_counts*call_out_duration
    data["call_out_counts_duration"] = data['call_out_counts'] * data["call_out_duration"]
    data['call_out_counts_duration'] = (data['call_out_counts_duration'] - data['call_out_counts_duration'].min()) / (data['call_out_counts_duration'].max() - data['call_out_counts_duration'].min())
    mean_counts = data['call_out_counts_duration'].mean()
    data["call_out_counts_duration"] = data["call_out_counts_duration"].fillna(mean_counts)
    data = data.drop('call_out_counts', axis=1)
    data = data.drop('call_out_duration', axis=1)

    # other_num归一化
    data['other_num'] = (data['other_num'] - data['other_num'].min()) / (data['other_num'].max() - data['other_num'].min())
    mean_counts = data['other_num'].mean()
    data["other_num"] = data["other_num"].fillna(mean_counts)

    # has_nan = data["other_num"].isna().any()
    # 获取开始列和结束列的索引
    start_index = data.columns.get_loc('other_num')  # 获取开始列的索引

    end_index = data.columns.get_loc("type_1_name")      # 获取结束列的索引

    # 确保开始索引小于结束索引
    if start_index < end_index:

        # 删除开始列和结束列之间的所有列
        data = data.drop(data.columns[start_index + 1:end_index], axis=1)
    
    brand_counts = data['type_1_name'].value_counts()
    small_brands = brand_counts[brand_counts < 2381].index
    brand_to_number = {}
    counter = 2
    for brand in data['type_1_name']:
        if brand in small_brands:
            brand_to_number[brand] = 1
        else:
            brand_to_number[brand] = counter
            counter += 1

    # 应用编码
    data['type_1_name'] = data['type_1_name'].map(brand_to_number)
    data['type_1_name'] = data['type_1_name'].fillna(0)
    
    brand_counts = data['type_2_name'].value_counts()
    small_brands = brand_counts[brand_counts < 9867].index
    brand_to_number = {}
    counter = 2
    for brand in data['type_2_name']:
        if brand in small_brands:
            brand_to_number[brand] = 1
        else:
            brand_to_number[brand] = counter
            counter += 1

    # 应用编码
    data['type_2_name'] = data['type_2_name'].map(brand_to_number)
    data['type_2_name'] = data['type_2_name'].fillna(0)
    
    brand_counts = data['type_3_name'].value_counts()
    small_brands = brand_counts[brand_counts < 12341].index
    brand_to_number = {}
    counter = 2
    for brand in data['type_3_name']:
        if brand in small_brands:
            brand_to_number[brand] = 1
        else:
            brand_to_number[brand] = counter
            counter += 1

    # 应用编码
    data['type_3_name'] = data['type_3_name'].map(brand_to_number)
    data['type_3_name'] = data['type_3_name'].fillna(0)
    
    brand_counts = data['type_4_name'].value_counts()
    small_brands = brand_counts[brand_counts < 9332].index
    brand_to_number = {}
    counter = 2
    for brand in data['type_4_name']:
        if brand in small_brands:
            brand_to_number[brand] = 1
        else:
            brand_to_number[brand] = counter
            counter += 1

    # 应用编码
    data['type_4_name'] = data['type_4_name'].map(brand_to_number)
    data['type_4_name'] = data['type_4_name'].fillna(0)
    
    brand_counts = data['type_5_name'].value_counts()
    small_brands = brand_counts[brand_counts < 7610].index
    brand_to_number = {}
    counter = 2
    for brand in data['type_5_name']:
        if brand in small_brands:
            brand_to_number[brand] = 1
        else:
            brand_to_number[brand] = counter
            counter += 1

    # 应用编码
    data['type_5_name'] = data['type_5_name'].map(brand_to_number)
    data['type_5_name'] = data['type_5_name'].fillna(0)  


    # type_1_flow 归一化
    mean_type_1_num = data['type_1_flow'].mean()
    data['type_1_flow'] = data['type_1_flow'].fillna(mean_type_1_num)
    data['type_1_flow'] = (data['type_1_flow'] - data['type_1_flow'].min()) / (data['type_1_flow'].max() - data['type_1_flow'].min())

    # type_1_num 归一化
    mean_type_1_num = data['type_1_num'].mean()
    data['type_1_num'] = data['type_1_num'].fillna(mean_type_1_num)
    data['type_1_num'] = (data['type_1_num'] - data['type_1_num'].min()) / (data['type_1_num'].max() - data['type_1_num'].min())


    # type_2_flow 归一化
    mean_type_2_num = data['type_2_flow'].mean()
    data['type_2_flow'] = data['type_2_flow'].fillna(mean_type_2_num)
    data['type_2_flow'] = (data['type_2_flow'] - data['type_2_flow'].min()) / (data['type_2_flow'].max() - data['type_2_flow'].min())

    # type_2_num 归一化
    mean_type_2_num = data['type_2_num'].mean()
    data['type_2_num'] = data['type_2_num'].fillna(mean_type_2_num)
    data['type_2_num'] = (data['type_2_num'] - data['type_2_num'].min()) / (data['type_2_num'].max() - data['type_2_num'].min())


    # type_3_flow 归一化
    mean_type_3_num = data['type_3_flow'].mean()
    data['type_3_flow'] = data['type_3_flow'].fillna(mean_type_3_num)
    data['type_3_flow'] = (data['type_3_flow'] - data['type_3_flow'].min()) / (data['type_3_flow'].max() - data['type_3_flow'].min())

    # type_3_num 归一化
    mean_type_3_num = data['type_3_num'].mean()
    data['type_3_num'] = data['type_3_num'].fillna(mean_type_3_num)
    data['type_3_num'] = (data['type_3_num'] - data['type_3_num'].min()) / (data['type_3_num'].max() - data['type_3_num'].min())


    # type_4_flow 归一化
    mean_type_4_num = data['type_4_flow'].mean()
    data['type_4_flow'] = data['type_4_flow'].fillna(mean_type_4_num)
    data['type_4_flow'] = (data['type_4_flow'] - data['type_4_flow'].min()) / (data['type_4_flow'].max() - data['type_4_flow'].min())

    # type_4_num 归一化
    mean_type_4_num = data['type_4_num'].mean()
    data['type_4_num'] = data['type_4_num'].fillna(mean_type_4_num)
    data['type_4_num'] = (data['type_4_num'] - data['type_4_num'].min()) / (data['type_4_num'].max() - data['type_4_num'].min())


    # type_5_flow 归一化
    mean_type_5_num = data['type_5_flow'].mean()
    data['type_5_flow'] = data['type_5_flow'].fillna(mean_type_5_num)
    data['type_5_flow'] = (data['type_5_flow'] - data['type_5_flow'].min()) / (data['type_5_flow'].max() - data['type_5_flow'].min())

    # type_5_num 归一化
    mean_type_5_num = data['type_5_num'].mean()
    data['type_5_num'] = data['type_5_num'].fillna(mean_type_5_num)
    data['type_5_num'] = (data['type_5_num'] - data['type_5_num'].min()) / (data['type_5_num'].max() - data['type_5_num'].min())


    # acce_flow_1 归一化
    mean_acce_flow_1 = data['acce_flow_1'].mean()
    data['acce_flow_1'] = data['acce_flow_1'].fillna(mean_acce_flow_1)
    data['acce_flow_1'] = (data['acce_flow_1'] - data['acce_flow_1'].min()) / (data['acce_flow_1'].max() - data['acce_flow_1'].min())

    # acce_num_1 归一化
    mean_acce_num_1 = data['acce_num_1'].mean()
    data['acce_num_1'] = data['acce_num_1'].fillna(mean_acce_num_1)
    data['acce_num_1'] = (data['acce_num_1'] - data['acce_num_1'].min()) / (data['acce_num_1'].max() - data['acce_num_1'].min())


    # acce_flow_2 归一化
    mean_acce_flow_2 = data['acce_flow_2'].mean()
    data['acce_flow_2'] = data['acce_flow_2'].fillna(mean_acce_flow_2)
    data['acce_flow_2'] = (data['acce_flow_2'] - data['acce_flow_2'].min()) / (data['acce_flow_2'].max() - data['acce_flow_2'].min())

    # acce_num_2 归一化
    mean_acce_num_2 = data['acce_num_2'].mean()
    data['acce_num_2'] = data['acce_num_2'].fillna(mean_acce_num_2)
    data['acce_num_2'] = (data['acce_num_2'] - data['acce_num_2'].min()) / (data['acce_num_2'].max() - data['acce_num_2'].min())


    # acce_flow_3 归一化
    mean_acce_flow_3 = data['acce_flow_3'].mean()
    data['acce_flow_3'] = data['acce_flow_3'].fillna(mean_acce_flow_3)
    data['acce_flow_3'] = (data['acce_flow_3'] - data['acce_flow_3'].min()) / (data['acce_flow_3'].max() - data['acce_flow_3'].min())

    # acce_num_3 归一化
    mean_acce_num_3 = data['acce_num_3'].mean()
    data['acce_num_3'] = data['acce_num_3'].fillna(mean_acce_num_3)
    data['acce_num_3'] = (data['acce_num_3'] - data['acce_num_3'].min()) / (data['acce_num_3'].max() - data['acce_num_3'].min())

    data = data.drop('acce_flow_4', axis=1)
    data = data.drop('acce_num_4', axis=1)

    # acce_flow_5 归一化
    mean_acce_flow_5 = data['acce_flow_5'].mean()
    data['acce_flow_5'] = data['acce_flow_5'].fillna(mean_acce_flow_5)
    data['acce_flow_5'] = (data['acce_flow_5'] - data['acce_flow_5'].min()) / (data['acce_flow_5'].max() - data['acce_flow_5'].min())

    # acce_num_5 归一化
    mean_acce_num_5 = data['acce_num_5'].mean()
    data['acce_num_5'] = data['acce_num_5'].fillna(mean_acce_num_5)
    data['acce_num_5'] = (data['acce_num_5'] - data['acce_num_5'].min()) / (data['acce_num_5'].max() - data['acce_num_5'].min())
    
    data = data.drop('acce_flow_6', axis=1)
    data = data.drop('acce_num_6', axis=1)

    # acce_flow_7 归一化
    mean_acce_flow_7 = data['acce_flow_7'].mean()
    data['acce_flow_7'] = data['acce_flow_7'].fillna(mean_acce_flow_7)
    data['acce_flow_7'] = (data['acce_flow_7'] - data['acce_flow_7'].min()) / (data['acce_flow_7'].max() - data['acce_flow_7'].min())

    # acce_num_7 归一化
    mean_acce_num_7 = data['acce_num_7'].mean()
    data['acce_num_7'] = data['acce_num_7'].fillna(mean_acce_num_7)
    data['acce_num_7'] = (data['acce_num_7'] - data['acce_num_7'].min()) / (data['acce_num_7'].max() - data['acce_num_7'].min())
    

    # acce_flow_8 归一化
    mean_acce_flow_8 = data['acce_flow_8'].mean()
    data['acce_flow_8'] = data['acce_flow_8'].fillna(mean_acce_flow_8)
    data['acce_flow_8'] = (data['acce_flow_8'] - data['acce_flow_8'].min()) / (data['acce_flow_8'].max() - data['acce_flow_8'].min())

    # acce_num_8 归一化
    mean_acce_num_8 = data['acce_num_8'].mean()
    data['acce_num_8'] = data['acce_num_8'].fillna(mean_acce_num_8)
    data['acce_num_8'] = (data['acce_num_8'] - data['acce_num_8'].min()) / (data['acce_num_8'].max() - data['acce_num_8'].min())
    

    # acce_flow_9 归一化
    mean_acce_flow_9 = data['acce_flow_9'].mean()
    data['acce_flow_9'] = data['acce_flow_9'].fillna(mean_acce_flow_9)
    data['acce_flow_9'] = (data['acce_flow_9'] - data['acce_flow_9'].min()) / (data['acce_flow_9'].max() - data['acce_flow_9'].min())

    # acce_num_9 归一化
    mean_acce_num_9 = data['acce_num_9'].mean()
    data['acce_num_9'] = data['acce_num_9'].fillna(mean_acce_num_9)
    data['acce_num_9'] = (data['acce_num_9'] - data['acce_num_9'].min()) / (data['acce_num_9'].max() - data['acce_num_9'].min())


    # acce_flow_10 归一化
    mean_acce_flow_10 = data['acce_flow_10'].mean()
    data['acce_flow_10'] = data['acce_flow_10'].fillna(mean_acce_flow_10)
    data['acce_flow_10'] = (data['acce_flow_10'] - data['acce_flow_10'].min()) / (data['acce_flow_10'].max() - data['acce_flow_10'].min())

    # acce_num_10 归一化
    mean_acce_num_10 = data['acce_num_10'].mean()
    data['acce_num_10'] = data['acce_num_10'].fillna(mean_acce_num_10)
    data['acce_num_10'] = (data['acce_num_10'] - data['acce_num_10'].min()) / (data['acce_num_10'].max() - data['acce_num_10'].min())
    
    data = data.drop('acce_flow_11', axis=1)
    data = data.drop('acce_num_11', axis=1)
    data = data.drop('acce_flow_12', axis=1)
    data = data.drop('acce_num_12', axis=1)
    data = data.drop('acce_flow_13', axis=1)
    data = data.drop('acce_num_13', axis=1)
    

    # acce_flow_14 归一化
    mean_acce_flow_14 = data['acce_flow_14'].mean()
    data['acce_flow_14'] = data['acce_flow_14'].fillna(mean_acce_flow_14)
    data['acce_flow_14'] = (data['acce_flow_14'] - data['acce_flow_14'].min()) / (data['acce_flow_14'].max() - data['acce_flow_14'].min())

    # acce_num_14 归一化
    mean_acce_num_14 = data['acce_num_14'].mean()
    data['acce_num_14'] = data['acce_num_14'].fillna(mean_acce_num_14)
    data['acce_num_14'] = (data['acce_num_14'] - data['acce_num_14'].min()) / (data['acce_num_14'].max() - data['acce_num_14'].min())
    

    # acce_flow_15 归一化
    mean_acce_flow_15 = data['acce_flow_15'].mean()
    data['acce_flow_15'] = data['acce_flow_15'].fillna(mean_acce_flow_15)
    data['acce_flow_15'] = (data['acce_flow_15'] - data['acce_flow_15'].min()) / (data['acce_flow_15'].max() - data['acce_flow_15'].min())

    # acce_num_15 归一化
    mean_acce_num_15 = data['acce_num_15'].mean()
    data['acce_num_15'] = data['acce_num_15'].fillna(mean_acce_num_15)
    data['acce_num_15'] = (data['acce_num_15'] - data['acce_num_15'].min()) / (data['acce_num_15'].max() - data['acce_num_15'].min())
    data['label'] = data['label'].fillna(0)
    data = data.fillna(0)
    return data

# 读取数据
data = pd.read_csv("all_data_with_nan_one.csv")
print(data.info())  # 查看数据的基本信息

# 数据预处理
data = data.iloc[:, :]  # 选择所有列
data = data_process(data)  # 假设 data_process 是一个预处理函数
print(data.head())  # 查看预处理后的数据

# 特征选择
X = data.drop('label', axis=1)  # 特征
print(f"X shape: {X.shape}")  # 查看特征的形状
Y = data['label']  # 目标变量
print(f"Y shape: {Y.shape}")  # 查看目标变量的形状

# 选择最佳的 40 个特征
selector = SelectKBest(f_classif, k=40)
X_new = selector.fit_transform(X, Y)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, test_size=0.2, random_state=42)

# 记录开始时间
start_time = time.time()

# 建立模型
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)

# 预测1000个样本的时间
X_sample = X_new[:1000]  # 假设从 X_new 中选择前 1000 个样本进行预测
predictions_sample = model.predict(X_sample)  # 进行预测

end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time

print(f"Predicting 1000 samples took {elapsed_time:.4f} seconds.")  # 输出所需时间

# 评估模型
predictions = model.predict(X_test)
print(classification_report(Y_test, predictions))
scores = cross_val_score(model, X_new, Y, cv=5)
print("Cross-Validation Accuracy Scores:", scores)


# 参数调优
param_grid = {'n_estimators': [50, 100, 200]}
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, Y_train)
print("Best parameters:", grid.best_params_)


# 热力图
selected_indices = selector.get_support(indices=True)
selected_features = X.iloc[:, selected_indices]
correlation_matrix = selected_features.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap of Selected Features')
plt.show()



# ROC曲线
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(Y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# 混淆矩阵
y_pred = model.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()