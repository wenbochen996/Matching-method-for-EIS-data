import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from math import sqrt

all_data = pd.read_csv(r"D:\金属腐蚀\个人总结\论文\最终程序版本\2023.8完整数据集.csv")
all_data = all_data.sample(frac=1, random_state=42)

all_features = all_data.iloc[:,0:-2].values
all_labels = all_data.iloc[:,-1:]   #固定为一列

def maxminnorm(array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t

all_features = maxminnorm(all_features)
all_features = np.nan_to_num(all_features)
all_labels = all_labels.values

#特征选择
np.random.seed(1)
# 创建随机森林分类器
rfc1 = RandomForestClassifier(n_estimators=100, random_state=42)
# 训练随机森林分类器
rfc1.fit(all_features, all_labels)
# 输出特征重要性评估结果
importance1 = rfc1.feature_importances_
# 根据特征重要性评估结果筛选特征
selected_features1 = []
for i,v in enumerate(importance1):
    if v > 0.0014:
        selected_features1.append(i)
#将选择出的特征，建立新的数据集select_features
select_features = all_features[:, selected_features1]

train_data, test_data, train_labels, test_labels = select_features[:2000],select_features[2000:],all_labels[:2000],all_labels[2000:]

# 建立一个Random Forest分类器
rf = RandomForestClassifier(max_features = int(sqrt(len(select_features[0]))), random_state=42)

# 设置要搜索的参数范围
param_grid = {'max_depth': [8,10,15,20,30,35,40], 'n_estimators': [10,30,50,70,90,100,120]}
#param_grid = {'max_depth': [2,5,None], 'n_estimators': [50,70]}
# 使用交叉验证来选择最佳参数，并记录得分
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(select_features, all_labels)

# 将交叉验证得分转换为数据框
results = pd.DataFrame(grid_search.cv_results_)

# 将max_depth和n_estimators参数转换为矩阵
scores = np.array(results.mean_test_score).reshape(len(param_grid['max_depth']), len(param_grid['n_estimators']))

# 使用Seaborn库生成热力图
fig, ax = plt.subplots()
sns.heatmap(scores, ax=ax,cmap='coolwarm', xticklabels=param_grid['n_estimators'], yticklabels=param_grid['max_depth'],annot=True, fmt='.3f')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
#plt.title('Random Forest Grid Search')
plt.tight_layout()
plt.show()