import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

X_sepal = iris.data[:, [0, 1]] # 花萼特征集：两个特征长度和宽度

X_petal = iris.data[:, [2,3]] # 花瓣特征集：两个特征长度和宽度

y = iris.target # 标签集

X_train_petal, X_test_petal, y_train_petal, y_test_petal = train_test_split(X_petal, y, test_size=0.2, random_state=0)

print("花瓣训练集样本数: ", len(X_train_petal))
print("花瓣测试集样本数: ", len(X_test_petal))

scaler = StandardScaler()

X_train_sepal = scaler.fit_transform(X_train_petal)

X_test_sepal = scaler.transform(X_test_petal)

# 合并特征集和标签集, 留待以后数据展示之用
X_combined_sepal = np.vstack((X_train_sepal, X_test_sepal)) # 合并特征集
Y_combined_sepal = np.hstack((y_train_petal, y_test_petal)) # 合并标签集 因为此时的y是1D所以是水平合并

lr = LogisticRegression(penalty='l2', C=0.1) # 设定L2正则化和C参数
lr.fit(X_train_petal, y_train_petal)
score = lr.score(X_test_petal, y_test_petal) # 验证集分数评估
print("SKlearn 逻 辑 回 归 测 试 准 确 率{:.2f}%".format(score*100))