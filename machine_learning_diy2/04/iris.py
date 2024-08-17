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

X_train_sepal, X_test_sepal, y_train_sepal, y_test_sepal = train_test_split(X_sepal, y, test_size=0.3, random_state=0)

print("花瓣训练集样本数: ", len(X_train_sepal))
print("花瓣测试集样本数: ", len(X_test_sepal))

scaler = StandardScaler()   # 标准化工具

X_train_sepal = scaler.fit_transform(X_train_sepal)

X_test_sepal = scaler.transform(X_test_sepal)

# 合并特征集和标签集, 留待以后数据展示之用
X_combined_sepal = np.vstack((X_train_sepal, X_test_sepal)) # 合并特征集
Y_combined_sepal = np.hstack((y_train_sepal, y_test_sepal)) # 合并标签集 因为此时的y是1D所以是水平合并

lr = LogisticRegression(penalty='l2', C=0.1) # 设定L2正则化和C参数
lr.fit(X_train_sepal, y_train_sepal)
score = lr.score(X_test_sepal, y_test_sepal) # 验证集分数评估
print("SKlearn 逻 辑 回 归 测 试 准 确 率{:.2f}%".format(score*100))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decison_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('o','x','v')
    colors = ('red', 'blue', 'lightgreen')
    color_Map = ListedColormap(colors[:len(np.unique(y))])
    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1
    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z, alpha=0.4, cmap=color_Map)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    X_test, y_test = X[test_idx, :], y[test_idx]

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1],
        alpha = 0.8, c = color_Map(idx),
        marker = markers[idx], label=cl)

# 使用不同的C值进行逻辑回归分类，并绘制分类结果：
from sklearn.metrics import accuracy_score # 导入准确率指标

C_param_range = [0.01, 0.1, 1, 10, 100, 1000]
sepal_acc_table = pd.DataFrame(columns=['C_param_range', 'Accuracy'])
sepal_acc_table['C_param_range'] = C_param_range
plt.figure(figsize=(10,10))
j = 0
for i in C_param_range:
    lr = LogisticRegression(penalty='l2', C=i, random_state=0)
    lr.fit(X_train_sepal, y_train_sepal)
    y_pred_sepal = lr.predict(X_test_sepal)
    sepal_acc_table.iloc[j,1] = accuracy_score(y_test_sepal, y_pred_sepal)
    j += 1
    plt.subplot(3,2,j)
    plt.subplots_adjust(hspace=0.4)
    plot_decison_regions(X=X_combined_sepal,
                         y=Y_combined_sepal,
                         classifier=lr,
                         test_idx=range(0,150))
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('C=%s'%i)
    plt.show()


