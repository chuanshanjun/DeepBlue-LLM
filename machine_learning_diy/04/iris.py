import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import datasets


iris = datasets.load_iris()

X_sepal = iris.data[:, [0,1]] # 花萼特征集：两个特征长度和宽度

X_petal = iris.data[:, [2,3]] # 花瓣特征集，两个特征长度和宽度

y = iris.target # 标签集

x1 = np.array(X_sepal)
x2 = np.array(X_petal)
y1 = np.array(y)

# 将数据组合后，画出图形
temp = np.hstack((x1, x2, y1.reshape(-1, 1)))
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df_iris = pd.DataFrame(temp, columns=columns)

print(df_iris.head())

# 0代表Setosa山鸢尾，1代表Versicolor杂色鸢尾，2代表Virginica维吉尼亚鸢尾
plt.scatter(x=df_iris.sepal_length[df_iris.species==0],
            y=df_iris.sepal_width[df_iris.species==0],
            c='red', marker='.')

plt.scatter(x=df_iris.sepal_length[df_iris.species==1],
            y=df_iris.sepal_width[df_iris.species==1],
            c='blue', marker='x')

plt.scatter(x=df_iris.sepal_length[df_iris.species==2],
            y=df_iris.sepal_width[df_iris.species==2],
            c='green', marker='v')

plt.legend(['Setosa','versicolor','virginica'])
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()


# 对数据集进行标准化
from sklearn.model_selection import train_test_split # 导入拆分数据集工具
from sklearn.preprocessing import StandardScaler # 导入标准化工具

X_train_sepal, X_test_sepal, y_train_sepal, y_test_sepal = train_test_split(X_sepal, y,
                                                                            test_size=0.3, random_state=0) # 拆分数据集

print("花瓣训练集样本数：", len(X_train_sepal))
print("花瓣测试集样本数：", len(X_test_sepal))

scaler = StandardScaler()                           # 标准化工具
X_train_sepal = scaler.fit_transform(X_train_sepal) # 测试集数据标准化
X_test_sepal = scaler.transform(X_test_sepal)       # 测试集数据标准化

# 合并特征集和标签集，留待以后数据展示之用
X_combined_sepal = np.vstack((X_train_sepal, X_test_sepal)) # 合并特征集
Y_combined_sepal = np.hstack((y_train_sepal, y_test_sepal)) # 合并标签集

# 通过Sklearn 的 Logistic Regression 函数，实现多元分类功能
from sklearn.linear_model import LogisticRegression # 导入逻辑回归模型

lr = LogisticRegression(penalty='l2', C=0.1) # 设立L2正则化和C参数
lr.fit(X_train_sepal, y_train_sepal) # 训练机器
score = lr.score(X_test_sepal, y_test_sepal) # 验证集分数评估

print("Sklearn 逻辑回归测试准确率{:.2f}%".format(score*100))


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

# 可以将不同C值取值的变化的学习曲线(准确率)画出来

# 设置 C 值为10是的逻辑回归准确率
lr = LogisticRegression(penalty='l2', C=10) # 设定L2正则化和C参数
lr.fit(X_train_sepal, y_train_sepal) # 训练机器
score = lr.score(X_test_sepal, y_test_sepal) # 测试集分数
print('Sklearn 逻辑回归测试准确率{:.2f}%'.format(score*100))

