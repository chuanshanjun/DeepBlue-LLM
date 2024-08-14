# 使用线性回归算法预测波士顿房价

# 1 导入数据
from keras.src.datasets import boston_housing

(X_train_house, y_train_house), (X_test_house, y_test_house) = boston_housing.load_data()

print ("数据集张量形状:", X_train_house.shape)
print ("第一个数据样本:\n", X_train_house[0])

print ("标签张量形状:", y_train_house.shape)
print ("第一个数据样本的标签:", y_train_house[0])
print ("第二个数据样本的标签:", y_train_house[1])

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_house, y_train_house)

print('给预测评分: ', model.score(X_test_house, y_test_house))