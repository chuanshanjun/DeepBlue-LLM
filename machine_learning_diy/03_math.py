import math # 导入数学工具包

y = math.log(100000000, 10) # 以10为底，在x值等于一亿的情况

print("以10为底，求一亿的对数: ", y) # y值等于8


import numpy as np # 导入Num Py 库

# 标量 - 0D(阶)张量
X = np.array(5) # 创建0D张量，也就是标量

print("X的值: ", X)

print("X的阶: ", X.ndim) # ndim属性显示标量的阶

print("X的数据类型: ", X.dtype) # dtype属性显示标量的数据类型

print("X的形状: ", X.shape) # shape属性显示标量的形状


# 标量可以直接赋值，例如通过for 循环
n = 0
for gender in [0, 1]:
    n = n + 1

print("n: ", n)

# 向量 - 1D(阶)张量

X = np.array([5, 6, 7, 8, 9]) # 创建1D张量

print("X的值: ", X)

print("X的阶: ", X.ndim)

print("X的数据类型: ", X.dtype)

print("X的形状: ", X.shape)

X = np.array([5]) # 1维向量，也就是1D数组里面只有一个元素


# 查看波士顿房价
from keras.datasets import boston_housing # 波士顿房价数据集

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

print("X_train的形状: ", X_train.shape)

print("X_tain的第一个样本的形状: ", X_train[0].shape)

print("y_train的形状: ", y_train.shape)

print("X_test的形状: ", X_test.shape)

print("y_test的形状: ", y_test.shape)

# 向量的点积 - 注意是向量！！！
weight = np.array([1, -1.8, 1, 1, 2]) # 权重向量（也就是多项式的参数）

X = np.array([1, 6, 7, 8, 9]) # 特征向量（也就是一个特定样本中的特征值）

y_hat = np.dot(weight, X) # 通过点积运算构建预测函数

print("函数返回结果: ", y_hat) # 输出预测结果

# 下面的语句也能实现相同的功能(点积)
y_hat = weight.dot(X)

print("第二种 点积的方式 函数返回结果: ", y_hat) # 输出预测结果

# 矩阵 - 2D(阶)张量
print("X_train的内容: ", X_train) # X_train 是2D张量，即矩阵

# 序列数据 - 3D(阶)张量

# 图像数据 - 4D(阶)张量

# 视频数据 - 5D(阶)张量

# 张量计算

list = [1,2,3,4,5]

array_01 = np.array(list) # list -> np.array

array_02 = np.array((6,7,8,9,10)) # tuple -> np.array

array_03 = np.array([[1,2,3], [4,5,6]]) # list -> 2D

print('list: ', list)

print('list -> np.array: ', array_01)

print('tuple -> np.array: ', array_02)

print('2D: ', array_03)

print('数组的形状: ', array_01.shape)

# print('列表形状: ', list.shape) # 列表没有形状会报错

# 直接赋值而得来的是Python内置的列表，要用array方法转换才能得到NumPy数组

# NumPy 直接创建数组的方法

array_04 = np.arange(1,5,1) # 通过arange函数生成数组

array_05 = np.linspace(1,5,5) # 通过linspace函数生成数组

print(array_04)

print(array_05)

# 通过索引或切片访问张量
array_06 = np.arange(10)
print(array_06)

index_01 = array_06[3] # 索引-第4个元素
print('第4个元素: ', index_01)

index_02 = array_06[-1] # 索引-最后一个元素
print('第-1个元素: ', index_02)

slice_01 = array_06[:4] # 从0到4切片
print('从0到4切片: ', slice_01)

slice_02 = array_06[0:12:4] # 从0到12切片，步长4
print('从0到12切片，步长4: ', slice_02)


from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 对多阶张量进行切片
print("在对多阶张量进行切片前，原始张量形状: ", X_train.shape)

# 对样本轴切片后面的数据保持
X_train_slice = X_train[10000:15000,:,:]
print("切片后的多阶张量形状: ", X_train_slice.shape)

array_07 = np.array([[1,2,3], [4,5,6]])
print(array_07[1:2], '它的形状是', array_07[1:2].shape)

print(array_07[1:2][0], '它的形状是', array_07[1:2][0].shape)

# 张量的变形和转置


# 张量的整体操作和逐元素运算
print('array_07: ', array_07)

# 数组内所有元素+1
array_07 += 1
print('array_07 + 1: ', array_07)

# 上面可以使用for循环使用实现
for i in range(array_07.shape[0]):
    for j in range(array_07.shape[1]):
        array_07[i,j] += 1
print(array_07)

# 输出每个元素的平方根
print(np.sqrt(array_07))


# 张量的变形(reshaping)以及转置
# eg (2,3) -> (3,2)
print(array_07, '形状是', array_07.shape)

print(array_07.reshape(3, 2), '形状是', array_07.reshape(3, 2).shape)

# 变形时只有，赋值才会改变数组本身
array_07 = array_07.reshape(3, 2)
print('转置前: ', array_07)

# 上面称为转置，可以直接使用 T
print('转置后: ', array_07.T)

# 张量变形一定要注意
# 复习一下标量
array_08 = np.array(10)
print(array_08, '形状是', array_08.shape, '阶是: ', array_08.ndim)

array_09 = np.arange(10)
print(array_09, '形状是', array_09.shape, '阶是: ', array_09.ndim)

# 将上面的向量 - 1D(阶)张量 -> 矩阵 - 2D(阶)张量

array_09 = array_09.reshape(10, 1)
print(array_09, '形状是', array_09.shape, '阶是: ', array_09.ndim)

# 广播 跳过
print('---以下是广播---')
array_08 = np.array([[0,0,0], [10,10,10], [20,20,20], [30,30,30]])
array_09 = np.array([[0, 1, 2]])
array_10 = np.array([[0], [1], [2], [3]])
list_11 = [[0,1,2]]
print('array_08的形状: ', array_08.shape)
print('array_09的形状: ', array_09.shape)
print('array_10的形状: ', array_10.shape)
array_12 = array_09.reshape(3)
print('array_12的形状: ', array_12.shape)
array_13 = np.array([1])
print('array_13的形状: ', array_13.shape)
array_14 = array_13.reshape(1,1)
print('array_14的形状: ', array_14.shape)
print('08 + 09 结果: ', array_08 + array_09)
print('08 + 10 结果: ', array_08 + array_10)
print('08 + 11 结果: ', array_08 + list_11)
print('08 + 12 结果: ', array_08 + array_12)
print('08 + 13 结果: ', array_08 + array_13)
print('08 + 14 结果: ', array_08 + array_14)


# 向量和矩阵点积运算

