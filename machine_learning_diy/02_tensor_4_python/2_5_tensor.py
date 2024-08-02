import numpy as np

# 2.5.1 机器学习中张量的创建
list = [1, 2, 3, 4, 5]                   # 创建列表
array_01 = np.array([1, 2, 3, 4, 5])     # 列表转换为数组
array_02 = np.array((6, 7, 8, 9, 10))    # 元组转换为数组
array_03 = np.array([[1,2,3], [4,5,6]])  # 列表转换为2D数组，注：它是一个列表，其中包含两列表元素

print('列表:                ', list)
print('列表转换为数组:        ', array_01)
print('元组转换为数组:        ', array_02)
print('2D数组:              ', array_03)
print('数组 array_01 的形状: ', array_01.shape)
# print('列表的形状:           ', list.shape)          # 列表没有形状会报错

# 输出内容如下：
# 注意：1）列表和数组最大的区别在于，列表每个元素中以"逗号"做分隔
# 列表:                 [1, 2, 3, 4, 5]
# 列表转换为数组:         [1 2 3 4 5]
# 元组转换为数组:         [ 6  7  8  9 10]
# 2D数组:               [[1 2 3]
#  [4 5 6]]
# 数组 array_01 的形状:  (5,)


# 2.5.1 机器学习中创建张量
array_04 = np.arange(1, 5, 1) # 通过arange函数生成数组
print('array_04    : ', array_04)
print('array_04形状 : ', array_04.shape)

array_05 = np.linspace(1,5,5)
print('array_05    : ', array_05)
print('array_05形状 : ', array_05.shape)


# 2.5.2 通过索引和切片访问张量中的数据
# 复杂数组访问
array_07 = np.array([[1,2,3], [4,5,6]])
print(array_07[1:2], '它的形状是: ', array_07[1:2].shape)
print(array_07[1:2][0], '它的形状为: ', array_07[1:2][0].shape)

# 2.5.3 张量的整体操作和逐元素运算


# 2.5.4 张量的变形和转置
# 注意刚才这种行变列，列变行，特殊的变形，也称为矩阵转置，更简单的方法使用T，但两个改变后的数据顺序不一样
# 总结一下：
# reshape(3, 2) 改变了数组的形状，但保持了原有的数据顺序。第一行的前两个元素成为新数组的第一行，接着是第二行的前两个元素成为新数组的第二行，依此类推。
# T 或 .transpose() 是一个转置操作，它交换了数组的行和列。对于一个 2x3 的矩阵，转置后会得到一个 3x2 的矩阵，其中原矩阵的第一行变成了新矩阵的第一列，以此类推。

array_07 = np.array([[1,2,3], [4,5,6]])
print(array_07, '形状是', array_07.shape)
print(array_07.reshape(3,2), '形状是: ', array_07.reshape(3,2).shape)

print(array_07, 'array_07转值: ', array_07.T)


array_06 = np.arange(10)
print(array_06, 'array_06形状是: ', array_06.shape, '阶为: ', array_06.ndim)

# 很关键，通过reshape操作，升阶了
array_06 = array_06.reshape(10, 1)
print(array_06, 'array_06形状是: ', array_06.shape, '阶为: ', array_06.ndim)


# 2.2.5 广播
array_08 = np.array([[0,0,0], [10,10,10], [20,20,20], [30,30,30]])
array_09 = np.array([[0,1,2]])
print(array_09, 'array_09形状是: ', array_09.shape, '阶为: ', array_09.ndim)

array_10 = np.array([[0], [1], [2], [3]])
print(array_10, 'array_10形状是: ', array_10.shape, '阶为: ', array_10.ndim)

list_11 = [[0,1,2]]
print('list_11: ', list_11)

array_12 = array_09.reshape(3)
print(array_12, 'array_12形状是: ', array_12.shape, '阶为: ', array_12.ndim)

array_12_2 = array_09.reshape(3, 1)
print(array_12_2, 'array_12_2形状是: ', array_12_2.shape, '阶为: ', array_12_2.ndim)

array_13 = np.array([1])
print(array_13, 'array_13形状是: ', array_13.shape, '阶为: ', array_13.ndim)

array_14 = array_13.reshape(1,1)
print(array_14, 'array_14形状是: ', array_14.shape, '阶为: ', array_14.ndim)

print('08+09: ', array_08 + array_09)

print('08+10: ', array_08 + array_10)

print('08+11: ', array_08 + list_11)

print('08+12: ', array_08 + array_12)

print('08+13: ', array_08 + array_13)

print('08+14: ', array_08 + array_14)


# 2.5.6 向量和矩阵的点积运算
vector_01 = np.array([1, 2, 3])
vector_02 = np.array([[1], [2], [3]])
vector_03 = np.array([2])
vector_04 = vector_02.reshape(1,3)
print('vector_01 形状: ', vector_01.shape, ' 阶: ', vector_01.ndim)
print('vector_02 形状: ', vector_02.shape, ' 阶: ', vector_02.ndim)
print('vector_03 形状: ', vector_03.shape, ' 阶: ', vector_03.ndim)
print('vector_04 形状: ', vector_04.shape, ' 阶: ', vector_04.ndim)

print('01和01的点积: ', vector_01.dot(vector_01))
print('01和02的点积: ', vector_01.dot(vector_02))
print('04和02的点积: ', vector_04.dot(vector_02))

print('01和数字的点积: ', vector_01.dot(2))
print('02和03的点积: ', vector_02.dot(vector_03))
print('02和04的点积: ', vector_02.dot(vector_04))
# print('01和03的点积: ', vector_01.dot(vector_03)) 报错
# print('02和02的点积: ', vector_02.dot(vector_02))