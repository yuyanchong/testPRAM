import pandas as pd
import numpy
import math
import torch
from catboost import CatBoostClassifier
from catboost import MultiTargetCustomObjective
import pandas as pd
from sklearn.model_selection import train_test_split
import method

#读取一个数据集
data = pd.read_csv('data/entrance.csv')

#将数据集中的秘密属性转换为数字
class_X_mapping = {'Average': 1, 'Good': 2, 'Vg': 3, 'Excellent': 4}
data['Class_ X_Percentage'] = data['Class_ X_Percentage'].map(class_X_mapping)
data['Class_XII_Percentage'] = data['Class_XII_Percentage'].map(class_X_mapping)

#初始化一个numpy数组secret
secret = ['Class_ X_Percentage','Class_XII_Percentage']

#notscret是数据集中除了秘密属性之外的属性集合
notsecret = data.columns.drop(secret)

#扰动矩阵参数
qnum0 = data[secret[0]].unique().shape[0]
qnum1 = data[secret[1]].unique().shape[0]
e1 = e2 = 5

#初始化一个概率转移矩阵，参数为qnum和e
qm = method.qmatrix(qnum0,e2)

#对秘密属性值进行第一次扰动
#对每一个秘密属性值，先转换为one-hot编码，然后通过qm矩阵进行扰动得到新的one-hot编码，最后将one-hot编码转换为数字
for i in range(len(secret)):
    secret_data = data[secret[i]]
    newsecret = method.getnewarry(secret_data,qm)
    #将扰动后的秘密属性值赋值给原数据集
    data[secret[i]] = newsecret

data.to_csv('data/test1entrance.csv',index=False)

sec_opt1=len(data[secret[0]].unique())
sec_opt2=len(data[secret[1]].unique())
records = len(data)

# # #将数据集分为训练集和测试集
# X_train1, X_test1, y_train1, y_test1 = train_test_split(data[notsecret], data[secret[0]], test_size=0.1, random_state=42)
# X_train2, X_test2, y_train2, y_test2 = train_test_split(data[notsecret], data[secret[1]], test_size=0.1, random_state=4)
#
# #训练模型
# categorical_features_indices = numpy.where(X_train1.dtypes != numpy.float)[0]
# model1 = CatBoostClassifier(iterations=500, depth=5,cat_features=categorical_features_indices,learning_rate=0.02, loss_function='MultiClass',logging_level='Verbose')
# model1.fit(X_train1,y_train1)
# model1.save_model('model1.cbm')
# result1 = model1.predict_proba(data[notsecret])
#
# model2 = CatBoostClassifier(iterations=500, depth=5,cat_features=categorical_features_indices,learning_rate=0.02, loss_function='MultiClass',logging_level='Verbose')
# model2.fit(X_train2,y_train2)
# model2.save_model('model2.cbm')

#加载模型
model1 = CatBoostClassifier()
model1.load_model('model1.cbm')
model2 = CatBoostClassifier()
model2.load_model('model2.cbm')
#预测
result1 = model1.predict_proba(data[notsecret])
result2 = model2.predict_proba(data[notsecret])

#qm矩阵的逆矩阵是通过numpy.linalg.inv(qm)函数得到的
#result中的每一行都左乘qm矩阵的逆矩阵
#结果保存在reverse1和reverse2中
reverse1 = numpy.dot(result1,numpy.linalg.inv(qm))
reverse2 = numpy.dot(result2,numpy.linalg.inv(qm))

#计算第二次扰动的概率转移矩阵
secondpm1 = []
secondpm2 = []

for k in range(len(reverse1)):
    k1 = result1[k]
    pi1 = reverse1[k]
    aqmatrix1 = numpy.zeros((sec_opt1, sec_opt1))
    for i in range(sec_opt1):
        for j in range(sec_opt1):
            aqmatrix1[i][j] = qm[j][i] * pi1[i] / k1[j]
    secondpm1.append(aqmatrix1)

for k in range(len(reverse2)):
    k2 = result2[k]
    pi2 = reverse2[k]
    aqmatrix2 = numpy.zeros((sec_opt2, sec_opt2))
    for i in range(sec_opt2):
        for j in range(sec_opt2):
            aqmatrix2[i][j] = qm[j][i] * pi2[i] / k2[j]
    secondpm2.append(aqmatrix2)

# #第二次扰动
secret_data = data[secret[0]]
newsecret = []
onehots = method.getonehot(secret_data)
for j in range(records):
    newsecret.append(method.tranonehot(onehots[j], secondpm1[j]))
data[secret[0]] = newsecret

secret_data = data[secret[1]]
newsecret = []
onehots = method.getonehot(secret_data)
for j in range(records):
    newsecret.append(method.tranonehot(onehots[j], secondpm2[j]))
data[secret[1]] = newsecret

#把数据集中秘密属性的值映射回去
class_X_mapping = {1: 'Average', 2: 'Good', 3: 'Vg', 4: 'Excellent'}
data['Class_ X_Percentage'] = data['Class_ X_Percentage'].map(class_X_mapping)
data['Class_XII_Percentage'] = data['Class_XII_Percentage'].map(class_X_mapping)

#保存扰动后的数据集
data.to_csv('data/test2entrance.csv',index=False)
