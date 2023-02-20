import numpy
import math
import torch
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

#输入两个矩阵，输出两个矩阵的乘积
def matrixmult(a, b):
    return numpy.dot(a, b)

def qmatrix(prolength, e): #给定属性的取值数prolength和扰动参数e，输出概率转移矩阵Q
    k = prolength #该离散属性的取值个数，代表了取值概率向量的长度
    qeye = math.exp(e)/(k-1+math.exp(e)) #对角线上的转移概率
    qelse = 1/(k-1+math.exp(e)) #其他转移概率
    #得到矩阵Q，对角线上的元素为qeye，其他元素为qelse
    qmatrix = numpy.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                qmatrix[i][j] = qeye
            else:
                qmatrix[i][j] = qelse
    return qmatrix

def prammatrix2(k, qmatrix):
    #输入一个向量k，一个矩阵qmatrix，k由catboost预测得到。
    #向量k和qmatrix的维度都是n，n是离散属性的个数
    #向量pi是由矩阵qmatrix的逆矩阵和向量k相乘得到的
    pi = numpy.dot(numpy.linalg.inv(qmatrix), k)
    #矩阵Qa，Qa的第i行第j列的元素是qmatrix的第j行第i列的元素乘以pi的第i个元素再除以k的第j个元素
    aqmatrix = numpy.zeros((len(k), len(k)))
    for i in range(len(k)):
        for j in range(len(k)):
            aqmatrix[i][j] = qmatrix[j][i]*pi[i]/k[j]
    return aqmatrix

#函数compund，输入一个矩阵数组，输出数组中矩阵的克罗内积
def compund(matrixarray):
    result = matrixarray[0]
    for i in range(1, len(matrixarray)):
        result = matrixmult(result, matrixarray[i])
    return result

#训练一个用来预测取值概率向量的catboost
def train(dataset,notsecret,secret):
    #读取数据集
    data = pd.read_csv('data.csv')
    #划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data[notsecret], data[secret], test_size=0.1, random_state=42)
    #训练模型
    categorical_features_indices = numpy.where(X_train.dtypes != numpy.float)[0]
    model = CatBoostClassifier(iterations=100, depth=5,cat_features=categorical_features_indices,learning_rate=0.5, loss_function='Logloss',
                                logging_level='Verbose')
    model.fit(X_train,y_train)
    model.save_model('model.cbm')

#读取一个模型，输入一个数据集和秘密属性集合，输出秘密属性的取值概率向量
def predict(dataset,notsecret):
    #读取模型
    model = CatBoostClassifier()
    model.load_model('model.cbm')
    #预测
    result = model.predict_proba(dataset[notsecret])
    return result


#输入数据集dataset，属性集合secret，扰动矩阵ematrix,输出扰动后的数据集
#扰动矩阵ematrix是一个矩阵，矩阵的维度是集合secret中属性的个数相乘，由函数prammatrix得到
def perturb(dataset, secret, ematrix):
    #计算扰动后的数据集
    perturbed = dataset
    for i in range(len(dataset)):
        for j in range(len(secret)):
            perturbed[i][secret[j]] = numpy.random.multinomial(1, ematrix[j]).argmax()
    return perturbed

#输入整数数组，把数组中的元素转换成one-hot向量，数组的值是one-hot向量中为1的元素的下标,one-hot向量的维度是数组中最大的元素
def getonehot(array):
    #计算one-hot向量的维度
    dim = max(array)
    #计算one-hot向量
    onehot = numpy.zeros((len(array), dim))
    for i in range(len(array)):
        onehot[i][array[i]-1] = 1
    return onehot

#输入一个one-hot向量，一个概率转移矩阵，这个向量和q矩阵相乘得到概率向量,根据概率向量随机得到一个one-hot，取one-hot中为1的元素的下标+1
def tranonehot(onehot, qmatrix):
    #计算概率向量
    prob = numpy.dot(qmatrix, onehot)
    #处理概率向量，把概率向量中的元素限制在0到1之间
    for i in range(len(prob)):
        if prob[i] < 0:
            prob[i] = 0
        if prob[i] > 1:
            prob[i] = 1
    #把概率向量中的元素归一化
    prob = prob / sum(prob)
    #根据概率向量随机得到一个one-hot
    result = numpy.random.multinomial(1, prob).argmax()+1
    return result


#输入一个数组，把数组中的元素转换成one-hot向量，对每一个one-hot向量，根据概率转移矩阵qmatrix得到一个新的one-hot向量，把新的one-hot向量转换成整数，输出整数数组
def getnewarry(arry, qmatrix):
    result = []
    onehots = getonehot(arry)
    for i in range(len(onehots)):
        result.append(tranonehot(onehots[i], qmatrix))
    return result

def softmax(w):
    e = numpy.exp(numpy.array(w) / 1)
    dist = e / numpy.sum(e)
    return dist


#随机一个1*5的向量q，元素是0到1之间的随机数
q = [0.1,0.1,0.1,0.7]
qm = qmatrix(4,4)
ver = numpy.linalg.inv(qm)
q2= [0.20492662,0.20492662,0.20492662,0.38522013]
print(numpy.dot(q2,ver))
print(numpy.dot(ver,q2))
aqmatrix = numpy.zeros((4,4))
for i in range(4):
    for j in range(4):
        aqmatrix[i][j] = qm[j][i] * q[i] / q2[j]
final = numpy.dot(aqmatrix,qm)
test = [4,3,2,4,4,4,3,1,4,2,4]
show = []
onehots = getonehot(test)
print(final)
for i in range(len(onehots)):
    show.append(tranonehot(onehots[i], final))
print(show)





