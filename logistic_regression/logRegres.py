# coding=utf-8

from numpy import *

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 逻辑回归梯度上升优化算法 每次使用全部样本点来更新权值
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)              # 将二维数组转化为numpy矩阵 从而进行矩阵运算
    labelMat = mat(classLabels).transpose()  # 对行向量做转置变为列向量 便于后面的计算
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)    # 矩阵运算 得到的h是 100x1 的列向量
        error = (labelMat - h)  # 列向量相减
        weights = weights + alpha * dataMatrix.transpose() * error  # 根据梯度方向进行权值的调整
    return weights


# 随机梯度上升算法 每次使用一个样本点来更新回归系数
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   # initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 改进的随机梯度上升算法 每次随机选取一个样本点来更新 同时步长alpha的值也在改变
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)   # initialize to all ones
    for j in range(numIter):  # 定义了遍历整个数据集的次数 是一个可调的参数
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0+j+i) + 0.0001    # alpha的值随着迭代的进行不断下降 但不是严格下降 这里的常量可以调整
            randIndex = int(random.uniform(0, len(dataIndex)))  # go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


# 画出最优拟合曲线
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')  # 将label值为0的点画出来
    ax.scatter(xcord2, ycord2, s=30, c='green')            # 将label值为1的点画出来
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]  # 画出最佳拟合曲线 曲线上的点满足方程:W0X0+W1X1+W2X2=0
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


# 自包含的测试函数 从疝气病症预测病马的死亡率
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)  # 计算回归系数
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))


if __name__ == '__main__':
    # dataArr, labelArr = loadDataSet()
    # w1 = gradAscent(dataArr, labelArr)
    # plotBestFit(w1.getA())
    # w2 = stocGradAscent0(array(dataArr), labelArr)
    # plotBestFit(w2)
    # w3 = stocGradAscent1(array(dataArr), labelArr, 15)
    # plotBestFit(w3)
    multiTest()
