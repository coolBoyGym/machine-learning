# coding=utf-8
"""
贝叶斯决策理论的核心思想：选择具有最高概率的决策
条件概率公式： P(A | B) = P(A and B) / P(B)
贝叶斯准则：已知 P(x | c)，要求 P(c | x)
                p(c | x ) = p(x | c)p(c) / p(x)
朴素贝叶斯分类器中的两条假设：
1. 假设样本的各个特征之间相互独立
2. 假设每个特征同等重要
"""
from numpy import *


# 创建一些实验样本
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    # 1 is abusive, 0 not
    return postingList, classVec

# 基于已有的文本内容创建词汇表
def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)

# 将一段文本转化为词条向量(0-1类型)
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec

# 将一段文本转化为词条向量(数值类型)
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 对于两个类别分别计算p(w|Ci)
# 三个返回值分别是：由p(wi|C0)构成的向量, 由p(wi|C1)构成的向量以及p(c1)
# 在垃圾邮件分类的例子中, C1代表垃圾邮件 C0代表非垃圾邮件 wi代表一个单词
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)      # change to ones() 避免受到概率为0的项的影响
    p0Denom = 2.0; p1Denom = 2.0                        # change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)          # change to log() 使用对数乘法 避免因太多的小数相乘导致下溢出
    p0Vect = log(p0Num / p0Denom)          # change to log()
    return p0Vect, p1Vect, pAbusive


# 使用分类器对样本进行分类
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

# 自包含的测试函数
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb)


# 将一段字符串转化为小写字符列表
def textParse(bigString):    # input is big string, output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 自包含的垃圾邮件分类测试函数
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):  # 正例和负例分别有25个
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = range(50)
    testSet = []   # create train set and test set
    for i in range(10):  # 取50个样本中的10个作为测试样本 剩下的作为训练样本
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:      # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error", docList[docIndex]
    print 'the error rate is:', float(errorCount)/len(testSet)


if __name__ == '__main__':
    # listOPosts, listClasses = loadDataSet()
    # myVocabList = createVocabList(listOPosts)
    # print 'myVocabList =', myVocabList
    # print 'example vector =', setOfWords2Vec(myVocabList, listOPosts[0])
    #
    # trainMat = []
    # for postinDoc in listOPosts:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print 'p0V = ', p0V
    # print 'p1V = ', p1V
    # print 'pAb = ', pAb
    #
    # testingNB()
    spamTest()
