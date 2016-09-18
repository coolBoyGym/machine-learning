# coding=utf-8
import treePlotter
import trees

if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = trees.createTree(lenses, lensesLabels)
    treePlotter.createPlot(lensesTree)

    # 使用决策树预测样本
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    predictResult = trees.classify(lensesTree, lensesLabels, ['young', 'myope',	'no', 'normal'])
    print predictResult
    # print lenses
    # print lensesTree

