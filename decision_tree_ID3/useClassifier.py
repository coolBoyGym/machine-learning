# coding=utf-8
import treePlotter
import trees
"""
使用决策树算法预测隐形眼镜的类型
lenses.txt中存储了20个样本,每个样本包含四个属性值与一个标签
通过训练样本构建决策树 再根据输入样本的属性值对其进行分类
"""

if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = trees.createTree(lenses, lensesLabels)
    treePlotter.createPlot(lensesTree)

    # 使用决策树预测样本
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    predictResult = trees.classify(lensesTree, lensesLabels, ['young', 'myope', 'no', 'normal'])
    print predictResult
    # print lenses
    # print lensesTree

