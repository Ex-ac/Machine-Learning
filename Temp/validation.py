# -*- coding:utf-8 -*-

import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.data import Dataset
import numpy as np
import pandas as pd
from sklearn import metrics


def preprocessFeatures(dataFrame):
    seriesSelect = ["longitude", "latitude", "housing_median_age",
                    "total_rooms",
                    "total_bedrooms", "population", "households", "median_income"]

    features = dataFrame[seriesSelect]

    features["rooms_per_person"] = features["total_rooms"] / \
        features["population"]
    return features


def preprocessTargets(dataFrame):
    targets = pd.DataFrame()

    targets["median_house_value"] = (dataFrame["median_house_value"] / 1000.0)

    return targets


def inputFun(inputFeatures, inputTargets, batchSize=1, shuffle=True, numEpoch=None):

    features = {key: np.array(value)
                for key, value in dict(inputFeatures).items()}

    ds = Dataset.from_tensor_slices((features, inputTargets))
    ds = ds.batch(batchSize).repeat(numEpoch)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels


def createColumnFeatures(features):
    return [tf.feature_column.numeric_column(i) for i in features]


def trainModel(learningRate, steps, trainDataFrame, validationDataFrame, batchSize):

    periods = 10
    stepPerPeriod = steps / periods

    trainFeatures = preprocessFeatures(trainDataFrame)
    trainTargets = preprocessTargets(trainDataFrame)

    validationFeatures = preprocessFeatures(validationDataFrame)
    validationTargets = preprocessTargets(validationDataFrame)

    optimizer = tf.train.GradientDescentOptimizer(learningRate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

    linearRegressor = tf.estimator.LinearRegressor(
        feature_columns=createColumnFeatures(trainFeatures), optimizer=optimizer)

    def trainInputFun(): return inputFun(trainFeatures, trainTargets, batchSize)

    def trainPredictInputFun(): return inputFun(
        trainFeatures, trainTargets, batchSize=1, shuffle=False, numEpoch=1)

    def validationPredictInputFun(): return inputFun(
        validationFeatures, validationTargets, shuffle=False, numEpoch=1)

    trainRSMEs = []
    validationRSMEs = []

    print("Training Model...")
    for i in range(0, periods):
        linearRegressor.train(steps=stepPerPeriod, input_fn=trainInputFun)

        trainPredictions = linearRegressor.predict(
            input_fn=trainPredictInputFun)
        trainPredictions = np.array([i["predictions"][0]
                                     for i in trainPredictions])

        validationPredictions = linearRegressor.predict(
            input_fn=validationPredictInputFun)
        validationPredictions = np.array(
            [i["predictions"][0] for i in validationPredictions])

        trainRSME = math.sqrt(metrics.mean_squared_error(
            trainTargets, trainPredictions))

        validationRSME = math.sqrt(metrics.mean_squared_error(
            validationTargets, validationPredictions))

        trainRSMEs.append(trainRSME)
        validationRSMEs.append(validationRSME)

        print("\tPeriod %2d, %0.2f, %0.2f" % (i, trainRSME, validationRSME))

    print("Training finished.")

    plt.figure()
    plt.ylabel("RSME")
    plt.xlabel("Periods")
    plt.title("RSEM vs. Periods")

    plt.plot(trainRSMEs, label="Train")
    plt.plot(validationRSMEs, label="Validations")

    plt.legend()

    return linearRegressor


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.2f}".format

dataFrame = pd.read_csv("E:/Study/Python/Machine-Learning/data.csv", sep=",")


dataFrame = dataFrame.reindex(np.random.permutation(dataFrame.index))


trainingDataFrame = dataFrame.head(12000)
validationDataFrame = dataFrame.tail(5000)


# 绘制在同一个图中绘制俩个数据集的直方图，观察数据分布是否均匀，该方法只适合在各个特征都是没有关联的，例如latitude, longitude是无关的关联数据
'''
columns = len(trainingDataFrame.columns);

for index, i in zip(trainingDataFrame.columns, range(0, columns)):

    figure = plt.figure(i);

    plt.title(index);

    plt.subplot(1, 2, 1);

    plt.title("traning " + index);

    trainingDataFrame[index].hist(density = True);

    plt.subplot(1, 2, 2);
    plt.title("validation " + index);
    validationDataFrame[index].hist(density = True);

'''

linearRegressor = trainModel(
    0.00003, 500, trainingDataFrame, validationDataFrame, 5)


testDataFrame = pd.read_csv(
    "E:/Study/Python/Machine-Learning/test.csv", sep=',')

testPerdictionFeatures = preprocessFeatures(testDataFrame)
testPerdictionTargets = preprocessTargets(testDataFrame)


def testPerdictionInputFun(): return inputFun(testPerdictionFeatures,
                                              testPerdictionTargets,
                                              batchSize=1,
                                              shuffle=False, numEpoch=1)


testPredictions = linearRegressor.predict(input_fn=testPerdictionInputFun)
testPredictions = np.array([i["predictions"][0] for i in testPredictions])

testRMSE = math.sqrt(metrics.mean_squared_error(
    testPerdictionTargets, testPredictions))

print("test RMSE: %0.2f" % testRMSE)

plt.show()
