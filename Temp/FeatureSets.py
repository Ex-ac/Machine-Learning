# -*- coding:utf-8 -*-

from IPython import display
from matplotlib import gridspec
from matplotlib import cm
from matplotlib import pyplot as plt
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics


def preprocessFeatures(dataFrame, selectedFeatureList):

    selectedFeatures = dataFrame[selectedFeatureList]

    resultFeatures = selectedFeatures.copy()

    resultFeatures["rooms_per_person"] = dataFrame["total_rooms"] / \
        dataFrame["population"]

    return resultFeatures


def preprocessTargets(dataFrame):

    resultTargets = pd.DataFrame()

    resultTargets["median_house_value"] = dataFrame["median_house_value"] / \
        1000.00
    return resultTargets


def creatureFeatureColunm(inputFeature):
    return set([tf.feature_column.numeric_column(i) for i in inputFeature])


def inputFun(inputDataFrame, batchSize=1, shuffle=True, numEpoch=None):

    features = {key: np.array(value) for key, value in dict(
        inputDataFrame["features"]).items()}

    ds = Dataset.from_tensor_slices((features, inputDataFrame["targets"]))
    ds = ds.batch(batchSize).repeat(numEpoch)

    if shuffle:
        ds = ds.shuffle(10000)

    resultFeatures, resultTargets = ds.make_one_shot_iterator().get_next()
    return resultFeatures, resultTargets


def trainModel(learningRate, steps, batchSize, trainingExample,
               validationExample):

    priods = 10
    stepsPerPeriod = steps / priods

    def trainingInputFun(): return inputFun(trainingExample, batchSize)

    def predictTraningInputFun(): return inputFun(
        trainingExample, shuffle=False, numEpoch=1)

    def predictValidationInputFun(): return inputFun(
        validationExample, shuffle=False, numEpoch=1)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

    linearRegressor = tf.estimator.LinearRegressor(
        feature_columns=creatureFeatureColunm(
                                            trainingExample["features"]),
        optimizer=optimizer)

    print("Training Model...")

    trainingRMSEs = []
    validationRMSEs = []
    for i in range(0, priods):

        linearRegressor.train(input_fn=trainingInputFun, steps=stepsPerPeriod)

        trainingPredictions = linearRegressor.predict(
            input_fn=predictTraningInputFun)

        trainingPredictions = np.array(
            [i["predictions"][0] for i in trainingPredictions])

        trainingRMSE = math.sqrt(metrics.mean_squared_error(
            trainingExample["targets"], trainingPredictions))

        validationPredictions = linearRegressor.predict(
            input_fn=predictValidationInputFun)

        validationPredictions = np.array(
            [i["predictions"][0] for i in validationPredictions])

        validationRMSE = math.sqrt(metrics.mean_squared_error(
            validationExample["targets"], validationPredictions))

        print("\t%2d %0.2f %0.2f" % (i, trainingRMSE, validationRMSE))

        trainingRMSEs.append(trainingRMSE)

        validationRMSEs.append(validationRMSE)

    print("Finished Training!")

    plt.figure()

    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("RMSE vs. Priods")

    plt.tight_layout()

    plt.plot(trainingRMSEs, label="training")
    plt.plot(validationRMSEs, label="validations")

    plt.legend()

    return linearRegressor


pd.options.display.float_format = "{:.2f}".format
dataFrame = pd.read_csv("E:/Study/Python/Machine-Learning/data.csv", sep=',')
dataFrame = dataFrame.reindex(np.random.permutation(dataFrame.index))
print(dataFrame.describe())

# selectedFeatureList = [
#    "latitude",
#    "longitude",
#    "housing_median_age",
#    "total_rooms",
#    "population",
#    "households",
#    "median_income"
# ]
selectedFeatureList = [
    "median_income",
    "households",
    "latitude",
]
features = preprocessFeatures(dataFrame, selectedFeatureList)
targets = preprocessTargets(dataFrame)

correlation = features.copy()
correlation["targets"] = targets["median_house_value"]
print(correlation.corr())

trainingExample = {
    "features": features.head(15000),
    "targets": targets.head(15000),
}

validationExample = {
    "features": features.tail(2000),
    "targets": targets.tail(2000),
}


linearRegressor = trainModel(learningRate=0.01, steps=1000, batchSize=100,
                             trainingExample=trainingExample,
                             validationExample=validationExample)

testDataFrame = pd.read_csv(
    "E:/Study/Python/Machine-Learning/test.csv", sep=',')
testFeatures = preprocessFeatures(testDataFrame, selectedFeatureList)
testTargets = preprocessTargets(testDataFrame)
testExample = {
    "features": testFeatures,
    "targets": testTargets,
}


def predictTestInputFun(): return inputFun(testExample, 1, False, 1)


testPredictions = linearRegressor.predict(input_fn=predictTestInputFun)
testPredictions = np.array([i['predictions'][0] for i in testPredictions])
rmse = math.sqrt(metrics.mean_squared_error(
    testExample["targets"], testPredictions))
print(rmse)
plt.show()
