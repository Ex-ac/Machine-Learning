# -*- coding:utf-8 -*-

import math

from IPython import display
from matplotlib import gridspec
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

def preprocessFeatures(dataFrame, selectedFeatures):
	
	features = pd.DataFrame();

	features = dataFrame[selectedFeatures];

	features["rooms_per_person"] = (dataFrame["total_rooms"] / dataFrame["population"]);

	return features;

def preprocessTargets(dataFrame):

	targets = pd.DataFrame();
	
	targets["median_house_value"] = dataFrame["median_house_value"] / 1000.0;
	
	return targets;

def createFeatureColumns(dataFrame):
	return set([tf.feature_column.numeric_column(i) for i in dataFrame]);

def inputFun(inputFeatures, inputTargets, batchSize = 1, shuffle = True, numEpoch = None):
	features = {key : np.array(value) for key, value in dict(inputFeatures).items()};

	ds = Dataset.from_tensor_slices((features, inputTargets));
	ds = ds.batch(batchSize).repeat(numEpoch);

	if shuffle:
		ds = ds.shuffle(10000);
	
	features, targets = ds.make_one_shot_iterator().get_next();
	return features, targets;

def trainModel(learningRate, steps, trainFeatures, trainTargets, batchSize, validationFeaturs, validationTargets):
	periods = 10;
	steps_per_period = steps / periods;


	optimizer = tf.train.GradientDescentOptimizer(learningRate);
	optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0);

	linearRegressor = tf.estimator.LinearRegressor(feature_columns = createFeatureColumns(trainFeatures), optimizer = optimizer);


	trainInputFun = lambda : inputFun(trainFeatures, trainTargets, batchSize);

	predictTrainInputFun = lambda : inputFun(trainFeatures, trainTargets, 1, False, 1);

	predictValidationInputFun = lambda : inputFun(validationFeaturs, validationTargets, 1, False, 1);

	trainRMSEs = [];
	validationRMSEs = [];

	print("Train Model...");

	for i in range(0, periods):
		
		linearRegressor.train(input_fn = trainInputFun, steps = steps_per_period);

		trainPredictions = linearRegressor.predict(input_fn = predictTrainInputFun);

		trainPredictions = np.array([i["predictions"][0] for i in trainPredictions]);

		trainRMSE = math.sqrt(metrics.mean_squared_error(trainTargets, trainPredictions));

		trainRMSEs.append(trainRMSE);
				
		validationPredictions = linearRegressor.predict(input_fn = predictValidationInputFun);

		validationPredictions = np.array([i["predictions"][0] for i in validationPredictions]);

		validationRMSE = math.sqrt(metrics.mean_squared_error(validationTargets, validationPredictions));

		validationRMSEs.append(validationRMSE);
		
		print("\t%2d %0.2f %0.2f" % (i, trainRMSE, validationRMSE));
	print("Train finished");

	plt.figure();

	plt.plot(trainRMSEs, label = "Training");
	plt.plot(validationRMSEs, label = "Validation");

	plt.legend();

	return linearRegressor;

def selectedAndTransdormFeatures(dataFrame):
	selectedFeatures = pd.DataFrame();
	selectedFeatures["median_income"] = dataFrame["median_income"];

	for r in zip(range(32, 44), range(33, 45)):
		selectedFeatures["latitude_%d_to_%d" % r] = dataFrame["latitude"].apply(lambda x : 1.0 if x >= r[0] and x < r[1] else 0.0);
	
	return selectedFeatures;

pd.options.display.float_format = "{:.2f}".format;

dataFrame = pd.read_csv("./data.csv");

dataFrame = dataFrame.reindex(np.random.permutation(dataFrame.index));

selectedFeatures = ["median_house_value", "latitude"];
print(selectedFeatures);

featuresDataFrame = preprocessFeatures(dataFrame, selectedFeatures);
targetsDataFrame = preprocessTargets(dataFrame);


corrDataFrame = featuresDataFrame.copy();

corrDataFrame["target"] = targetsDataFrame["median_house_value"];

print(corrDataFrame.corr());

plt.scatter(featuresDataFrame["latitude"], targetsDataFrame["median_house_value"])

featuresDataFrame = selectedAndTransdormFeatures(dataFrame);

trainFeatures = featuresDataFrame.head(12000);
trainTargets = targetsDataFrame.head(12000);


validationFeaturs = featuresDataFrame.tail(5000);
validationTargets = targetsDataFrame.tail(5000);

linearRegressor = trainModel(0.01, 1500, trainFeatures, trainTargets, 50, validationFeaturs, validationTargets);

testDataFrame = pd.read_csv("./test.csv");

testFeatures = selectedAndTransdormFeatures(testDataFrame);
testTargets = preprocessTargets(testDataFrame);

testPredictFun = lambda : inputFun(testFeatures, testTargets, 1, False, 1);
testPredictions = linearRegressor.predict(input_fn = testPredictFun);

testPredictions = np.array([i["predictions"][0] for i in testPredictions]);



testRMSE = math.sqrt(metrics.mean_squared_error(testTargets, testPredictions));

print(testRMSE);

plt.show();

		

