# -*- coding:utf-8 -*-
import math

import imp
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR);
pd.options.display.max_rows = 10;
pd.options.display.float_format = "{:.1f}".format;

californiaHousingDataFrame = pd.read_csv("https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv", sep = ",");


californiaHousingDataFrame = californiaHousingDataFrame.reindex(np.random.permutation(californiaHousingDataFrame.index));

californiaHousingDataFrame["median_house_value"] /= 1000.0;

# 通过这种方式可以同时索引多个类
myFeature = californiaHousingDataFrame[["total_rooms"]];


# 必须是元组，此处可能有多个特征组
featureColumns = [tf.feature_column.numeric_column("total_rooms")];

targets = californiaHousingDataFrame["median_house_value"];

myOptimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0000001);
myOptimizer = tf.contrib.estimator.clip_gradients_by_norm(myOptimizer, 5.0);

linearRegressor = tf.estimator.LinearRegressor(feature_columns = featureColumns, optimizer = myOptimizer);



def myInputFn(features, targets, batchSize = 1, shuffle = True, numEpochs = None):
	# 数据格式转换
	features = {key:np.array(value) for key, value in dict(features).items()};

	# 创建数据集
	ds = Dataset.from_tensor_slices((features, targets));
	# 数据集的批次，和按照重复的周期数
	ds = ds.batch(batch_size = batchSize).repeat(numEpochs); 




	if shuffle:
		# 数据的抽样大小
		ds.shuffle(buffer_size = 10000);
	
	features, labels = ds.make_one_shot_iterator().get_next();
	return features, labels;

'''
learning = linearRegressor.train(input_fn = lambda : myInputFn(myFeature, targets), steps = 100);

predictionInputFn = lambda : myInputFn(myFeature, targets, numEpochs = 1, shuffle = False);

predictions = linearRegressor.predict(input_fn = predictionInputFn);

predictions = np.array([item["predictions"][0] for item in predictions]);
meanSquaredError = metrics.mean_squared_error(predictions, targets);
rootMeanSquaredError = math.sqrt(meanSquaredError);

print(meanSquaredError, rootMeanSquaredError);
'''

def trainModel(learningRate, steps, inputData, inputFeatures = [], inputLabels = [], batchSize = 10):
	
	period = 10;
	stepsPerPeriod = steps / period;

	featureData = inputData[inputFeatures];
	labelData = inputData[inputLabels];

	featureColumns = [tf.feature_column.numeric_column(item) for item in inputFeatures];

	trainInputFun = lambda : myInputFn(featureData, labelData, batchSize = batchSize);

	predictionInputFun = lambda : myInputFn(featureData, labelData, numEpochs = 1, shuffle = False);


	optimizer = tf.train.GradientDescentOptimizer(learning_rate = learningRate);
	optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0);

	linearRegressor = tf.estimator.LinearRegressor(feature_columns = featureColumns, optimizer = optimizer);

	results = [];

	for i in range(0, period):
		linearRegressor.train(input_fn = trainInputFun, steps = stepsPerPeriod);

		predictions = linearRegressor.predict(input_fn = predictionInputFun);
		predictions = np.array([item["predictions"][0] for item in predictions])

		rootMeanSquaredError = math.sqrt(metrics.mean_squared_error(predictions, labelData));

		
		weights = [linearRegressor.get_variable_value("linear/linear_model/%s/weights" % feature)[0] for feature in inputFeatures];
		bias = linearRegressor.get_variable_value("linear/linear_model/bias_weights");

		temp = {"RMES" : rootMeanSquaredError, "Weights" : weights, "bias" : bias};

		results.append(temp);

		print("Period %2d : %2f" % (i, rootMeanSquaredError));

	print("Model training finished.");

trainModel(0.00001, 1500, californiaHousingDataFrame, ["population"], ["median_house_value"], 1000);
