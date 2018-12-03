# -*- coding:utf-8 -*-
import math
from IPython import display
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR);
pd.options.display.max_rows = 10;
pd.options.display.float_format = "{:.2f}".format

dataFrame = pd.read_csv("E:/Study/Python/Machine-Learning/data.csv", sep = ',');
dataFrame = dataFrame.reindex(np.random.permutation(dataFrame.index));

def preprocess_features(dataFrame = pd.DataFrame()):

	selectedFeature = dataFrame[["longitude", "latitude", "housing_median_age", "total_rooms","total_bedrooms", "population", "households", "median_income"]];

	processFeatures = selectedFeature.copy();
	processFeatures["rooms_per_person"] = processFeatures["total_rooms"] / processFeatures["population"];
	return processFeatures;

def preprocess_targets(dataFrame = pd.DataFrame()):
	outputTarget = pd.DataFrame();
	outputTarget["median_house_value"] = dataFrame["median_house_value"] / 1000.0;
	return outputTarget;

trainingExamples = preprocess_features(dataFrame.head(12000));
print(trainingExamples.describe());

trainingTargets = preprocess_targets(dataFrame.head(12000));
print(trainingTargets.describe());

validationExamples = preprocess_features(dataFrame.tail(5000));
print(validationExamples.describe());

validationTargets = preprocess_targets(dataFrame.tail(5000));
print(validationTargets.describe());

plt.figure(figsize=(13, 8));

ax = plt.subplot(1, 2, 1);
ax.set_title("Validation Data");

ax.set_autoscaley_on(False);
ax.set_ylim([32, 43]);

ax.set_autoscalex_on(False);
ax.set_xlim([-126, -112]);

plt.scatter(validationExamples["longitude"], validationExamples["latitude"], cmap="coolwarm", c=validationTargets["median_house_value"] / validationTargets["median_house_value"].max());


ax = plt.subplot(1,2,2)
ax.set_title("Training Data")

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(trainingExamples["longitude"], trainingExamples["latitude"], cmap="coolwarm", c=trainingTargets["median_house_value"] / trainingTargets["median_house_value"].max())

plt.show();

def inputFun(features, targets, batchSize = 1, shuffle = True, numEpochs = None):
	
	features = {key:np.array(value) for key, value in dict(features).items()};

	ds = Dataset.from_tensor_slices((features, targets));
	ds = ds.batch(batchSize).repeat(numEpochs);

	if shuffle:
		ds = ds.shuffle(10000);
	features, labels = ds.make_one_shot_iterator().get_next();

	return features, labels;

def createFeaturesColumns(features):
	return set([tf.feature_column.numeric_column(feature) for feature in features]);

def trainModel(learningRata, steps, batchSize, trainingExamples, trainingTargets, validationExamples, validationTargets):

	periods = 10;
	stepPerPeriod = steps / periods;

	optimizer = tf.train.GradientDescentOptimizer(learningRata);
	optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0);

	linearRegressor = tf.estimator.LinearRegressor(feature_columns = createFeaturesColumns(trainingExamples), optimizer = optimizer);

	trainingInputFun = lambda : inputFun(trainingExamples, trainingTargets, batchSize);

	predictTrainingInputFun = lambda : inputFun(trainingExamples, trainingTargets, 1, False, 1)

	perdictValidationInputFun = lambda : inputFun(validationExamples, validationTargets, 1, False, 1);

	print("Training model...");
	print("RMSE (on training data):");

	trainingRMSEs = [];
	validationRMSEs = [];
	for i in range(0, periods):
		linearRegressor.train(trainingInputFun, steps = stepPerPeriod);

		trainingPredicitions = linearRegressor.predict(input_fn = predictTrainingInputFun);
		trainingPredicitions = np.array([item["predictions"][0] for item in trainingPredicitions]);

		validationPredicitions = linearRegressor.predict(input_fn = perdictValidationInputFun);
		validationPredicitions = np.array([item["predictions"][0] for item in validationPredicitions]);

		trainingRootMeanSquaredError = math.sqrt(metrics.mean_squared_error(trainingTargets, trainingPredicitions));

		validationRootMeanSquaredError = math.sqrt(metrics.mean_squared_error(validationTargets, validationPredicitions));

		trainingRMSEs.append(trainingRootMeanSquaredError);
		validationRMSEs.append(validationRootMeanSquaredError);

		print("    %2d : %0.2f	%0.2f" % (i, trainingRootMeanSquaredError, validationRootMeanSquaredError));
	
	print("Model training finished.");

	plt.ylabel("RMES");
	plt.xlabel("Periods");

	plt.title("Root Mean Squared Error vs. Periods");

	plt.tight_layout();
	plt.plot(trainingRMSEs, label = "training");
	plt.plot(validationRMSEs, label = "validations");

	plt.legend();


	return linearRegressor;

linearRegressor = trainModel(0.00003, 500, 5, trainingExamples, trainingTargets, validationExamples, validationTargets);

plt.show();

testDataFrame = pd.read_csv("https://download.mlcc.google.cn/mledu-datasets/california_housing_test.csv", sep=",");

testFreatures = preprocess_features(testDataFrame);
testTargets = preprocess_targets(testDataFrame);

testInputFun = lambda : inputFun(testFreatures, testTargets, 1, False, 1);

testPerdictions = linearRegressor.predict(input_fn = testInputFun);
testPerdictions = np.array([i["predictions"][0] for i in testPerdictions]);

rootMeanSquaredError = math.sqrt(metrics.mean_squared_error(testTargets, testPerdictions));

print("RMSE (on test data) : %0.2f" % rootMeanSquaredError);


cmake .. -A x64 -DCMAKE_BUILD_TYPE=Release -DSWIG_EXECUTABLE="D:\swigwin\swig.exe" -DPYTHON_EXECUTABLE="D:\Python36\python.exe" -DPYTHON_LIBRARIES="D:\Python36\libs\python36.dll" -Dtensorflow_WIN_CPU_SIMD_OPTIONS=/arch:AVX2