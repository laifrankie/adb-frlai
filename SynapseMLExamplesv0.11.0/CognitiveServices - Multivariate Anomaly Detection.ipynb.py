# Databricks notebook source
# MAGIC %md
# MAGIC # Recipe: Cognitive Services - Multivariate Anomaly Detection 
# MAGIC This recipe shows how you can use SynapseML and Azure Cognitive Services on Apache Spark for multivariate anomaly detection. Multivariate anomaly detection allows for the detection of anomalies among many variables or time series, taking into account all the inter-correlations and dependencies between the different variables. In this scenario, we use SynapseML to train a model for multivariate anomaly detection using the Azure Cognitive Services, and we then use to the model to infer multivariate anomalies within a dataset containing synthetic measurements from three IoT sensors. 
# MAGIC 
# MAGIC To learn more about the Anomaly Detector Cognitive Service please refer to [ this documentation page](https://docs.microsoft.com/en-us/azure/cognitive-services/anomaly-detector/). 
# MAGIC 
# MAGIC ### Prerequisites
# MAGIC - An Azure subscription - [Create one for free](https://azure.microsoft.com/en-us/free/)
# MAGIC 
# MAGIC ### Setup
# MAGIC #### Create an Anomaly Detector resource
# MAGIC Follow the instructions below to create an `Anomaly Detector` resource using the Azure portal or alternatively, you can also use the Azure CLI to create this resource.
# MAGIC 
# MAGIC - In the Azure Portal, click `Create` in your resource group, and then type `Anomaly Detector`. Click on the Anomaly Detector resource.
# MAGIC - Give the resource a name, and ideally use the same region as the rest of your resource group. Use the default options for the rest, and then click `Review + Create` and then `Create`.
# MAGIC - Once the Anomaly Detector resource is created, open it and click on the `Keys and Endpoints` panel on the left. Copy the key for the Anomaly Detector resource into the `ANOMALY_API_KEY` environment variable, or store it in the `anomalyKey` variable in the cell below.
# MAGIC 
# MAGIC #### Create a Storage Account resource
# MAGIC In order to save intermediate data, you will need to create an Azure Blob Storage Account. Within that storage account, create a container for storing the intermediate data. Make note of the container name, and copy the connection string to that container. You will need this later to populate the `containerName` variable and the `BLOB_CONNECTION_STRING` environment variable.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Enter your service keys
# MAGIC Let's start by setting up the environment variables for our service keys. The next cell sets the `ANOMALY_API_KEY` and the `BLOB_CONNECTION_STRING` environment variables based on the values stored in our Azure Key Vault. If you are running this in your own environment, make sure you set these environment variables before you proceed.

# COMMAND ----------

import os
from pyspark.sql import SparkSession
from synapse.ml.core.platform import find_secret

# Bootstrap Spark Session
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC Now, lets read the `ANOMALY_API_KEY` and `BLOB_CONNECTION_STRING` environment variables and set the `containerName` and `location` variables.

# COMMAND ----------

# An Anomaly Dectector subscription key
anomalyKey = find_secret("anomaly-api-key")
# Your storage account name
storageName = "anomalydetectiontest"
# A connection string to your blob storage account
storageKey = find_secret("madtest-storage-key")
# A place to save intermediate MVAD results
intermediateSaveDir = (
    "wasbs://madtest@anomalydetectiontest.blob.core.windows.net/intermediateData"
)
# The location of the anomaly detector resource that you created
location = "westus2"

# COMMAND ----------

# MAGIC %md
# MAGIC First we will connect to our storage account so that anomaly detector can save intermediate results there:

# COMMAND ----------

spark.sparkContext._jsc.hadoopConfiguration().set(
    f"fs.azure.account.key.{storageName}.blob.core.windows.net", storageKey
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's import all the necessary modules.

# COMMAND ----------

import numpy as np
import pandas as pd

import pyspark
from pyspark.sql.functions import col
from pyspark.sql.functions import lit
from pyspark.sql.types import DoubleType
import matplotlib.pyplot as plt

import synapse.ml
from synapse.ml.cognitive import *

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's read our sample data into a Spark DataFrame.

# COMMAND ----------

df = (
    spark.read.format("csv")
    .option("header", "true")
    .load("wasbs://publicwasb@mmlspark.blob.core.windows.net/MVAD/sample.csv")
)

df = (
    df.withColumn("sensor_1", col("sensor_1").cast(DoubleType()))
    .withColumn("sensor_2", col("sensor_2").cast(DoubleType()))
    .withColumn("sensor_3", col("sensor_3").cast(DoubleType()))
)

# Let's inspect the dataframe:
df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC We can now create an `estimator` object, which will be used to train our model. In the cell below, we specify the start and end times for the training data. We also specify the input columns to use, and the name of the column that contains the timestamps. Finally, we specify the number of data points to use in the anomaly detection sliding window, and we set the connection string to the Azure Blob Storage Account. 

# COMMAND ----------

trainingStartTime = "2020-06-01T12:00:00Z"
trainingEndTime = "2020-07-02T17:55:00Z"
timestampColumn = "timestamp"
inputColumns = ["sensor_1", "sensor_2", "sensor_3"]

estimator = (
    SimpleFitMultivariateAnomaly()
    .setSubscriptionKey(anomalyKey)
    .setLocation(location)
    .setStartTime(trainingStartTime)
    .setEndTime(trainingEndTime)
    .setIntermediateSaveDir(intermediateSaveDir)
    .setTimestampCol(timestampColumn)
    .setInputCols(inputColumns)
    .setSlidingWindow(200)
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have created the `estimator`, let's fit it to the data:

# COMMAND ----------

model = estimator.fit(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Once the training is done, we can now use the model for inference. The code in the next cell specifies the start and end times for the data we would like to detect the anomalies in. It will then show the results.

# COMMAND ----------

inferenceStartTime = "2020-07-02T18:00:00Z"
inferenceEndTime = "2020-07-06T05:15:00Z"

result = (
    model.setStartTime(inferenceStartTime)
    .setEndTime(inferenceEndTime)
    .setOutputCol("results")
    .setErrorCol("errors")
    .setInputCols(inputColumns)
    .setTimestampCol(timestampColumn)
    .transform(df)
)

result.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC When we called `.show(5)` in the previous cell, it showed us the first five rows in the dataframe. The results were all `null` because they were not inside the inference window.
# MAGIC 
# MAGIC To show the results only for the inferred data, lets select the columns we need. We can then order the rows in the dataframe by ascending order, and filter the result to only show the rows that are in the range of the inference window. In our case `inferenceEndTime` is the same as the last row in the dataframe, so can ignore that. 
# MAGIC 
# MAGIC Finally, to be able to better plot the results, lets convert the Spark dataframe to a Pandas dataframe.
# MAGIC 
# MAGIC This is what the next cell does:

# COMMAND ----------

rdf = (
    result.select(
        "timestamp",
        *inputColumns,
        "results.interpretation",
        "isAnomaly",
        "results.severity"
    )
    .orderBy("timestamp", ascending=True)
    .filter(col("timestamp") >= lit(inferenceStartTime))
    .toPandas()
)

rdf

# COMMAND ----------

# MAGIC %md
# MAGIC Let's now format the `interpretation` column that stores the contribution score from each sensor to the detected anomalies. The next cell formats this data, and splits the contribution score of each sensor into its own column.

# COMMAND ----------

def parse(x):
    if len(x) > 0:
        return dict([item[:2] for item in x])
    else:
        return {"sensor_1": 0, "sensor_2": 0, "sensor_3": 0}


rdf["contributors"] = rdf["interpretation"].apply(parse)
rdf = pd.concat(
    [
        rdf.drop(["contributors"], axis=1),
        pd.json_normalize(rdf["contributors"]).rename(
            columns={
                "sensor_1": "series_1",
                "sensor_2": "series_2",
                "sensor_3": "series_3",
            }
        ),
    ],
    axis=1,
)
rdf

# COMMAND ----------

# MAGIC %md
# MAGIC Great! We now have the contribution scores of sensors 1, 2, and 3 in the `series_0`, `series_1`, and `series_2` columns respectively. 
# MAGIC 
# MAGIC Let's run the next cell to plot the results. The `minSeverity` parameter in the first line specifies the minimum severity of the anomalies to be plotted. 

# COMMAND ----------

minSeverity = 0.1


####### Main Figure #######
plt.figure(figsize=(23, 8))
plt.plot(
    rdf["timestamp"],
    rdf["sensor_1"],
    color="tab:orange",
    linestyle="solid",
    linewidth=2,
    label="sensor_1",
)
plt.plot(
    rdf["timestamp"],
    rdf["sensor_2"],
    color="tab:green",
    linestyle="solid",
    linewidth=2,
    label="sensor_2",
)
plt.plot(
    rdf["timestamp"],
    rdf["sensor_3"],
    color="tab:blue",
    linestyle="solid",
    linewidth=2,
    label="sensor_3",
)
plt.grid(axis="y")
plt.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
plt.legend()

anoms = list(rdf["severity"] >= minSeverity)
_, _, ymin, ymax = plt.axis()
plt.vlines(np.where(anoms), ymin=ymin, ymax=ymax, color="r", alpha=0.8)

plt.legend()
plt.title(
    "A plot of the values from the three sensors with the detected anomalies highlighted in red."
)
plt.show()

####### Severity Figure #######
plt.figure(figsize=(23, 1))
plt.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
plt.plot(
    rdf["timestamp"],
    rdf["severity"],
    color="black",
    linestyle="solid",
    linewidth=2,
    label="Severity score",
)
plt.plot(
    rdf["timestamp"],
    [minSeverity] * len(rdf["severity"]),
    color="red",
    linestyle="dotted",
    linewidth=1,
    label="minSeverity",
)
plt.grid(axis="y")
plt.legend()
plt.ylim([0, 1])
plt.title("Severity of the detected anomalies")
plt.show()

####### Contributors Figure #######
plt.figure(figsize=(23, 1))
plt.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
plt.bar(
    rdf["timestamp"], rdf["series_1"], width=2, color="tab:orange", label="sensor_1"
)
plt.bar(
    rdf["timestamp"],
    rdf["series_2"],
    width=2,
    color="tab:green",
    label="sensor_2",
    bottom=rdf["series_1"],
)
plt.bar(
    rdf["timestamp"],
    rdf["series_3"],
    width=2,
    color="tab:blue",
    label="sensor_3",
    bottom=rdf["series_1"] + rdf["series_2"],
)
plt.grid(axis="y")
plt.legend()
plt.ylim([0, 1])
plt.title("The contribution of each sensor to the detected anomaly")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC <img width="1300" src="https://mmlspark.blob.core.windows.net/graphics/mvad_plot.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC The plots above show the raw data from the sensors (inside the inference window) in orange, green, and blue. The red vertical lines in the first figure show the detected anomalies that have a severity greater than or equal to `minSeverity`. 
# MAGIC 
# MAGIC The second plot shows the severity score of all the detected anomalies, with the `minSeverity` threshold shown in the dotted red line.
# MAGIC 
# MAGIC Finally, the last plot shows the contribution of the data from each sensor to the detected anomalies. This helps us diagnose and understand the most likely cause of each anomaly.