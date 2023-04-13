# Databricks notebook source
# MAGIC %md
# MAGIC <img width="200" src="https://mmlspark.blob.core.windows.net/graphics/emails/vw-blue-dark-orange.svg" />
# MAGIC 
# MAGIC # Binary Classification with VowalWabbit on Criteo Dataset 

# COMMAND ----------

# MAGIC %md
# MAGIC ## SparkML Vector input

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read dataset

# COMMAND ----------

from pyspark.sql import SparkSession

# Bootstrap Spark Session
spark = SparkSession.builder.getOrCreate()

from synapse.ml.core.platform import *

from synapse.ml.core.platform import materializing_display as display

# COMMAND ----------

import pyspark.sql.types as T
from pyspark.sql import functions as F

schema = T.StructType(
    [
        T.StructField("label", T.IntegerType(), True),
        *[T.StructField("i" + str(i), T.IntegerType(), True) for i in range(1, 13)],
        *[T.StructField("s" + str(i), T.StringType(), True) for i in range(26)],
    ]
)

df = (
    spark.read.format("csv")
    .option("header", False)
    .option("delimiter", "\t")
    .schema(schema)
    .load("wasbs://publicwasb@mmlspark.blob.core.windows.net/criteo_day0_1k.csv.gz")
)
# print dataset basic info
print("records read: " + str(df.count()))
print("Schema: ")
df.printSchema()

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Use VowalWabbitFeaturizer to convert data features into vector

# COMMAND ----------

from synapse.ml.vw import VowpalWabbitFeaturizer

featurizer = VowpalWabbitFeaturizer(
    inputCols=[
        *["i" + str(i) for i in range(1, 13)],
        *["s" + str(i) for i in range(26)],
    ],
    outputCol="features",
)

df = featurizer.transform(df).select("label", "features")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Split the dataset into train and test

# COMMAND ----------

train, test = df.randomSplit([0.85, 0.15], seed=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Training

# COMMAND ----------

from synapse.ml.vw import VowpalWabbitClassifier

model = VowpalWabbitClassifier(
    numPasses=20,
    labelCol="label",
    featuresCol="features",
    passThroughArgs="--holdout_off --loss_function logistic",
).fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Prediction

# COMMAND ----------

predictions = model.transform(test)
display(predictions)

# COMMAND ----------

from synapse.ml.train import ComputeModelStatistics

metrics = ComputeModelStatistics(
    evaluationMetric="classification", labelCol="label", scoredLabelsCol="prediction"
).transform(predictions)
display(metrics)