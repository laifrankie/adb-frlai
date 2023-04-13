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
# MAGIC #### Reformat into VW-native format
# MAGIC See VW [docs](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format) for format details

# COMMAND ----------

# create VW string format
cols = [
    F.col("label"),
    F.lit("|"),
    *[F.col("i" + str(i)) for i in range(1, 13)],
    *[F.col("s" + str(i)) for i in range(26)],
]

df = df.select(F.concat_ws(" ", *cols).alias("value"))
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Split the dataset into train and test

# COMMAND ----------

train, test = df.randomSplit([0.6, 0.4], seed=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Training

# COMMAND ----------

from synapse.ml.vw import VowpalWabbitGeneric

# number of partitions determines data parallelism
train = train.repartition(2)

model = VowpalWabbitGeneric(
    numPasses=5,
    passThroughArgs="--holdout_off --loss_function logistic --link logistic",
).fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Prediction

# COMMAND ----------

predictions = model.transform(test)

predictions = predictions.withColumn(
    "prediction", F.col("prediction").cast("double")
).withColumn("label", F.substring("value", 0, 1).cast("double"))

display(predictions)

# COMMAND ----------

from synapse.ml.train import ComputeModelStatistics

metrics = ComputeModelStatistics(
    evaluationMetric="classification", labelCol="label", scoredLabelsCol="prediction"
).transform(predictions)
display(metrics)