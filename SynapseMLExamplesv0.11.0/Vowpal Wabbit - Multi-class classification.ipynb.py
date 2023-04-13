# Databricks notebook source
# MAGIC %md
# MAGIC <img width="200" src="https://mmlspark.blob.core.windows.net/graphics/emails/vw-blue-dark-orange.svg" />
# MAGIC 
# MAGIC # Multi-class Classification using Vowpal Wabbit

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
        T.StructField("sepal_length", T.DoubleType(), False),
        T.StructField("sepal_width", T.DoubleType(), False),
        T.StructField("petal_length", T.DoubleType(), False),
        T.StructField("petal_width", T.DoubleType(), False),
        T.StructField("variety", T.StringType(), False),
    ]
)

df = (
    spark.read.format("csv")
    .option("header", True)
    .schema(schema)
    .load("wasbs://publicwasb@mmlspark.blob.core.windows.net/iris.txt")
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

from pyspark.ml.feature import StringIndexer

from synapse.ml.vw import VowpalWabbitFeaturizer

indexer = StringIndexer(inputCol="variety", outputCol="label")
featurizer = VowpalWabbitFeaturizer(
    inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    outputCol="features",
)

# label needs to be integer (0 to n)
df_label = indexer.fit(df).transform(df).withColumn("label", F.col("label").cast("int"))

# featurize data
df_featurized = featurizer.transform(df_label).select("label", "features")

display(df_featurized)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Split the dataset into train and test

# COMMAND ----------

train, test = df_featurized.randomSplit([0.8, 0.2], seed=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Training

# COMMAND ----------

from synapse.ml.vw import VowpalWabbitClassifier


model = (
    VowpalWabbitClassifier(
        numPasses=5,
        passThroughArgs="--holdout_off --oaa 3 --holdout_off --loss_function=logistic --indexing 0 -q ::",
    )
    .setNumClasses(3)
    .fit(train)
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Prediction

# COMMAND ----------

predictions = model.transform(test)

display(predictions)