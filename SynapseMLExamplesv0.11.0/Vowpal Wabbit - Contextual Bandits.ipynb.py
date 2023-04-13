# Databricks notebook source
# MAGIC %md
# MAGIC <img width="200" src="https://mmlspark.blob.core.windows.net/graphics/emails/vw-blue-dark-orange.svg" />
# MAGIC 
# MAGIC # Contextual-Bandits using Vowpal Wabbit
# MAGIC 
# MAGIC In the contextual bandit problem, a learner repeatedly observes a context, chooses an action, and observes a loss/cost/reward for the chosen action only. Contextual bandit algorithms use additional side information (or context) to aid real world decision-making. They work well for choosing actions in dynamic environments where options change rapidly, and the set of available actions is limite
# MAGIC 
# MAGIC An in-depth tutorial can be found [here](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Contextual_bandits_and_Vowpal_Wabbit.html)
# MAGIC 
# MAGIC [Azure Personalizer](https://azure.microsoft.com/en-us/products/cognitive-services/personalizer) emits logs in DSJSON-format. This example demonstrates how to perform off-policy evaluation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step1: Read the dataset

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
        T.StructField("input", T.StringType(), False),
    ]
)

df = (
    spark.read.format("text")
    .schema(schema)
    .load("wasbs://publicwasb@mmlspark.blob.core.windows.net/decisionservice.json")
)
# print dataset basic info
print("records read: " + str(df.count()))
print("Schema: ")
df.printSchema()

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Use VowalWabbitFeaturizer to convert data features into vector

# COMMAND ----------

from synapse.ml.vw import VowpalWabbitDSJsonTransformer

df_train = (
    VowpalWabbitDSJsonTransformer()
    .setDsJsonColumn("input")
    .transform(df)
    .withColumn("splitId", F.lit(0))
    .repartition(2)
)

# Show structured nature of rewards
df_train.printSchema()

# exclude JSON to avoid overflow
display(df_train.drop("input"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Train model
# MAGIC 
# MAGIC VowaplWabbitGeneric performs these steps:
# MAGIC 
# MAGIC * trains a model for each split (=group)
# MAGIC * synchronizes accross partitions after every split
# MAGIC * store the 1-step ahead predictions in the model

# COMMAND ----------

from synapse.ml.vw import VowpalWabbitGeneric

model = (
    VowpalWabbitGeneric()
    .setPassThroughArgs(
        "--cb_adf --cb_type mtr --clip_p 0.1 -q GT -q MS -q GR -q OT -q MT -q OS --dsjson --preserve_performance_counters"
    )
    .setInputCol("input")
    .setSplitCol("splitId")
    .setPredictionIdCol("EventId")
    .fit(df_train)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Predict and evaluate

# COMMAND ----------

df_predictions = model.getOneStepAheadPredictions()  # .show(5, False)
df_headers = df_train.drop("input")

df_headers_predictions = df_headers.join(df_predictions, "EventId")
display(df_headers_predictions)

# COMMAND ----------

from synapse.ml.vw import VowpalWabbitCSETransformer

metrics = VowpalWabbitCSETransformer().transform(df_headers_predictions)

display(metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC For each field of the reward column the metrics are calculated

# COMMAND ----------

per_reward_metrics = metrics.select("reward.*")

display(per_reward_metrics)