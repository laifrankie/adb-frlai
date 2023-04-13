# Databricks notebook source
# MAGIC %md
# MAGIC ## Deep Learning - Deep Text Classifier

# COMMAND ----------

# MAGIC %md
# MAGIC ### Environment Setup on databricks

# COMMAND ----------

# install cloudpickle 2.0.0 to add synapse module for usage of horovod
%pip install cloudpickle==2.0.0 --force-reinstall --no-deps

# COMMAND ----------

import synapse
import cloudpickle

cloudpickle.register_pickle_by_value(synapse)

# COMMAND ----------

! horovodrun --check-build

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read Dataset

# COMMAND ----------

import urllib

urllib.request.urlretrieve(
    "https://mmlspark.blob.core.windows.net/publicwasb/text_classification/Emotion_classification.csv",
    "/tmp/Emotion_classification.csv",
)

import pandas as pd
from pyspark.ml.feature import StringIndexer

df = pd.read_csv("/tmp/Emotion_classification.csv")
df = spark.createDataFrame(df)

indexer = StringIndexer(inputCol="Emotion", outputCol="label")
indexer_model = indexer.fit(df)
df = indexer_model.transform(df).drop(("Emotion"))

train_df, test_df = df.randomSplit([0.85, 0.15], seed=1)
display(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training

# COMMAND ----------

from horovod.spark.common.store import DBFSLocalStore
from pytorch_lightning.callbacks import ModelCheckpoint
from synapse.ml.dl import *

checkpoint = "bert-base-uncased"
run_output_dir = f"/dbfs/FileStore/test/{checkpoint}"
store = DBFSLocalStore(run_output_dir)

epochs = 1

callbacks = [ModelCheckpoint(filename="{epoch}-{train_loss:.2f}")]

# COMMAND ----------

deep_text_classifier = DeepTextClassifier(
    checkpoint=checkpoint,
    store=store,
    callbacks=callbacks,
    num_classes=6,
    batch_size=16,
    epochs=epochs,
    validation=0.1,
    text_col="Text",
)

deep_text_model = deep_text_classifier.fit(train_df.limit(6000).repartition(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediction

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

pred_df = deep_text_model.transform(test_df.limit(500))
evaluator = MulticlassClassificationEvaluator(
    predictionCol="prediction", labelCol="label", metricName="accuracy"
)
print("Test accuracy:", evaluator.evaluate(pred_df))