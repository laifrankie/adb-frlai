# Databricks notebook source
# MAGIC %md
# MAGIC # Startup Investment Attribution - Understand Outreach Effort's Effect"

# COMMAND ----------

# MAGIC %md
# MAGIC ![image-alt-text](https://camo.githubusercontent.com/4ac8c931fd4600d2b466975c87fb03b439ebc7f6debd58409aea0db10457436d/68747470733a2f2f7777772e6d6963726f736f66742e636f6d2f656e2d75732f72657365617263682f75706c6f6164732f70726f642f323032302f30352f4174747269627574696f6e2e706e67)

# COMMAND ----------

# MAGIC %md
# MAGIC **This sample notebook aims to show the application of using SynapseML's DoubleMLEstimator for inferring causality using observational data.**

# COMMAND ----------

# MAGIC %md
# MAGIC A startup that sells software would like to know whether its outreach efforts were successful in attracting new customers or boosting consumption among existing customers. In other words, they would like to learn the treatment effect of each investment on customers' software usage.
# MAGIC 
# MAGIC In an ideal world, the startup would run several randomized experiments where each customer would receive a random assortment of investments. However, this can be logistically prohibitive or strategically unsound: the startup might not have the resources to design such experiments or they might not want to risk losing out on big opportunities due to lack of incentives.
# MAGIC 
# MAGIC In this customer scenario walkthrough, we show how SynapseML causal package can use historical investment data to learn the investment effect.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Background
# MAGIC In this scenario, a startup that sells software provides discounts incentives to its customer. A customer might be given or not.
# MAGIC 
# MAGIC The startup has historical data on these investments for 2,000 customers, as well as how much revenue these customers generated in the year after the investments were made. They would like to use this data to learn the optimal incentive policy for each existing or new customer in order to maximize the return on investment (ROI).
# MAGIC 
# MAGIC The startup faces a challenge:  the dataset is biased because historically the larger customers received the most incentives. Thus, they need a causal model that can remove the bias.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data
# MAGIC The data* contains ~2,000 customers and is comprised of:
# MAGIC 
# MAGIC * Customer features: details about the industry, size, revenue, and technology profile of each customer.
# MAGIC * Interventions: information about which incentive was given to a customer.
# MAGIC * Outcome: the amount of product the customer bought in the year after the incentives were given.
# MAGIC 
# MAGIC 
# MAGIC | Feature Name    | Type | Details                                                                                                                                     |
# MAGIC |-----------------|------|---------------------------------------------------------------------------------------------------------------------------------------------|
# MAGIC | Global Flag     | W    | whether the customer has global offices                                                                                                     | 
# MAGIC | Major Flag      | W    | whether the customer is a large consumer in their industry (as opposed to SMC - Small Medium Corporation - or SMB - Small Medium Business)  |
# MAGIC | SMC Flag        | W    | whether the customer is a Small Medium Corporation (SMC, as opposed to major and SMB)                                                       |
# MAGIC | Commercial Flag | W    | whether the customer's business is commercial (as opposed to public secor)                                                                  |
# MAGIC | IT Spend        | W    | $ spent on IT-related purchases                                                                                                             |
# MAGIC | Employee Count  | W    | number of employees                                                                                                                         |
# MAGIC | PC Count        | W    | number of PCs used by the customer                                                                                                          |                                                                                      |
# MAGIC | Discount        | T    | whether the customer was given a discount (binary)                                                                                          |
# MAGIC | Revenue         | Y    | $ Revenue from customer given by the amount of software purchased                                                                           |

# COMMAND ----------

from pyspark.sql import SparkSession

# Bootstrap Spark Session
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Import the sample multi-attribution data
data = (
    spark.read.format("csv")
    .option("inferSchema", True)
    .option("header", True)
    .load(
        "wasbs://publicwasb@mmlspark.blob.core.windows.net/multi_attribution_sample.csv"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Get Causal Effects with SynapseML DoubleMLEstimator

# COMMAND ----------

from synapse.ml.causal import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression

treatmentColumn = "Discount"
outcomeColumn = "Revenue"

dml = (
    DoubleMLEstimator()
    .setTreatmentModel(LogisticRegression())
    .setTreatmentCol(treatmentColumn)
    .setOutcomeModel(LinearRegression())
    .setOutcomeCol(outcomeColumn)
    .setMaxIter(20)
)

model = dml.fit(data)

# COMMAND ----------

# Get average treatment effect, it returns a numeric value, e.g. 5166.78324
# It means, on average, customers who received a discount spent $5,166 more on software
model.getAvgTreatmentEffect()

# COMMAND ----------

# Get treatment effect's confidence interval, e.g.  [4765.826181160708, 5371.2817538168965]
model.getConfidenceInterval()