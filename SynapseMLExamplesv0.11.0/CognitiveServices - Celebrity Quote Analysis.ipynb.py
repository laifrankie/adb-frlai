# Databricks notebook source
# MAGIC %md
# MAGIC # Celebrity Quote Analysis with The Cognitive Services on Spark

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://mmlspark.blob.core.windows.net/graphics/SparkSummit2/cog_services.png" width="800" style="float: center;"/>

# COMMAND ----------

from synapse.ml.cognitive import *
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, udf
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import SparkSession
from synapse.ml.core.platform import find_secret

spark = SparkSession.builder.getOrCreate()

# put your service keys here
cognitive_key = find_secret("cognitive-api-key")
cognitive_location = "eastus"
bing_search_key = find_secret("bing-search-key")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extracting celebrity quote images using Bing Image Search on Spark
# MAGIC 
# MAGIC Here we define two Transformers to extract celebrity quote images.
# MAGIC 
# MAGIC <img src="https://mmlspark.blob.core.windows.net/graphics/Cog%20Service%20NB/step%201.png" width="600" style="float: center;"/>

# COMMAND ----------

imgsPerBatch = 10  # the number of images Bing will return for each query
offsets = [
    (i * imgsPerBatch,) for i in range(100)
]  # A list of offsets, used to page into the search results
bingParameters = spark.createDataFrame(offsets, ["offset"])

bingSearch = (
    BingImageSearch()
    .setSubscriptionKey(bing_search_key)
    .setOffsetCol("offset")
    .setQuery("celebrity quotes")
    .setCount(imgsPerBatch)
    .setOutputCol("images")
)

# Transformer to that extracts and flattens the richly structured output of Bing Image Search into a simple URL column
getUrls = BingImageSearch.getUrlTransformer("images", "url")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recognizing Images of Celebrities
# MAGIC This block identifies the name of the celebrities for each of the images returned by the Bing Image Search.
# MAGIC 
# MAGIC <img src="https://mmlspark.blob.core.windows.net/graphics/Cog%20Service%20NB/step%202.png" width="600" style="float: center;"/>

# COMMAND ----------

celebs = (
    RecognizeDomainSpecificContent()
    .setSubscriptionKey(cognitive_key)
    .setLocation(cognitive_location)
    .setModel("celebrities")
    .setImageUrlCol("url")
    .setOutputCol("celebs")
)

# Extract the first celebrity we see from the structured response
firstCeleb = SQLTransformer(
    statement="SELECT *, celebs.result.celebrities[0].name as firstCeleb FROM __THIS__"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading the quote from the image.
# MAGIC This stage performs OCR on the images to recognize the quotes.
# MAGIC 
# MAGIC <img src="https://mmlspark.blob.core.windows.net/graphics/Cog%20Service%20NB/step%203.png" width="600" style="float: center;"/>

# COMMAND ----------

from synapse.ml.stages import UDFTransformer

recognizeText = (
    RecognizeText()
    .setSubscriptionKey(cognitive_key)
    .setLocation(cognitive_location)
    .setImageUrlCol("url")
    .setMode("Printed")
    .setOutputCol("ocr")
    .setConcurrency(5)
)


def getTextFunction(ocrRow):
    if ocrRow is None:
        return None
    return "\n".join([line.text for line in ocrRow.recognitionResult.lines])


# this transformer wil extract a simpler string from the structured output of recognize text
getText = (
    UDFTransformer()
    .setUDF(udf(getTextFunction))
    .setInputCol("ocr")
    .setOutputCol("text")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding the Sentiment of the Quote
# MAGIC 
# MAGIC <img src="https://mmlspark.blob.core.windows.net/graphics/Cog%20Service%20NB/step4.jpg" width="600" style="float: center;"/>

# COMMAND ----------

sentimentTransformer = (
    TextSentiment()
    .setLocation(cognitive_location)
    .setSubscriptionKey(cognitive_key)
    .setTextCol("text")
    .setOutputCol("sentiment")
)

# Extract the sentiment score from the API response body
getSentiment = SQLTransformer(
    statement="SELECT *, sentiment.document.sentiment as sentimentLabel FROM __THIS__"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tying it all together
# MAGIC 
# MAGIC Now that we have built the stages of our pipeline it's time to chain them together into a single model that can be used to process batches of incoming data
# MAGIC 
# MAGIC <img src="https://mmlspark.blob.core.windows.net/graphics/Cog%20Service%20NB/full_pipe_2.jpg" width="800" style="float: center;"/>

# COMMAND ----------

from synapse.ml.stages import SelectColumns

# Select the final coulmns
cleanupColumns = SelectColumns().setCols(
    ["url", "firstCeleb", "text", "sentimentLabel"]
)

celebrityQuoteAnalysis = PipelineModel(
    stages=[
        bingSearch,
        getUrls,
        celebs,
        firstCeleb,
        recognizeText,
        getText,
        sentimentTransformer,
        getSentiment,
        cleanupColumns,
    ]
)

celebrityQuoteAnalysis.transform(bingParameters).show(5)