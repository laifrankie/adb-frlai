# Databricks notebook source
import requests
import time
import os
api_base = 'https://oai-frlai-demo-eus-01.openai.azure.com/'
api_key = os.getenv("2148802e6bc3473fb83a8ff737f6e36b")
api_version = '2022-08-03-preview'
url = "{}dalle/text-to-image?api-version={}".format(api_base, api_version)
headers= { "api-key": api_key, "Content-Type": "application/json" }
body = {
 "caption": "watercolor painting of the Seattle skyline",
 "resolution": "1024x1024"
}


# COMMAND ----------

requests.post(url, headers=headers, json=body)
#print (submission)
#operation_location = submission.headers['operation-location']
#retry_after = submission.headers['Retry-after']

# COMMAND ----------

status = ""
while (status != "Succeeded"):
 time.sleep(int(retry_after))
 response = requests.get(operation_location, headers=headers)
 status = response.json()['status']
image_url = response.json()['result']['contentUrl']

