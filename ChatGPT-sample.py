# Databricks notebook source
!pip install openai
# Install python openai

!pip install tiktoken --upgrade
# Install python tiktoken

# COMMAND ----------

import os
import openai
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
#openai.api_base = os.getenv("https://oai-frlai-demo-scus-02.openai.azure.com/")  # Your Azure OpenAI resource's endpoint value.
#openai.api_key = os.getenv("48455672c80a4a4ba702d95b76d301b3")
openai.api_base = "https://oai-frlai-demo-scus-02.openai.azure.com/"  # Your Azure OpenAI resource's endpoint value.
openai.api_key = "48455672c80a4a4ba702d95b76d301b3"

response = openai.ChatCompletion.create(
    engine="gpt-35-turbo-0301", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
    messages=[
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": "What's the difference between garbanzo beans and chickpeas?"},
    ]
)

print(response)

print(response['choices'][0]['message']['content'])


# COMMAND ----------

# DBTITLE 1,Creating a basic conversation loop
conversation=[{"role": "system", "content": "You are a helpful assistant."}]

while(True):
    user_input = input()      
    conversation.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo-0301", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
        messages = conversation
    )

    conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    print("\n" + response['choices'][0]['message']['content'] + "\n")

# COMMAND ----------

import tiktoken

system_message = {"role": "system", "content": "You are a helpful assistant."}
max_response_tokens = 250
token_limit= 4096
conversation=[]
conversation.append(system_message)

def num_tokens_from_messages(messages, model="gpt-35-turbo-0301"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

while(True):
    user_input = input("")     
    conversation.append({"role": "user", "content": user_input})
    conv_history_tokens = num_tokens_from_messages(conversation)

    while (conv_history_tokens+max_response_tokens >= token_limit):
        del conversation[1] 
        conv_history_tokens = num_tokens_from_messages(conversation)
        
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
        messages = conversation,
        temperature=.7,
        max_tokens=max_response_tokens,
    )

    conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    print("\n" + response['choices'][0]['message']['content'] + "\n")
