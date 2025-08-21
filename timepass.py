# 1. Import the OpenAI library

from openai import OpenAI

# 2. Create a client instance pointing to the local server
client = OpenAI(
    base_url="http://localhost:1234/v1",  # The address of your LM Studio server
    api_key="not-needed"                  # API key is not required for local servers
)

# 3. Create the chat completion request
completion = client.chat.completions.create(
  # The model name doesn't matter, as the server has only one model loaded.
  # You can use any string here; it's a placeholder.
  model="local-model",
  messages=[
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain the importance of the M4 chip's unified memory for running LLMs."}
  ]
)

# 4. Print the response from the model
print(completion.choices[0].message.content)