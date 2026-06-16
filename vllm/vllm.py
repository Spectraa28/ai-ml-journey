from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  
)

# Call the chat completions API
response = client.chat.completions.create(
    model="qwen2.5:0.5b",  
    messages=[
        {"role": "user", "content": "Why is the sky blue?"}
    ]
)

print(response.choices[0].message.content)
