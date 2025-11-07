from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="EMPTY"
)

def ask_llm(prompt: str):
    resp = client.chat.completions.create(
        model="Qwen/Qwen3-14B-AWQ", # ,
        messages=[
            {
                "role": "system",
                "content": ""
            },
            {"role": "user", "content": prompt}
        ]
    )
    response = resp.choices[0].message.content
    return response