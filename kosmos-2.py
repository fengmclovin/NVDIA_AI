import requests, base64

invoke_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2"

with open("soccer.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

assert len(image_b64) < 180_000, \
  "To upload larger images, use the assets API (see docs)"

headers = {
  "Authorization": "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
  "Accept": "application/json"
}

payload = {
  "messages": [
    {
      "role": "user",
      "content": f'Who is in this photo? <img src="data:image/png;base64,{image_b64}" />'
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.20,
  "top_p": 0.20
}

response = requests.post(invoke_url, headers=headers, json=payload)

print(response.json())
