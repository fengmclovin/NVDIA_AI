 import requests

invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-xl"

headers = {
    "Authorization": "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
    "Accept": "application/json",
}

payload = {
  "text_prompts": [
    {
      "text": "underwater world, plants, shells, creatures, high detail, sharp focus, 4k",
      "weight": 1
    },
    {
      "text": "",
      "weight": -1
    }
  ],
  "sampler": "K_DPM_2_ANCESTRAL",
  "steps": 25,
  "cfg_scale": 5,
  "seed": 0
}

# re-use connections
session = requests.Session()

response = session.post(invoke_url, headers=headers, json=payload)

response.raise_for_status()
response_body = response.json()
print(response_body)
