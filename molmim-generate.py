import requests

invoke_url = "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate"

headers = {
    "Authorization": "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
    "Accept": "application/json",
}

payload = {
  "algorithm": "CMA-ES",
  "num_molecules": 30,
  "property_name": "QED",
  "minimize": False,
  "min_similarity": 0.3,
  "particles": 30,
  "iterations": 10,
  "smi": "[H][C@@]12Cc3c[nH]c4cccc(C1=C[C@H](NC(=O)N(CC)CC)CN2C)c34"
}

# re-use connections
session = requests.Session()

response = session.post(invoke_url, headers=headers, json=payload)

response.raise_for_status()
response_body = response.json()
print(response_body)
