import os
import requests
import ollama
import re
import time

class CreativityBenchmarkBase:
    def __init__(self, model_name, use_api):
        self.model = model_name
        self.hf_token = os.getenv("HF_TOKEN")
        self.use_api = use_api

    def _generate(self, prompt, temperature=0.7, max_tokens=2000):
        if self.use_api:
            if not self.hf_token:
                raise ValueError("Hugging Face API token not provided.")
            
            API_URL = f"https://api-inference.huggingface.co/models/{self.model}"
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "return_full_text": False,
                    "wait_for_model": True,
                    "seed": int(time.time() * 1000) % 1000000
                }
            }

            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                response.raise_for_status()
                
                response_json = response.json()
                if isinstance(response_json, list):
                    text = response_json[0].get("generated_text", "")
                elif "error" in response_json:
                    raise Exception(f"Model error: {response_json['error']}")
                else:
                    raise Exception(f"Unexpected API response format: {response_json}")

            except requests.exceptions.RequestException as e:
                if response.status_code in [503, 429]:
                    raise Exception(f"API temporary error ({response.status_code}): {response.text}")
                raise Exception(f"API request failed: {str(e)}")
                
        else:
            try:
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={"temperature": temperature, "max_tokens": max_tokens}
                )
                text = response["response"].strip()
            except Exception as e:
                raise Exception(f"Ollama generation failed: {str(e)}")

        # Remove all think/antthinking tags and their content using regex
        text = re.sub(r'^.*<\/\s*think\s*>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()

        return text