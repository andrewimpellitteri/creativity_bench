import os
import requests
import ollama

class CreativityBenchmarkBase:
    def __init__(self, model_name, use_api):
        self.model = model_name
        self.hf_token = os.getenv("HF_TOKEN")
        self.use_api = use_api

    def _generate(self, prompt, temperature=0.7, max_tokens=600):
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
                    "return_full_text": False
                }
            }
            response = requests.post(API_URL, headers=headers, json=payload)
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.text}")
            try:
                text = response.json()[0]["generated_text"]
            except (KeyError, IndexError) as e:
                raise Exception(f"Unexpected API response: {response.text}") from e
        else:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": temperature, "max_tokens": max_tokens}
            )
            text = response["response"].strip()
        start = text.lower().find("<think>")
        if start == -1:
            start = text.lower().find("<antthinking>")
        end = text.lower().find("</think>", start)
        if end == -1:
            end = text.lower().find("</antthinking>", start)
        text = text[:start].strip() + " " + text[end + 8:].strip()
        return text.strip()