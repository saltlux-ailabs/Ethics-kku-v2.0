import os
import openai
import time
import random

from typing import List
from concurrent.futures import ThreadPoolExecutor

os.environ["OPENAI_API_KEY"] = "your_key"

class GPT4:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI()
        self.model = model

    def _generate(self, system_prompt: str, user_prompt: str, temperature: float, max_retries: int = 3):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    # temperature=temperature
                )
                return response.choices[0].message.content
            except openai.InternalServerError as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Server error, retrying in {wait_time:.2f} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts: {e}")
                    raise e
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise e

    def batch_generate(self, system_prompts: List[str], user_prompts: List[str], temperature: float, num_workers: int = 8) -> List[str]:
        # Reduce workers to avoid rate limiting
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(lambda system_prompt, user_prompt: self._generate(system_prompt, user_prompt, temperature), system_prompts, user_prompts))
        return results