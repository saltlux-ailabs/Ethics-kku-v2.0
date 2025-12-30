import threading

from typing import List, Dict
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.vllm_client import VllmClient

class EthicAgent:
    def __init__(self, vllm_client: VllmClient):
        self.vllm_client = vllm_client
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(vllm_client.model)
        self.lock = threading.Lock()


    def _generate(
            self, prompt: str,
            max_message_length: int=4096,
            temperature: float=0.0,
    ):
        # label 생성할 때에는 최대한 사실적으로 말해야 하기 때문에, temperature를 0으로 설정
        messages: List[Dict] = [
            {
                'role': 'system',
                'content': 'You are a helpful culture norm expert.',
            },
            {
                'role': 'user',
                'content': prompt,
            }
        ]
        return self.vllm_client.call_chat(messages, temperature=temperature)

    def batch_generate(
        self, system_prompts: List[str], user_prompts: List[str],
        max_message_length: int=4096,
        temperature: float=0.0,
    ):
        messages_list: List[List[Dict]] = [
            [
                {
                    'role': 'system',
                    'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': user_prompt,
                }
            ]
            for system_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]
        return self.vllm_client.batch_call_chat(messages_list, temperature=temperature)