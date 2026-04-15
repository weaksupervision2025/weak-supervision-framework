import simplejson as json
from loguru import logger
import openai


class LLMQueryClient:
    def __init__(self, gpt_base_url, gpt_key, vllm_base_url, vllm_key) -> None:

        self.openai_client = openai.Client(
            base_url=gpt_base_url,
            api_key=gpt_key,
        )
        # openai_client = openai.Client(
        #     base_url="https://ai-gateway-api.query.consul-test/api/v1/compatible/openai",
        #     api_key="RemiJJpjdrYYC7ubXsKXgTOzChL3RJIMYVecQfqjH5UsJfqBv/391cb/zFtyggUo",
        # )

        self.vllm_client = openai.Client(
            base_url=vllm_base_url,
            api_key=vllm_key,
        )
        
    def _generate_vllm(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        schema: dict | None,
        max_tokens: int = 2048,
    ) -> str:
        if schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": False,
                },
            }
        else:
            response_format = {"type": "json_object"}
        completion = self.vllm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        return completion.choices[0].message.content
    
    def _generate_openai(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        schema: dict | None,
        max_tokens: int = 2048,
    ) -> str:
        if self.openai_client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        if schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": True,
                },
            }
        else:
            response_format = {"type": "json_object"}
        completion = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            #temperature=temperature,
            #max_tokens=max_tokens,
            response_format=response_format,
            reasoning_effort="minimal",
        )

        # completion = openai_client.chat.completions.create(
        #     model="openrouter/google/gemini-3-pro-preview",
        #     messages=[{"role": "system", "content": "you are a robot"}, {"role": "user", "content": "hi"}],
        #     response_format=response_format,
        # )
        #
        # print(completion.choices[0].message.content)
        return completion.choices[0].message.content

    def generate(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        schema: dict | None = None,
        max_tokens: int = 4096,
    ) -> str:
        if ("gpt" in model) or ("openrouter" in model):
            content = self._generate_openai(
                messages,
                model,
                temperature,
                schema,
                max_tokens=max_tokens,
            )
        else:
            content = self._generate_vllm(
                messages,
                model,
                temperature,
                schema,
                max_tokens=max_tokens,
            )
        content = '{' + '{'.join(content.split('{')[1:])
        content = '}'.join(content.split('}')[:-1]) + '}'
        return json.loads(content, strict=False)
