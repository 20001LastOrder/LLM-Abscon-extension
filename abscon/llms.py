import os
from typing import Any, Dict, Iterator, List, Optional

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.outputs import GenerationChunk
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from loguru import logger


class SelfHostedLLM(LLM):
    default_model: str = "Llama3.1-70b"
    llm_config: dict = {}
    llmApi: Any = None
    counter: int = 0
    url: str = os.environ.get("HOSTED_LLM_URL", "")
    token: str = os.environ.get("HOSTED_LLM_TOKEN", "")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _call(
        self,
        prompt: str = "",
        messages: list[dict] = [],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        self.counter += 1
        logger.debug(prompt)

        if len(messages) == 0:
            messages = [{"role": "user", "content": prompt}]

        params = {"messages": messages, "model": self.default_model, **self.llm_config}

        headers = {"Authorization": f"Bearer {self.token}"}

        response = requests.post(self.url, headers=headers, json=params, verify=False)

        if not response.ok:
            raise ValueError(f"Request to LLM failed: {response.text}")

        logger.debug(response.content)

        response = response.json()

        result = response["choices"][0]["message"]["content"]

        logger.debug(result)
        return result

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for char in self.llmApi.request(prompt, "", True):
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": f"HostedLLM {self.default_model}",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging 
        purposes only."""
        return f"HostedLLM: {self.default_model}"


def get_llm(args):
    if args.llm_type == "reasoning":
        # ChatDeepSeek can be used to retrieve the reasoning content
        llm = ChatDeepSeek(
            api_base=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=args.llm_name,
            temperature=args.temperature,
            # make sure only specific quantized models are used when using OpenRouter
            extra_body={"provider": {"quantizations": ["bf16", "fp16"]}},
        )
    else:
        llm = ChatOpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            model=args.llm_name,
            temperature=args.temperature,
        )

    return llm