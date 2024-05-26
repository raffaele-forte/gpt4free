from __future__ import annotations

from .Openai import Openai
from ...typing import AsyncResult, Messages

class Groq(Openai):
    label = "Groq"
    url = "https://console.groq.com/playground"
    working = True
    default_model = "llama3-8b-8192"
    models = ["mixtral-8x7b-32768", "llama3-8b-8192", "llama3-70b-8192", "gemma-7b-it"]
    model_aliases = {"mixtral-8x7b": "mixtral-8x7b-32768", "llama3-8b": "llama3-8b-8192", "llama3-70b": "llama3-70b-8192"}

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_base: str = "https://api.groq.com/openai/v1",
        **kwargs
    ) -> AsyncResult:
        return super().create_async_generator(
            model, messages, api_base=api_base, **kwargs
        )
