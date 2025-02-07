from abc import ABC, abstractmethod

import tiktoken
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import LlamaTokenizerFast

# Adapter interface


class TokenizerAdapter(ABC):
    @abstractmethod
    def count_tokens(self, text):
        pass


# GPT Tokenizer Adapter
class GPTTokenizerAdapter(TokenizerAdapter):
    def __init__(self, model_name):
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))


# Mistral Tokenizer Adapter
class MistralTokenizerAdapter(TokenizerAdapter):
    def __init__(self):
        self.model_name = "open-mixtral-8x22b"
        self.tokenizer = MistralTokenizer.from_model(self.model_name, strict=True)

    def count_tokens(self, text):
        tokenized = self.tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                messages=[
                    UserMessage(content=text),
                ],
                model=self.model_name,
            )
        )
        return len(tokenized.tokens)


# Llama Tokenizer Adapter
class LlamaTokenizerAdapter(TokenizerAdapter):
    def __init__(self):
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer",
        )

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))
