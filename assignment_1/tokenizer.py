import tiktoken
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import LlamaTokenizerFast
import pandas as pd

sample_texts = {
    "english": "The cat sat on the windowsill, watching the rain fall. Suddenly, a flash of lightning lit up the sky, startling the little creature. It leaped down and scurried to its favorite hiding spot under the bed.",
    "spanish": "El gato estaba sentado en el alféizar de la ventana, mirando la lluvia caer. De repente, un relámpago iluminó el cielo, sobresaltando a la pequeña criatura. Saltó y corrió a su escondite favorito debajo de la cama.",
    "arabic": "جلست القطة على حافة النافذة، تراقب هطول المطر. وفجأة، أضاءت ومضة من البرق السماء، مما أثار ذهول المخلوق الصغير. قفزت إلى أسفل وهرعت إلى مكان اختبائها المفضل تحت السرير.",
    "hindi": "बिल्ली खिड़की पर बैठी हुई बारिश को देख रही थी । अचानक, आसमान में बिजली चमकी, जिससे छोटा जीव चौंक गया । वह नीचे कूद गई और बिस्तर के नीचे अपनी पसंदीदा छिपने की जगह पर भाग गई।",
}

gpt_models = ["gpt-4o", "gpt-4", "gpt-3.5"]

# Adapter interface
class TokenizerAdapter:
    def count_tokens(self, text):
        raise NotImplementedError

# GPT Tokenizer Adapter
class GPTTokenizerAdapter(TokenizerAdapter):
    def __init__(self, model_name):
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

# Mistral Tokenizer Adapter
class MistralTokenizerAdapter(TokenizerAdapter):
    def count_tokens(self, text):
        model_name = "open-mixtral-8x22b"
        tokenizer = MistralTokenizer.from_model(model_name, strict=True)
        tokenized = tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                messages=[
                    UserMessage(content=text),
                ],
                model=model_name,
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

if __name__ == "__main__":
    result = []
    for lang, text in sample_texts.items():
        for model in gpt_models:
            gpt_adapter = GPTTokenizerAdapter(model)
            gpt_tokens = gpt_adapter.count_tokens(text)
            result.append({"lang": lang, "model": model, "tokens": gpt_tokens})
        
        mistral_adapter = MistralTokenizerAdapter()
        mistral_tokens = mistral_adapter.count_tokens(text)
        result.append({"lang": lang, "model": "mistral", "tokens": mistral_tokens})
        
        llama_adapter = LlamaTokenizerAdapter()
        llama_tokens = llama_adapter.count_tokens(text)
        result.append({"lang": lang, "model": "llama", "tokens": llama_tokens})

    df = pd.DataFrame(result)
    print(df)
