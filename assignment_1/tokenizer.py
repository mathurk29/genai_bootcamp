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


def get_gpt_encoder(model: str):
    return tiktoken.encoding_for_model(model)


def count_gpt_tokens(model, text):
    # encoding = tiktoken.get_encoding("o200k_base")
    encoding = get_gpt_encoder(model)
    tokens = encoding.encode(text)
    return len(tokens)


def count_mistral_tokens(text):
    # tokenizer = MistralTokenizer.v3(is_tekken=True)
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


def count_llama_tokens(text):
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer",
    )
    return len(tokenizer.encode(text))


if __name__ == "__main__":
    result = []
    for lang, text in sample_texts.items():
        for model in gpt_models:
            gpt_tokens = count_gpt_tokens(model,text)
            result.append({"lang": lang, "model": model, "tokens": gpt_tokens})
        mistral_tokens = count_mistral_tokens(text)
        result.append({"lang": lang, "model": "mistral", "tokens": mistral_tokens})
        llama_tokens = count_llama_tokens(text)
        result.append({"lang": lang, "model": "llama", "tokens": llama_tokens})

df = pd.DataFrame(result)
print(df)
