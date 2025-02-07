import pandas as pd
from adapter import (GPTTokenizerAdapter, LlamaTokenizerAdapter,
                     MistralTokenizerAdapter)
from sample_texts import sample_texts

gpt_models = ["gpt-4o", "gpt-4", "gpt-3.5"]


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
