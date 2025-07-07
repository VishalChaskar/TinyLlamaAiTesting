from datasets import load_dataset

def load_and_prepare_dataset(path, tokenizer):
    dataset = load_dataset("json", data_files=path)["train"]

    def format_chat(example):
        text = f"<|system|>You are a helpful assistant.<|end|><|user|>{example['instruction']}<|end|><|assistant|>{example['output']}<|end|>"

        tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_attention_mask=True
        )

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": tokens["input_ids"]
        }

    tokenized_dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_dataset