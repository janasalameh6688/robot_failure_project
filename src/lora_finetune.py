
from pathlib import Path
import math

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
ADAPTER_PATH = ROOT / "models" / "lora_maintenance_adapter"
REPORT_PATH = ROOT / "reports" / "lora_comparison.txt"

MODEL_NAME = "sshleifer/tiny-gpt2"
SAMPLE_SIZE = 250
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 1
SEED = 42


def make_example(row):
    failure = int(row["Machine failure"]) == 1

    if failure:
        recommendation = (
            "The dataset marks this machine as a failure. "
            "Inspect the machine and review its maintenance history."
        )
    else:
        recommendation = (
            "The dataset does not mark this machine as a failure. "
            "Continue scheduled monitoring and routine maintenance."
        )

    return (
        "### Instruction:\n"
        "Summarize this machine's sensor readings and provide a cautious "
        "maintenance recommendation.\n"
        "### Sensor data:\n"
        f"Type: {row['Type']}. "
        f"Air temperature: {row['Air temperature [K]']} K. "
        f"Process temperature: {row['Process temperature [K]']} K. "
        f"Rotational speed: {row['Rotational speed [rpm]']} rpm. "
        f"Torque: {row['Torque [Nm]']} Nm. "
        f"Tool wear: {row['Tool wear [min]']} minutes.\n"
        "### Response:\n"
        f"{recommendation}"
    )


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.items = []

        for text in texts:
            encoded = tokenizer(
                text,
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.items.append((input_ids, attention_mask, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def evaluate(model, loader, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
            losses.append(output.loss.item())

    model.train()
    return sum(losses) / len(losses)


def main():
    torch.manual_seed(SEED)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    # Sample across both target classes where possible.
    per_class = max(1, SAMPLE_SIZE // 2)
    parts = []

    for _, group in df.groupby("Machine failure"):
        parts.append(group.sample(
            n=min(len(group), per_class),
            random_state=SEED,
        ))

    sample = pd.concat(parts).sample(frac=1, random_state=SEED)
    texts = [make_example(row) for _, row in sample.iterrows()]

    split = max(1, int(len(texts) * 0.8))
    train_texts = texts[:split]
    val_texts = texts[split:]

    if not val_texts:
        raise ValueError("Not enough records for validation.")

    print(f"Sample records: {len(texts)}")
    print(f"Training examples: {len(train_texts)}")
    print(f"Validation examples: {len(val_texts)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Base model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = TextDataset(train_texts, tokenizer)
    val_dataset = TextDataset(val_texts, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    # Measure the base model before fine-tuning.
    before_loss = evaluate(model, val_loader, device)
    print(f"Base validation loss: {before_loss:.4f}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
        fan_in_fan_out=True,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4,
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for step, (input_ids, attention_mask, labels) in enumerate(
            train_loader, start=1
        ):
            optimizer.zero_grad()

            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )

            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += output.loss.item()

            if step % 10 == 0:
                print(
                    f"Epoch {epoch + 1}, step {step}, "
                    f"loss {output.loss.item():.4f}"
                )

        print(
            f"Epoch {epoch + 1} average training loss: "
            f"{total_loss / len(train_loader):.4f}"
        )

    after_loss = evaluate(model, val_loader, device)

    ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)

    before_ppl = math.exp(min(before_loss, 20))
    after_ppl = math.exp(min(after_loss, 20))

    report = (
        "LoRA fine-tuning experiment\n"
        "===========================\n"
        f"Base model: {MODEL_NAME}\n"
        f"Training examples: {len(train_texts)}\n"
        f"Validation examples: {len(val_texts)}\n"
        f"Epochs: {EPOCHS}\n"
        f"Device: {device}\n"
        f"Base validation loss: {before_loss:.4f}\n"
        f"Fine-tuned validation loss: {after_loss:.4f}\n"
        f"Base perplexity: {before_ppl:.4f}\n"
        f"Fine-tuned perplexity: {after_ppl:.4f}\n"
        f"Adapter saved to: {ADAPTER_PATH}\n"
        "\n"
        "Perplexity is calculated from token-level loss. "
        "Lower is better on the same evaluation data, but this "
        "small synthetic-template experiment does not establish "
        "real-world maintenance accuracy or causal relationships.\n"
    )

    REPORT_PATH.write_text(report, encoding="utf-8")
    print("\n" + report)
    print("LoRA experiment completed.")


if __name__ == "__main__":
    main()
