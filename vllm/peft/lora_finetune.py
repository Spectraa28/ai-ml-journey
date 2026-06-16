import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from trl import SFTTrainer , SFTConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float32,device_map="cpu")

print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj","v_proj"]
)

model = get_peft_model(model,lora_config)
model.print_trainable_parameters()


data = [
    {"messages": [
        {"role": "system", "content": "You are a legal document assistant."},
        {"role": "user", "content": "What is a penalty clause?"},
        {"role": "assistant", "content": "A penalty clause specifies financial consequences for breach of contract."}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a legal document assistant."},
        {"role": "user", "content": "What is indemnification?"},
        {"role": "assistant", "content": "Indemnification is a contractual obligation to compensate another party for losses."}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a legal document assistant."},
        {"role": "user", "content": "What is force majeure?"},
        {"role": "assistant", "content": "Force majeure excuses a party from contractual obligations due to extraordinary events beyond their control."}
    ]},
]

dataset = Dataset.from_list(data)
print(f"Dataset size: {len(dataset)} examples")

training_config = SFTConfig(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    use_cpu=True
)

trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()