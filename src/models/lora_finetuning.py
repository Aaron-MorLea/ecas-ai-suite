import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, concatenate_datasets
import yaml

class LoRAFineTuner:
    """Implements LoRA fine-tuning using Unsloth.
    Covers: LoRA, model adjustment, parameter optimization from job requirements."""

    def __init__(self, config_path: str = "configs/lora_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def load_model(self):
        """Load base model with Unsloth optimization."""
        model_config = self.config['model']
        lora_config = self.config['lora']
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config['name'],
            max_seq_length=model_config['max_seq_length'],
            load_in_4bit=model_config['load_in_4bit'],
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            target_modules=lora_config['target_modules'],
        )
        
        return model, tokenizer

    def prepare_datasets(self):
        """Load and combine datasets for training."""
        dataset1 = load_dataset("mlabonne/llmtwin")
        dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
        dataset = concatenate_datasets([dataset1, dataset2])
        return dataset

    def setup_trainer(self, model, tokenizer, train_dataset, eval_dataset):
        """Configure SFTTrainer with training arguments."""
        training_config = self.config['training']
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config['model']['max_seq_length'],
            dataset_num_proc=2,
            packing=True,
            args=TrainingArguments(
                learning_rate=training_config['learning_rate'],
                lr_scheduler_type=training_config['lr_scheduler_type'],
                per_device_train_batch_size=training_config['per_device_train_batch_size'],
                gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
                num_train_epochs=training_config['num_train_epochs'],
                fp16=training_config['fp16'],
                bf16=training_config['bf16'],
                logging_steps=training_config['logging_steps'],
                optim=training_config['optim'],
                weight_decay=training_config['weight_decay'],
                warmup_steps=training_config['warmup_steps'],
                output_dir=training_config['output_dir'],
                report_to=training_config['report_to'],
                seed=0,
            ),
        )
        return trainer

    def train(self):
        """Execute full training pipeline."""
        model, tokenizer = self.load_model()
        dataset = self.prepare_datasets()
        
        # Format dataset
        dataset = dataset.train_test_split(test_size=0.05)
        
        trainer = self.setup_trainer(
            model, 
            tokenizer, 
            dataset["train"], 
            dataset["test"]
        )
        
        trainer.train()
        
        # Save model
        model.save_pretrained_merged("models/final_model", tokenizer, save_method="merged_16bit")
        return trainer
