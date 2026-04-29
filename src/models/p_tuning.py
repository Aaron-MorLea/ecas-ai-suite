import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
import yaml

class PTuningFineTuner:
    """Implements P-tuning for efficient fine-tuning.
    Covers: P-tuning from job requirements."""

    def __init__(self, config_path: str = "configs/lora_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.p_tuning_config = self.config.get('p_tuning', {})

    def setup_prompt_tuning(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B"):
        """Configure P-tuning (soft prompt tuning)."""
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # P-tuning configuration
        peft_config = PromptTuningConfig(
            task_type="CAUSAL_LM",
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=self.p_tuning_config.get("num_virtual_tokens", 20),
            prompt_tuning_init_text=self.p_tuning_config.get(
                "prompt_tuning_init_text", 
                "Classify the following e-commerce query:"
            ),
            tokenizer_name_or_path=model_name,
        )
        
        model = get_peft_model(model, peft_config)
        return model, tokenizer

    def train_p_tuning(self, train_dataset, eval_dataset):
        """Train with P-tuning approach."""
        from transformers import Trainer, TrainingArguments
        
        model, tokenizer = self.setup_prompt_tuning()
        
        training_args = TrainingArguments(
            output_dir="./models/p_tuning_output",
            learning_rate=3e-4,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
        
        trainer.train()
        return trainer
