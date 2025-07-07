# LLM Fine-tuning Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Fine-tuning Strategies Comparison](#fine-tuning-strategies-comparison)
3. [Technical Specifications](#technical-specifications)
4. [Implementation Approaches](#implementation-approaches)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Cost Analysis](#cost-analysis)
8. [Production Deployment](#production-deployment)

## Introduction

Fine-tuning large language models (LLMs) has become essential for organizations seeking to adapt general-purpose models for specific domains, tasks, or behavioral patterns. This guide provides comprehensive coverage of modern fine-tuning approaches, from parameter-efficient methods to full model retraining, with practical implementation strategies for production environments.

### When to Fine-tune vs. Alternatives

Before diving into fine-tuning, consider these alternatives:
- **Prompt Engineering**: For simple task adaptation
- **Few-shot Learning**: When you have limited examples
- **Retrieval-Augmented Generation (RAG)**: For knowledge-intensive tasks
- **Fine-tuning**: When you need consistent behavior, domain-specific knowledge, or custom output formats

## Fine-tuning Strategies Comparison

### 1. Parameter-Efficient Fine-tuning (PEFT)

#### LoRA (Low-Rank Adaptation)
**Best for**: Resource-constrained environments, multiple task adaptation
**Memory savings**: 90-95% reduction in trainable parameters
**Performance**: 95-99% of full fine-tuning performance

```python
# LoRA configuration example
lora_config = {
    "r": 16,  # Rank of adaptation
    "lora_alpha": 32,  # LoRA scaling parameter
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.1,
    "bias": "none"
}
```

**Use cases**:
- Multi-tenant applications
- Rapid prototyping
- Domain adaptation with limited resources

#### QLoRA (Quantized LoRA)
**Best for**: Extreme resource constraints, consumer hardware
**Memory savings**: 99% reduction from full fine-tuning
**Performance**: 90-95% of full fine-tuning performance

```python
# QLoRA configuration
quantization_config = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4"
}
```

**Use cases**:
- Single GPU fine-tuning
- Edge deployment
- Cost-sensitive applications

#### Adapters
**Best for**: Multi-task learning, modular architectures
**Memory savings**: 80-90% reduction
**Performance**: 85-95% of full fine-tuning

**Use cases**:
- Multi-domain applications
- A/B testing different behaviors
- Incremental learning

### 2. Full Fine-tuning

**Best for**: Maximum performance, extensive domain shift
**Memory requirements**: Full model parameters + gradients + optimizer states
**Performance**: Baseline performance (100%)

**When to use**:
- Critical applications requiring maximum accuracy
- Significant domain shift from pre-trained data
- Abundant computational resources
- Long-term production deployment

### 3. Instruction Tuning

**Best for**: Improving model following capabilities
**Data requirements**: High-quality instruction-response pairs
**Performance**: Significant improvement in task following

```python
# Instruction tuning data format
instruction_data = {
    "instruction": "Summarize the following text in 2 sentences:",
    "input": "Long text to summarize...",
    "output": "Concise summary response..."
}
```

**Use cases**:
- Customer service chatbots
- Task-specific assistants
- Improving model compliance

### 4. RLHF (Reinforcement Learning from Human Feedback)

**Best for**: Alignment with human preferences
**Complexity**: High (requires reward model training)
**Performance**: Superior alignment with human values

**Phases**:
1. **Supervised Fine-tuning (SFT)**: Initial instruction following
2. **Reward Model Training**: Learning human preferences
3. **PPO Training**: Policy optimization using reward signals

**Use cases**:
- Conversational AI
- Content generation with safety constraints
- Reducing harmful or biased outputs

## Technical Specifications

### Quantization Methods

#### 4-bit Quantization (NF4)
```python
# NF4 quantization setup
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

**Benefits**: 4x memory reduction, minimal performance loss
**Considerations**: Requires bitsandbytes library, CUDA support

#### 8-bit Quantization
```python
# 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)
```

**Benefits**: 2x memory reduction, better stability than 4-bit
**Use case**: Balance between memory savings and precision

### Data Requirements

#### Dataset Size Guidelines
- **LoRA/QLoRA**: 1,000-10,000 examples
- **Full Fine-tuning**: 10,000-100,000+ examples
- **Instruction Tuning**: 5,000-50,000 instruction pairs
- **RLHF**: 10,000+ preference pairs

#### Data Quality Checklist
- [ ] Consistent formatting
- [ ] Diverse examples covering edge cases
- [ ] Balanced class distribution
- [ ] Cleaned and deduplicated
- [ ] Proper validation/test splits


## Implementation Approaches

### 1. LoRA Implementation

#### Setup and Configuration
```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA
model = get_peft_model(model, lora_config)
```

#### Training Loop
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train()
```

### 2. QLoRA Implementation

```python
import torch
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# Quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Apply LoRA to quantized model
model = get_peft_model(model, lora_config)
```

### 3. Full Fine-tuning Implementation

```python
# Full fine-tuning requires more careful memory management
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader

# Optimized training arguments for full fine-tuning
training_args = TrainingArguments(
    output_dir="./full-finetune",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=500,
    logging_steps=50,
    save_strategy="steps",
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
    learning_rate=5e-6,
    weight_decay=0.01,
    fp16=True,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    optim="adamw_torch_fused"
)
```

### 4. Instruction Tuning Implementation

```python
# Data preprocessing for instruction tuning
def format_instruction_data(examples):
    prompts = []
    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        prompts.append(prompt)
    return {"text": prompts}

# Apply formatting
train_dataset = train_dataset.map(format_instruction_data, batched=True)
```

### 5. RLHF Implementation

```python
from trl import PPOTrainer, PPOConfig
from transformers import pipeline

# Phase 1: Supervised Fine-tuning (already covered above)

# Phase 2: Reward Model Training
def train_reward_model(model, tokenizer, preference_dataset):
    # Implementation details for reward model training
    pass

# Phase 3: PPO Training
ppo_config = PPOConfig(
    model_name="sft-model",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    reward_model=reward_model
)
```

## Performance Optimization

### Memory Optimization Techniques

#### Gradient Checkpointing
```python
# Enable gradient checkpointing to trade compute for memory
model.gradient_checkpointing_enable()

# In training arguments
training_args = TrainingArguments(
    gradient_checkpointing=True,
    dataloader_pin_memory=False,
    # ...
)
```

#### DeepSpeed Integration
```python
# DeepSpeed ZeRO configuration
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        }
    },
    "fp16": {
        "enabled": True
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-4,
            "weight_decay": 0.01
        }
    }
}
```

#### Flash Attention
```python
# Enable Flash Attention for memory-efficient attention computation
from transformers import LlamaConfig

config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
config.use_flash_attention_2 = True

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    config=config,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
```

### Training Optimization

#### Learning Rate Scheduling
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

# Cosine annealing with warmup
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

#### Batch Size Optimization
```python
# Dynamic batch size finder
def find_optimal_batch_size(model, tokenizer, dataset, max_batch_size=32):
    for batch_size in range(1, max_batch_size + 1):
        try:
            # Test training step with current batch size
            test_training_step(model, tokenizer, dataset, batch_size)
            optimal_batch_size = batch_size
        except torch.cuda.OutOfMemoryError:
            break
    return optimal_batch_size
```

### Quality Improvement Techniques

#### Data Quality Enhancement
```python
# Automatic data quality scoring
def score_data_quality(text):
    score = 0
    # Length check
    if 10 <= len(text.split()) <= 500:
        score += 1
    # Language detection
    if detect_language(text) == "en":
        score += 1
    # Toxicity filtering
    if toxicity_score(text) < 0.1:
        score += 1
    return score / 3

# Filter high-quality examples
filtered_dataset = dataset.filter(lambda x: score_data_quality(x["text"]) > 0.8)
```

#### Regularization Techniques
```python
# LoRA with different regularization
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,  # Dropout for regularization
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Training with weight decay
training_args = TrainingArguments(
    weight_decay=0.01,  # L2 regularization
    warmup_steps=100,
    # ...
)
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Out of Memory Errors
**Symptoms**: CUDA out of memory, RuntimeError
**Solutions**:
1. Reduce batch size and increase gradient accumulation
2. Enable gradient checkpointing
3. Use smaller model or quantization
4. Implement DeepSpeed ZeRO

```python
# Emergency OOM recovery
try:
    trainer.train()
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    # Reduce batch size by half
    training_args.per_device_train_batch_size //= 2
    training_args.gradient_accumulation_steps *= 2
```

#### Training Instability
**Symptoms**: Loss spikes, NaN values, diverging training
**Solutions**:
1. Lower learning rate
2. Increase warmup steps
3. Use gradient clipping
4. Check data quality

```python
# Gradient clipping
training_args = TrainingArguments(
    max_grad_norm=1.0,  # Clip gradients
    warmup_steps=500,   # Longer warmup
    learning_rate=1e-5, # Lower learning rate
    # ...
)
```

#### Poor Convergence
**Symptoms**: Loss plateaus, slow improvement
**Solutions**:
1. Increase model capacity (higher rank for LoRA)
2. Improve data quality
3. Adjust learning rate schedule
4. Check for data leakage

```python
# Higher capacity LoRA
lora_config = LoraConfig(
    r=64,  # Increased from 16
    lora_alpha=128,  # Scaled accordingly
    # ...
)
```

#### Model Overfitting
**Symptoms**: Training loss decreases but validation loss increases
**Solutions**:
1. Increase regularization
2. Reduce training epochs
3. Implement early stopping
4. Add more diverse training data

```python
# Early stopping callback
from transformers import EarlyStoppingCallback

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
```

### Debugging Tools

#### Memory Profiling
```python
import torch.profiler

# Profile memory usage
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    trainer.train()
```

#### Training Metrics Monitoring
```python
# Custom logging callback
class DetailedLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            # Log additional metrics
            if torch.cuda.is_available():
                logs["gpu_memory"] = torch.cuda.memory_allocated() / 1024**3
            logs["learning_rate"] = state.log_history[-1]["learning_rate"]
```

## Cost Analysis

### Computational Requirements

#### Training Cost Estimation
```python
def estimate_training_cost(
    model_size_b, 
    dataset_size, 
    method="lora",
    gpu_type="A100",
    cloud_provider="aws"
):
    # GPU specifications
    gpu_specs = {
        "A100": {"memory": 40, "hourly_cost": 4.0},
        "V100": {"memory": 16, "hourly_cost": 2.5},
        "RTX4090": {"memory": 24, "hourly_cost": 1.5}
    }
    
    # Training time estimation
    if method == "qlora":
        training_hours = (dataset_size / 1000) * 0.5
        gpu_memory_needed = model_size_b * 0.5
    elif method == "lora":
        training_hours = (dataset_size / 1000) * 1.0
        gpu_memory_needed = model_size_b * 2
    elif method == "full":
        training_hours = (dataset_size / 1000) * 4.0
        gpu_memory_needed = model_size_b * 4
    
    # Check if GPU can handle the model
    if gpu_memory_needed > gpu_specs[gpu_type]["memory"]:
        return "GPU insufficient for this configuration"
    
    total_cost = training_hours * gpu_specs[gpu_type]["hourly_cost"]
    return {
        "training_hours": training_hours,
        "total_cost": total_cost,
        "gpu_memory_needed": gpu_memory_needed
    }
```

#### Cost Comparison Table

| Model Size | Method | GPU Hours | AWS Cost | Azure Cost | GCP Cost |
|------------|--------|-----------|----------|------------|----------|
| 7B | QLoRA | 2-5 | $8-20 | $10-25 | $9-22 |
| 7B | LoRA | 5-10 | $20-40 | $25-50 | $22-45 |
| 7B | Full FT | 20-40 | $80-160 | $100-200 | $90-180 |
| 13B | QLoRA | 4-8 | $16-32 | $20-40 | $18-36 |
| 13B | LoRA | 8-16 | $32-64 | $40-80 | $36-72 |
| 70B | QLoRA | 16-32 | $128-256 | $160-320 | $144-288 |

### Budget Optimization Strategies

#### Multi-tier Approach
1. **Prototype**: Start with QLoRA on smaller dataset
2. **Validation**: Scale to LoRA with full dataset
3. **Production**: Consider full fine-tuning if performance critical

#### Resource Scheduling
```python
# Spot instance training for cost savings
def train_with_spot_instances(model, dataset, max_retries=3):
    for attempt in range(max_retries):
        try:
            trainer.train()
            break
        except SpotInstanceInterruption:
            # Save checkpoint and retry
            trainer.save_model(f"checkpoint-attempt-{attempt}")
            continue
```

## Production Deployment

### Model Serving Optimization

#### Model Quantization for Inference
```python
# Post-training quantization
from transformers import AutoModelForCausalLM
import torch

# Load fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./fine-tuned-model")

# Quantize for inference
model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)
```

#### Model Versioning and A/B Testing
```python
# Model registry integration
class ModelRegistry:
    def __init__(self):
        self.models = {}
    
    def register_model(self, name, version, model_path):
        self.models[f"{name}:{version}"] = model_path
    
    def get_model(self, name, version="latest"):
        return self.models.get(f"{name}:{version}")
    
    def a_b_test(self, model_a, model_b, traffic_split=0.5):
        # Implement A/B testing logic
        pass
```

### Monitoring and Maintenance

#### Performance Monitoring
```python
# Model performance tracking
import wandb

def log_inference_metrics(model, test_dataset):
    metrics = {
        "perplexity": calculate_perplexity(model, test_dataset),
        "bleu_score": calculate_bleu(model, test_dataset),
        "response_time": measure_response_time(model),
        "throughput": measure_throughput(model)
    }
    wandb.log(metrics)
```

#### Automated Retraining Pipeline
```python
# Continuous learning pipeline
class ContinuousLearningPipeline:
    def __init__(self, base_model, threshold=0.05):
        self.base_model = base_model
        self.performance_threshold = threshold
        
    def check_performance_drift(self, new_data):
        current_performance = evaluate_model(self.base_model, new_data)
        if current_performance < self.performance_threshold:
            self.trigger_retraining(new_data)
    
    def trigger_retraining(self, new_data):
        # Automated retraining logic
        pass
```

### Security and Compliance

#### Model Security
```python
# Input validation and sanitization
def validate_input(text, max_length=1000):
    if len(text) > max_length:
        raise ValueError("Input too long")
    
    # Check for malicious patterns
    if contains_injection_patterns(text):
        raise ValueError("Potentially malicious input")
    
    return sanitize_text(text)
```

#### Compliance Monitoring
```python
# Bias detection and mitigation
def monitor_bias(model, test_cases):
    bias_metrics = {}
    for protected_attribute in ["gender", "race", "age"]:
        bias_score = calculate_bias_score(model, test_cases, protected_attribute)
        bias_metrics[protected_attribute] = bias_score
    
    return bias_metrics
```

## Conclusion

This comprehensive guide provides the foundation for implementing production-ready LLM fine-tuning solutions. The choice of approach depends on your specific requirements:

- **QLoRA**: Best for resource-constrained environments and rapid prototyping
- **LoRA**: Optimal balance of performance and efficiency
- **Full Fine-tuning**: Maximum performance for critical applications
- **Instruction Tuning**: Improved task following and user interaction
- **RLHF**: Superior alignment with human preferences

Remember to start with the simplest approach that meets your requirements, then scale up as needed. Continuous monitoring and evaluation are crucial for maintaining model performance in production environments.

