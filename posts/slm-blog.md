---
title: "The Rise of Small Language Models: A Comprehensive Survey"
subtitle: ""
date: "01-10-24"
---

## Introduction

In recent years, Large Language Models (LLMs) like GPT-3 and GPT-4 have dominated headlines in the AI world. However, a quieter revolution has been taking place in the realm of Small Language Models (SLMs). These more compact models, typically ranging from 100 million to 5 billion parameters, are designed for deployment on edge devices like smartphones, tablets, and IoT gadgets.

A recent survey paper by Lu et al. titled [Small Language Models: Survey, Measurements, and Insights](https://arxiv.org/pdf/2409.15790) provides a comprehensive overview of the state of SLM research and development. In this blog post, we'll dive deep into their findings, exploring the architecture, capabilities, and future directions of SLMs.

## Why Small Language Models Matter

Before we delve into the technical details, it's important to understand why SLMs are gaining traction:

1. **Accessibility**: SLMs can run on consumer devices, bringing AI capabilities to a wider audience.
2. **Privacy**: On-device processing reduces the need to send sensitive data to cloud servers.
3. **Latency**: Local inference can be faster than cloud-based alternatives, especially in areas with poor connectivity.
4. **Cost-effectiveness**: Reduced computational requirements make SLMs more economical to deploy at scale.

## SLM Architecture and Innovations

### Core Architecture

Most SLMs are based on the transformer architecture, specifically the decoder-only variant. However, researchers have introduced several innovations to optimize these models for smaller scales:

1. **Attention Mechanisms**: While Multi-Head Attention (MHA) remains common, newer models are adopting Group-Query Attention (GQA) and Multi-Query Attention (MQA) for improved efficiency.

2. **Feed-Forward Networks (FFN)**: There's a trend towards using Gated FFNs instead of standard FFNs.

3. **Activation Functions**: SiLU (Sigmoid Linear Unit) is becoming the dominant activation function, replacing ReLU and GELU variants.

4. **Normalization**: RMSNorm is gradually replacing LayerNorm for better performance.

5. **Vocabulary Size**: Recent models tend to have larger vocabularies, often exceeding 50,000 tokens.

Here's a code snippet illustrating a simplified SLM architecture with some of these innovations:

```python
import torch
import torch.nn as nn

class SLMBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = GroupQueryAttention(d_model, n_heads)
        self.ff = GatedFFN(d_model, d_ff)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class GatedFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w3(self.act(self.w1(x)) * self.w2(x))
```

### Architectural Innovations

The survey highlights several architectural innovations specific to SLMs:

1. **Parameter Sharing**: Some models, like MobiLLaMA, share weights across transformer layers to reduce model size.

2. **Layer-wise Parameter Scaling**: Models like OpenELM use different configurations for each transformer layer, optimizing parameter allocation.

3. **Nonlinearity Compensation**: Techniques like series activation functions and augmented shortcuts are used to address feature collapse issues.

## Training Datasets and Strategies

The choice of training data significantly impacts SLM performance. The survey identified several key datasets used for SLM training:

1. The Pile (825B tokens)
2. RefinedWeb (5T tokens)
3. RedPajama (1.2T tokens)
4. DCLM-baseline (1.35T tokens)
5. CulturaX (6.3T tokens)

Interestingly, the researchers found that SLMs are often "over-trained" compared to the Chinchilla optimal scaling law. For example, some 1B parameter models are trained on over 1.5T tokens, far exceeding the 20B tokens suggested by the Chinchilla law.

This graph illustrates how recent SLMs are often trained on much larger datasets than what the Chinchilla law suggests, potentially to compensate for their smaller size when deployed on resource-constrained devices.

## SLM Capabilities

The survey evaluated SLM performance across three main task categories:

1. Commonsense Reasoning
2. Problem-Solving
3. Mathematics

Key findings include:

- SLMs have shown significant improvements from 2022 to 2024, with performance increases of 10.4%, 13.5%, and 13.5% for the three task categories, respectively.
- Some SLMs, particularly the Phi family from Microsoft, achieve performance comparable to larger models like LLaMA 3.1 (7B parameters) on certain tasks.
- SLMs trained on open-source datasets are closing the gap with those trained on closed datasets, especially for commonsense tasks.

Here's a simplified Python script to evaluate an SLM on a commonsense reasoning task:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_commonsense(model, tokenizer, question, choices):
    prompt = f"Question: {question}\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer: "

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=1, num_return_sequences=1)
    
    predicted_answer = tokenizer.decode(output[0][-1])
    return predicted_answer

# Load model and tokenizer
model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example question
question = "What happens to water when it freezes?"
choices = ["It expands", "It shrinks", "It stays the same size", "It evaporates"]

answer = evaluate_commonsense(model, tokenizer, question, choices)
print(f"Predicted answer: {answer}")
```

## Runtime Performance

The survey also analyzed the runtime performance of SLMs on edge devices. Key findings include:

1. **Latency**: Inference latency varies significantly based on model architecture, not just parameter count. For example, Qwen1.5-0.5B (500M parameters) runs 31.9% faster than Qwen2-0.5B on a Jetson Orin NX 16GB, despite having 25.4% more parameters.

2. **Memory Usage**: Runtime memory usage generally correlates linearly with parameter count, but vocabulary size can have a significant impact. Models with larger vocabularies (e.g., Bloom series with 250,880 tokens) use more memory than those with smaller vocabularies.

3. **Quantization**: 4-bit quantization (Q4_K_M) provides the best balance of performance and accuracy, reducing latency by up to 50% in some cases.

Here's a code snippet demonstrating how to quantize an SLM using the GPTQ method:

```python
from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name = "microsoft/phi-2"
quantized_model_dir = "./quantized_phi2"

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define quantization config
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

# Quantize the model
quantized_model = AutoGPTQForCausalLM.from_pretrained(model, quantize_config)

# Save the quantized model
quantized_model.save_pretrained(quantized_model_dir)
```

## Future Directions and Challenges

The survey identifies several promising research directions for SLMs:

1. **Co-design of SLM architecture and device processors**: Optimizing SLM architectures for specific hardware to achieve better accuracy-speed tradeoffs.

2. **High-quality synthetic datasets**: Developing better techniques for curating and filtering training data to improve SLM performance.

3. **Deployment-aware scaling laws**: Refining the Chinchilla law to account for the unique constraints of edge deployment.

4. **On-device continual learning**: Developing efficient techniques for personalizing SLMs using on-device data without compromising privacy.

5. **Device-cloud collaboration**: Exploring hybrid approaches that leverage both on-device SLMs and cloud-based LLMs for optimal performance and privacy.

## Conclusion

Small Language Models represent a crucial frontier in AI research, promising to bring powerful language capabilities to edge devices while preserving privacy and reducing latency. As the survey by Lu et al. demonstrates, SLMs have made significant strides in recent years, narrowing the gap with larger models in many tasks.

However, challenges remain in optimizing these models for the unique constraints of edge deployment. As research continues in areas like efficient architectures, high-quality datasets, and on-device learning, we can expect SLMs to play an increasingly important role in bringing AI to our everyday devices.

The future of AI may not just be about building bigger models, but about making small models smarter and more efficient. As this survey shows, the small language model revolution is well underway, and its impact on how we interact with AI in our daily lives could be profound.