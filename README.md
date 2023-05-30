---
datasets:
- OpenAssistant/oasst1
pipeline_tag: text-generation
---

# Falcon-7b-chat-oasst1

Falcon-7b-chat-oasst1 is a chatbot-like model for dialogue generation. It was built by fine-tuning [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) on the [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) dataset. 
This model was fine-tuned in 8-bit using 🤗 [peft](https://github.com/huggingface/peft) adapters, [transformers](https://github.com/huggingface/transformers), and [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).
- The training relied on a recent method called "Low Rank Adapters" ([LoRA](https://arxiv.org/pdf/2106.09685.pdf)), instead of fine-tuning the entire model you just have to fine-tune adapters and load them properly inside the model. 
- Training took approximately 6 hours and was executed on a workstation with a single NVIDIA A100-SXM 40GB GPU (via Google Colab).
- See attached [Notebook](https://huggingface.co/dfurman/falcon-7b-chat-oasst1/blob/main/finetune_falcon7b_oasst1_with_bnb_peft.ipynb) for the code (and hyperparams) used to train the model. 

## Model Summary

- **Model Type:** Causal decoder-only
- **Language(s) (NLP):** English (primarily)
- **Base Model:** [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) (License: [TII Falcon LLM License](https://huggingface.co/tiiuae/falcon-7b#license), commercial use ok-ed)
- **Dataset:** [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) (License: [Apache 2.0](https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/LICENSE), commercial use ok-ed)
- **License:** Inherited from "Base Model" and "Dataset"

### Model Date

May 30, 2023

## Quick Start

To prompt the chat model, use the following format:

```
<human>: [Instruction]
<bot>:
```

### Example Dialogue

**Prompter**:
```
"""<human>: My name is Daniel. Write a short email to my closest friends inviting them to come to my home on Friday for a dinner party, I will make the food but tell them to BYOB.
<bot>:"""
```

**Falcon-7b-chat-oasst1**:
```
Dear friends,

I am so excited to host a dinner party at my home this Friday! I will be making a delicious meal, but I would love for you to bring your favorite bottle of wine to share with everyone.

Please let me know if you can make it and if you have any dietary restrictions I should be aware of. I look forward to seeing you soon!

Best,
Daniel
```

**Prompter**:
```
<human>: Create a list of five things to do in San Francisco.\n
<bot>:
```

**Falcon-7b-chat-oasst1**:
```
Here are four things to do in San Francisco:

1. Visit the Golden Gate Bridge: The Golden Gate Bridge is one of the most iconic landmarks in the world and is a must-see for any visitor to San Francisco. The bridge spans 1.7 miles and offers stunning views of the city and the Pacific Ocean.

2. Explore Chinatown: San Francisco's Chinatown is one of the largest Chinatowns in the world and is a great place to experience the culture and history of the Chinese community in the city. The area is full of shops, restaurants, and cultural attractions.

3. Visit Alcatraz Island: Alcatraz Island is a former prison and now a national park. The island is home to a variety of wildlife and offers stunning views of the San Francisco Bay.

4. Take a cable car ride: San Francisco's cable cars are a classic way to get around the city and offer a unique experience. The cars run on a cable system that was first installed in 1873 and is still in use today.

These are just a few of the many things to do in San Francisco. For more ideas, check out the official tourism website for the city.
```


### Direct Use

This model has been finetuned on conversation trees from [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) and should only be used on data of a similar nature.

### Out-of-Scope Use

Production use without adequate assessment of risks and mitigation; any use cases which may be considered irresponsible or harmful. 

## Bias, Risks, and Limitations

This model is mostly trained on English data, and will not generalize appropriately to other languages. Furthermore, as it is trained on a large-scale corpora representative of the web, it will carry the stereotypes and biases commonly encountered online.

### Recommendations

We recommend users of this model to develop guardrails and to take appropriate precautions for any production use.

## How to Get Started with the Model

### Setup
```python
# Install and import packages
!pip install -q -U bitsandbytes loralib einops
!pip install -q -U git+https://github.com/huggingface/transformers.git 
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
```

### GPU Inference in 8-bit

This requires a GPU with at least 12GB memory.

```python
# load the model
peft_model_id = "dfurman/falcon-7b-chat-oasst1"
config = PeftConfig.from_pretrained(peft_model_id)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, 
    return_dict=True, 
    load_in_8bit=True, 
    device_map="auto",
    use_auth_token=True,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, peft_model_id)
```

```python
# run the model
prompt = """<human>: My name is Daniel. Write a long email to my closest friends inviting them to come to my home on Friday for a dinner party, I will make the food but tell them to BYOB.
<bot>:"""

batch = tokenizer(
    prompt,
    padding=True,
    truncation=True,
    return_tensors='pt'
)
batch = batch.to('cuda:0')

with torch.cuda.amp.autocast():
    output_tokens = model.generate(
        input_ids = batch.input_ids, 
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.7,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# Inspect outputs
print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
```

## Reproducibility

- See attached [Notebook](https://huggingface.co/dfurman/falcon-7b-chat-oasst1/blob/main/finetune_falcon7b_oasst1_with_bnb_peft.ipynb) for the code (and hyperparams) used to train the model. 

### CUDA Info

- CUDA Version: 12.0
- GPU Name: NVIDIA A100-SXM
- Max Memory: {0: "37GB"}
- Device Map: {"": 0}

### Package Versions Employed

- `torch`: 2.0.1+cu118
- `transformers`: 4.30.0.dev0
- `peft`: 0.4.0.dev0
- `accelerate`: 0.19.0
- `bitsandbytes`: 0.39.0
- `einops`: 0.6.1
