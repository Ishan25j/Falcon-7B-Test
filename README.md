---
datasets:
- OpenAssistant/oasst1
pipeline_tag: text-generation
license: tii-falcon-llm
language:
- en
---

# 🚀 Falcon-7b-chat-oasst1

Falcon-7b-chat-oasst1 is a chatbot-like model for dialogue generation. It was built by fine-tuning [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) on the [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) dataset. 

## Model Summary

- **Model Type:** Causal decoder-only
- **Language(s):** English
- **Base Model:** [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) (License: [TII Falcon LLM License](https://huggingface.co/tiiuae/falcon-7b#license))
- **Dataset:** [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) (License: [Apache 2.0](https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/LICENSE))
- **License(s):** Inherited from "Base Model" and "Dataset"

## Model Details

The model was fine-tuned in 8-bit precision using 🤗 `peft` adapters, `transformers`, and `bitsandbytes`. Training relied on a method called "Low Rank Adapters" ([LoRA](https://arxiv.org/pdf/2106.09685.pdf)), specifically the [QLoRA](https://arxiv.org/abs/2305.14314) variant. The run took approximately 3 hours and was executed on a workstation with a single A100-SXM NVIDIA GPU with 37 GB of available memory. See attached [Colab Notebook](https://huggingface.co/dfurman/falcon-7b-chat-oasst1/blob/main/finetune_falcon7b_oasst1_with_bnb_peft.ipynb) for the code and hyperparams used to train the model. 

### Model Date

May 30, 2023

## Quick Start

To prompt the chat model, use the following format:

```
<human>: [Instruction]
<bot>:
```

### Example Dialogue 1

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

### Example Dialogue 2

**Prompter**:
```
<human>: Create a list of four things to do in San Francisco.
<bot>:
```

**Falcon-7b-chat-oasst1**:
```
Here are four things to do in San Francisco:

1. Visit the Golden Gate Bridge: The Golden Gate Bridge is one of the most iconic landmarks in the world and is a must-see for any visitor to San Francisco. The bridge offers stunning views of the city and the Pacific Ocean, and there are plenty of opportunities for photos and sightseeing along the way.

2. Explore Chinatown: San Francisco's Chinatown is one of the oldest and most vibrant Chinatowns in the world, and it's a great place to explore the city's Chinese culture and history. You can wander the streets, sample the local cuisine, and visit the many shops and markets.

3. Visit Alcatraz Island: Alcatraz Island is a former prison and military fort located in the middle of San Francisco Bay. The island is now a popular tourist attraction and offers guided tours that take visitors through the prison cells and other historic buildings.

4. Take a cable car ride: San Francisco's cable cars are a classic tourist attraction and offer a unique way to explore the city. The cars run along several routes and offer stunning views of the city and the surrounding area.

These are just a few of the many things to do in San Francisco. There are plenty of other activities, sights, and attractions to explore, so be sure to do your research and plan your trip accordingly.
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
# Install packages
!pip install -q -U bitsandbytes loralib einops
!pip install -q -U git+https://github.com/huggingface/transformers.git 
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
```

### GPU Inference in 8-bit

This requires a GPU with at least 12 GB of memory.

```python
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# load the model
peft_model_id = "dfurman/falcon-7b-chat-oasst1"
config = PeftConfig.from_pretrained(peft_model_id)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    device_map={"":0},
    trust_remote_code=True,
    load_in_8bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, peft_model_id)

# run the model
prompt = """<human>: My name is Daniel. Write a short email to my closest friends inviting them to come to my home on Friday for a dinner party, I will make the food but tell them to BYOB.
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

generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
# Inspect message response in the outputs
print(generated_text.split("<human>: ")[1].split("<bot>: ")[-1])
```

## Reproducibility

See attached [Colab Notebook](https://huggingface.co/dfurman/falcon-7b-chat-oasst1/blob/main/finetune_falcon7b_oasst1_with_bnb_peft.ipynb) for the code (and hyperparams) used to train the model. 

### CUDA Info

- CUDA Version: 12.0
- Hardware: 1 A100-SXM
- Max Memory: {0: "37GB"}
- Device Map: {"": 0}

### Package Versions Employed

- `torch`: 2.0.1+cu118
- `transformers`: 4.30.0.dev0
- `peft`: 0.4.0.dev0
- `accelerate`: 0.19.0
- `bitsandbytes`: 0.39.0
- `einops`: 0.6.1