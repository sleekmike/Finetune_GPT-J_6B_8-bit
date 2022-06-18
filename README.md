# Finetune_GPT-J_6B_8-bit

## Overview
This repository contains code to Fine-tuning GPT-J-6B(Quantized EleutherAI/gpt-j-6b with 8-bit weights) on colab or equivalent PC/Server GPU with your custom datasets. 

It contains code originally from Hivemind's Proof-of-concept notebook for fine-tuning on [colab](https://colab.research.google.com/drive/1ft6wQU0BhqG5PRlwgaZJv2VukKKjU4Es)

The model was modified and developed by [Hivemind](https://huggingface.co/hivemind/gpt-j-6B-8bit)

It is complimentary to my [medium post](https://michaelohanu.medium.com/fine-tuning-gpt-j-6b-on-google-colab-or-equivalent-desktop-or-server-gpu-b6dc849cb205)

For a tutorial on fine-tuning the original or vanilla GPT-J 6B, check out [Eleuther’s guide](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/howto_finetune.md)

## Usage 

Create a `virtualenv` and install the requirements:
```
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Then, place your training datasets (train and test) in the same directory as the script.

Run the fine-tuning code to start fine-tuning the model:
```python3 gpt-j-6b-8-bit.py```

To start the API for inferencing, run the command below:
```uvicorn api:app --reload```