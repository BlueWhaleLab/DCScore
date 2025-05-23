# DCScore
**Paper:** Measuring Diversity in Synthetic Datasets (ICML 2025)

## Overview

To evaluate the diversity of synthetic datasets, we propose **DCScore**, a simple yet effective diversity measurement method. DCScore treats the diversity evaluation as a sample classification task, considering mutual relationships among samples. DCScore aims to tackle the LLM-generated dataset diversity evaluation problem. The framework of DCScore is shown as follows.

<img src="./framework.png" style="zoom: 50%;" />

## Requirements
- transformers
- sentence_transformers
- torch
- numpy
- sklearn

## Examples
To evaluate the diversity of text datasets:
```python
import torch
from datasets import load_dataset
from dcscore_function import DCScore

# parameter settings
model_path_dict = {
                    'bert':'path/to/bert-base-uncased',
                    'simcse':'path/to/unsup-simcse-bert-base-uncased',
                    'llama2-7b':'path/to/llama-2-7b-hf',
                    'gpt2':'path/to/gpt2',
                    'bge': 'path/to/bge-large-en-v1.5',
                    'sen_bert': 'path/to/all-mpnet-base-v2'
                  }
model_name = 'sen_bert'
model_path = model_path_dict[model_name]
device = "cuda" if torch.cuda.is_available() else "CPU"
batch_size = 128
tau = 1
kernel_type = 'cs'

# evaluated dataset
text_list = ['who are you', 'I am fine', 'good job']

# dcscore class
dcscore_evaluator = DCScore(model_path)

# get embedding
embeddings, n, d = dcscore_evaluator.get_embedding(text_list, batch_size=batch_size)

# calculate dcscore based on embedding
dataset_dcscore = dcscore_evaluator.calculate_dcscore_by_embedding(embeddings, kernel_type=kernel_type, tau=tau)
```

To evaluate the diversity of image datasets:
```python
import torch
from datasets import load_dataset
from dcscore_function import DCScore
from ImageFilesDataset import ImageFilesDataset

# parameter settings
model_name = 'dinov2'
device = "cuda" if torch.cuda.is_available() else "CPU"
batch_size = 128
tau = 1
kernel_type = 'cs'
sampel_num = 4

# evaluated dataset
img_pth = './demo_images'
dataset = ImageFilesDataset(img_pth, name='None', extension='JPEG', n=sampel_num, conditional=False)

# dcscore class
dcscore_evaluator = DCScore(model_name)

# get embedding
embeddings, n, d = dcscore_evaluator.get_embedding(dataset, batch_size=batch_size)

# calculate dcscore based on embedding
dataset_dcscore = dcscore_evaluator.calculate_dcscore_by_embedding(embeddings, kernel_type=kernel_type, tau=tau)
```

## Citation Information
Paper: <https://arxiv.org/abs/2502.08512>
```
@misc{zhu2025measuringdiversitysyntheticdatasets,
      title={Measuring Diversity in Synthetic Datasets}, 
      author={Yuchang Zhu and Huizhe Zhang and Bingzhe Wu and Jintang Li and Zibin Zheng and Peilin Zhao and Liang Chen and Yatao Bian},
      year={2025},
      eprint={2502.08512},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.08512}, 
}
```