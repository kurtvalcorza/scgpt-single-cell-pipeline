---
license: mit
tags:
- single-cell
- biology
base_model:
- apliko/scGPT
---
# scGPT
scGPT is A foundation model for single-cell biology based on a generative pre trained transformer across a repository of over 33 million cells.

# Abstract
Generative pretrained models have achieved remarkable success in various domains such as language and computer vision. Specifically, the combination of large-scale diverse datasets and pretrained transformers has emerged as a promising approach for developing foundation models. Drawing parallels between language and cellular biology (in which texts comprise words; similarly, cells are defined by genes), our study probes the applicability of foundation models to advance cellular biology and genetic research. Using burgeoning single-cell sequencing data, we have constructed a foundation model for single-cell biology, scGPT, based on a generative pretrained transformer across a repository of over 33 million cells. Our findings illustrate that scGPT effectively distills critical biological insights concerning genes and cells. Through further adaptation of transfer learning, scGPT can be optimized to achieve superior performance across diverse downstream applications. This includes tasks such as cell type annotation, multi-batch integration, multi-omic integration, perturbation response prediction and gene network inference.

# Code

```python
from tdc.multi_pred.anndata_dataset import DataLoader
from tdc import tdc_hf_interface
from tdc.model_server.tokenizers.scgpt import scGPTTokenizer
import torch

# an example dataset
adata = DataLoader("cellxgene_sample_small",
                   "./data",
                   dataset_names=["cellxgene_sample_small"],
                   no_convert=True).adata

# code for loading the model and performing inference
scgpt = tdc_hf_interface("scGPT")
model = scgpt.load()  # This line can cause segmentation fault on inappropriate setup
tokenizer = scGPTTokenizer()
gene_ids = adata.var["feature_name"].to_numpy(
)  # Convert to numpy array
tokenized_data = tokenizer.tokenize_cell_vectors(
    adata.X.toarray(), gene_ids)
mask = torch.tensor([x != 0 for x in tokenized_data[0][1]],
                    dtype=torch.bool)

# Extract first embedding
first_embed = model(tokenized_data[0][0],
                    tokenized_data[0][1],
                    attention_mask=mask)
```

# TDC.scGPT Source Code
https://github.com/mims-harvard/TDC/blob/main/tdc/model_server/models/scgpt.py
* hf migration code available upon request
* weights extracted from base model

# TDC Citation
```
@inproceedings{
velez-arce2024signals,
title={Signals in the Cells: Multimodal and Contextualized Machine Learning Foundations for Therapeutics},
author={Alejandro Velez-Arce and Xiang Lin and Kexin Huang and Michelle M Li and Wenhao Gao and Bradley Pentelute and Tianfan Fu and Manolis Kellis and Marinka Zitnik},
booktitle={NeurIPS 2024 Workshop on AI for New Drug Modalities},
year={2024},
url={https://openreview.net/forum?id=kL8dlYp6IM}
}
```
# Additional Citations
- Cui, H., Wang, C., Maan, H. et al. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. Nat Methods 21, 1470–1480 (2024). https://doi.org/10.1038/s41592-024-02201-0

# Model Github
https://github.com/bowang-lab/scGPT