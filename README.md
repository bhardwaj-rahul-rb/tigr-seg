# TIGR-Seg: Text-Initialized Gaussian Refinement with Spatially Aware Visual Adaptation for Medical Image Segmentation

This is an official PyTorch implementation of "TIGR-Seg: Text-Initialized Gaussian Refinement with Spatially Aware Visual Adaptation for Medical Image Segmentation"

## News

<!-- - **[Sep. 2026]** 🎉 Our paper **TIGR-Seg** has been accepted to **ICLR 2027**! -->
- **[Sep. 2026]** 🚀 The official TIGR-Seg repository is released.
-----

>  Medical image segmentation remains challenging due to the limited availability of pixel-level annotations, heterogeneous target appearance, and variations across imaging modalities. Pretrained vision encoders provide strong representations, but updating all backbone parameters introduces substantial trainable capacity, while textual information often provides semantic and coarse spatial guidance rather than precise pixel-level localization. This work presents TIGR-Seg, a parameter-efficient text-guided segmentation framework comprising Gaussian-Gated Feature Adaptation (GGFA) and a Text-Initialized Gaussian Refinement (TIGR) Decoder. GGFA adapts a frozen pretrained vision encoder through context-conditioned feature modulation and a dynamically predicted correlated Gaussian spatial gate, enabling adaptation across both feature and spatial dimensions while updating only a small set of parameters. The TIGR Decoder incorporates stage-specific textual guidance across multiple decoding resolutions. At each stage, text initializes a Gaussian spatial hypothesis that is subsequently refined using visual evidence through reliability-weighted corrections. The refined spatial priors then guide feature aggregation, reconstruction, and progressive decoder refinement. Experiments across gastrointestinal endoscopy (BKAI), brain MRI (BTMRI), and breast ultrasound (BUSI) datasets demonstrate strong segmentation performance with a low trainable-parameter count. Additional evaluations support the effectiveness and generalizability of TIGR-Seg.
-----

## Framework
<!-- ![Framework]() -->
<p align="center">
  <img src="" width="95%">
</p>

<p align="center">
  <em>Architecture of TIGR-Seg.</em>
</p>


## Installation
Clone the repository:
```bash
git clone <repository-url>
cd TIGR-Seg
```

The main dependencies are as follows:  
```text
einops
linformer
monai
pandas 
pytorch_lightning
pyyaml
scipy
timm
torch
torchmetrics 
transformers 
thop
```

or use the following:
```bash
pip install requirements.txt
```

## Requirements
### Dataset
1. The images and corresponding segmentation masks used in this work are available from the following sources: [**BKAI**](https://www.kaggle.com/competitions/bkai-igh-neopolyp/data), [**BTMRI**](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427), and [**BUSI**](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset).

2. The text annotations for the **BKAI** and **BUSI** datasets are obtained from [MedVLSM](https://github.com/naamiinepal/medvlsm), while those for **BTMRI** are obtained from [MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg).

   **Thanks to Poudel *et al.*  and Koleilat *et al.* for their contributions. If you use this text annotations, please cite their work**.

### Pre-trained Model Weights
We have used [SwinV2-Tiny-Patch4-Window8-256](https://huggingface.co/microsoft/swinv2-tiny-patch4-window8-256) (vision) and [ClinicalBERT](https://huggingface.co/medicalai/ClinicalBERT) (text) in this experiment.

The models can be used as follows:
   ```python
   url = "microsoft/swinv2-tiny-patch4-window8-256"
   tokenizer = AutoTokenizer.from_pretrained(url,trust_remote_code=True)
   model = AutoModel.from_pretrained(url, trust_remote_code=True)
   ```

## Repository Structure
```text
TIGR-Seg/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── test/
│   │   ├── images/
│   │   └── masks/
│   ├── train_annotations.csv
│   └── test_annotations.csv
│
├── models/
├── utils/
├── train.py
├── test.py
├── requirements.txt
└── README.md
```

## Code Execution
### Training
To **train** the model, execute: ``` python train.py ```

### Evaluation
To **test** the model, execute: ``` python test.py ``` after *training* the model or using the learned weights given [*below*](#model-weights).

## Model Weights
The learned model weights are available below:
| Dataset | Model | Download link |
| ----------- | ------- | ---------------- |
| BKAI | TIGR-Seg | <div align="center"><a href="">Google Drive</a></div> |
| BTMRI | TIGR-Seg | <div align="center"><a href="">Google Drive</a></div> |
| BUSI | TIGR-Seg | <div align="center"><a href="">Google Drive</a></div> |

## Todo List
- [ ] Release entire code
- [X] Release model weights

## Acknowledgement
<!-- The work is inspired from [LViT](https://github.com/HUANGLIZI/LViT) and [Ariadne’s Thread](https://github.com/Junelin2333/LanGuideMedSeg-MICCAI2023). Thanks for the open source contributions! -->

## Citation
If you find this work useful, please cite our paper:
```bibtex

```
