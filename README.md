# A Reimplementation of Integrating ChatGPT into Secure Hospital Networks: A Case Study on Improving Radiology Report Analysis - A re-implementation

## Project Presentation

[**Click here to watch the CS598 Final Project Presentation**](https://mediaspace.illinois.edu/media/t/1_h32zotxz)

## Trained models, metrics, and data:

Labels are not shared, as MIMIC-CXR is restricted via PhysioNet, but access can be obtained after completing the necessary CITI program trainings and requesting.  You can find the dataset and the steps to gain access on PhysioNet [here](https://physionet.org/content/mimic-cxr/2.1.0/).

We provide all of our trained models and baselines at the following drive link for cross referencing your results against ours:

[Click here to access the models](https://drive.google.com/drive/folders/175MWvitD2-kimn7uNONE-McWBtpQqj7E?usp=sharing)

## Code Overview:
`databricks_assets` this folder contains all the jupyter notebooks and python files necessary for the data preprocessing steps and creating the "ground truth" labels from the teacher models.  You can use a free edition Databricks Workspace to run this code without any additional cost.

`train_local_models.py` is the file for training, testing, and saving the metrics for the models after you've obtained the labels.

`visualizations.ipynb` this jupyter notebook is used to create the embedding space visualizations seen in our paper.

## Results:

**Performance comparison between baseline and contrastive loss (New Data).**
*Bold indicates the best performance between the two methods for a given backbone.*

| Model | Accuracy | Specificity | Sensitivity | F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Bio_ClinicalBERT** | | | | |
| ... Document Baseline | **95.62** | 89.34 | **98.25** | **96.93** |
| ... Document Contrastive | 95.57 (-0.05) | **92.97** (+3.63) | 96.67 (-1.58) | 96.85 (-0.08) |
| **DeBERTa-v3-base** | | | | |
| ... Document Baseline | **96.15** | **93.88** | 97.11 | **97.26** |
| ... Document Contrastive | 96.11 (-0.04) | 92.59 (-1.29) | **97.59** (+0.48) | 97.25 (-0.01) |
| **RadBERT-4m** | | | | |
| ... Document Baseline | **96.38** | **93.95** | 97.40 | **97.43** |
| ... Document Contrastive | 96.29 (-0.09) | 93.50 (-0.45) | **97.46** (+0.06) | 97.37 (-0.06) |

## 📖 Citation

Original paper:

```bibtex
@InProceedings{kim2024integrating,
  title     = {Integrating ChatGPT into Secure Hospital Networks: A Case Study on Improving Radiology Report Analysis},
  author    = {Kyungsu Kim and Junhyun Park and Saul Langarica and Adham Mahmoud Alkhadrawi and Synho Do},
  booktitle = {Conference on Health, Inference, and Learning (CHIL)},
  publisher = {Proceedings of Machine Learning Research (PMLR)},
  volume    = {248},
  pages     = {72--87},
  year      = {2024}
}
