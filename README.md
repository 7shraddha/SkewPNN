# Skew Probabilistic Neural Network for handling imbalance data classification
 
### Introduction
The Probabilistic Neural Network (PNN) traditionally uses Gaussian kernel density estimation. However, this may not capture real-world complexities. This README file provides an overview of our approach, its implementation in MATLAB, and instructions for running the code.

### Methodology
In light of this, we delve into the skew normal distribution as a dynamic alternative, particularly beneficial for imbalanced datasets where traditional methods fall short. This shift aims to bolster performance and provide a more accurate reflection of underlying class distributions. Notably, hyperparameter fine-tuning becomes pivotal, and here, bat optimization steps in to address this challenge. This strategic refinement substantially equips the PNN to navigate uncertainty and tackle class imbalances. The outcome? Markedly improved classification accuracy and robust generalization even when faced with previously unseen data.

### Datasets
The repository also includes both imbalanced and balanced datasets that were utilized for experimentation. 

### Code Organization
The MATLAB code for our proposed approaches is organized as follows:

1. main_prog.m: This is the main script that demonstrates the usage of our approach. It includes the necessary steps to prepare the data, apply SkewPNN and BA-SkewPNN models.

2. classification.m: This script contains the implementation of the SkewPNN and BA-SkewPNN models. It takes the input data and applies the SkewPNN to perform classification task. 

3. Other necessory files are also included. 

### Citation
If you find our proposed approach or the provided code useful in your research work, please consider citing our paper:

Will be added soon
