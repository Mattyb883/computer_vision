# Age Estimation from Face Images

A supervised regression project to predict a person’s age from a webcam photo, built using a ResNet50 backbone and trained on the ChaLearn “Looking at People” dataset. This repository contains all code, data loaders, and pre-trained results needed to reproduce the analysis and model evaluation.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Environment & Requirements](#environment--requirements)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Jupyter Notebook](#jupyter-notebook)  
  - [Standalone Script (GPU)](#standalone-script-gpu)  
- [Results](#results)  
- [Evaluation & Conclusions](#evaluation--conclusions)  
- [Future Work](#future-work)  
- [Author](#author)  

---

## Project Overview

Good Seed supermarket chain wants to prevent under-age alcohol sales by automatically estimating a customer’s age at checkout. In this project you will:

1. Perform **Exploratory Data Analysis (EDA)** to understand age distributions and image data quality.  
2. Build and train a **convolutional neural network** (ResNet50 backbone + custom regression head) to predict age.  
3. Evaluate model performance using **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)**.  
4. Package functions into reusable scripts and Jupyter notebooks.  

---

## Dataset

- **Source:** ChaLearn “Looking at People” challenge, curated into `/datasets/faces/`.  
- **Contents:**  
  - `labels.csv` — two columns: `file_name`, `real_age`  
  - `final_files/` — ~7,600 JPEG face images  
- **Split:** 75% training (5,694 images) / 25% validation (1,897 images) via `ImageDataGenerator(validation_split=0.25)`.

---

## Environment & Requirements

- **Python 3.8+**  
- **TensorFlow 2.x**  
- **pandas, numpy, matplotlib, PIL**  
- **GPU recommended** (for training and fine-tuning)

Install dependencies via:

```bash
pip install -r requirements.txt
