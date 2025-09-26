# LLM-Based Tabular Data Classification

## Table of Contents

* [Overview](#overview)
* [Introduction](#introduction)
  * [Benefits](#benefits)
  * [Drawbacks](#drawbacks)
* [Project Structure](#project-structure)


## Overview

This project explores how large language models (LLMs) can be applied to tabular data classification by first converting each row into a natural language description. The LLM then predicts the class label and generates a plain-language explanation of its decision. This approach is compared with traditional machine learning models, focusing on trade-offs in accuracy and interpretability, with plans to evaluate fairness in future work.

Specifically, this project investigates whether modern LLMs can:
* Perform classification from natural language descriptions of rows in tabular data
* Provide understandable, human-like rationales for their decisions
* Generalize without the need for labeled training data
* Compete with classical models in accuracy and robustness

## Introduction

Tabular data is one of the most common data formats in real-world applications, ranging from healthcare records to financial transactions. Traditional machine learning models such as decision trees, logistic regression, and random forests are commonly used for classification tasks on tabular datasets due to their efficiency and strong performance.

However, these models often struggle to provide clear, human-understandable explanations for their predictions, which can limit trust and adoption in sensitive domains. Meanwhile, large language models (LLMs) have shown remarkable ability to understand and generate natural language, including reasoning and explanation tasks, but their application to structured tabular data is still emerging.

This project investigates a novel approach where tabular data rows are first converted into natural language descriptions that LLMs can process directly. By doing so, the LLM predicts the class label and generates a textual explanation of its decision, potentially improving interpretability and user trust. This approach is compared with traditional machine learning baselines, focusing on performance and interpretability, with future work planned to evaluate fairness and robustness.

### Benefits

Some key benefits of this approach include:

* **No Need for Labeled Training Data:** Because the LLM uses natural language prompts and its pre-trained knowledge, it can classify data even when labeled examples are unavailable or expensive to obtain.

* **Leverages Vast Domain Knowledge:** The LLM draws on knowledge embedded from massive text corpora (e.g., scientific articles like those in PubMed), capturing complex relationships beyond what may be found in the tabular data or what traditional models can represent.

* **Improved Interpretability:** By generating natural language explanations, the model makes its decisions more transparent and easier for humans to understand.

* **Robustness to Missing Data:** The approach naturally handles missing values by simply omitting those features from the prompt without requiring imputation or special handling.

* **Adaptable to New Datasets:** New features or changes in dataset schema can be incorporated by updating the prompt format, eliminating the need to retrain or redesign models extensively.


### Drawbacks

Some potential limitations of this approach include:

* **Inconsistent Outputs:** LLMs may produce different predictions or explanations for the same input prompt due to their generative nature. This variability can undermine reliability in classification tasks where consistent results are critical.

* **Potentially Lower Performance:** In many cases, this approach might underperform traditional machine learning methods that are explicitly trained on labeled tabular data. It may not always match the accuracy of specialized models.

* **Prompt Sensitivity:** The model's performance is highly dependent on the quality of the prompt. Designing effective prompts requires expertise and experimentation, making prompt engineering a non-trivial challenge.

* **Dependence on Pretrained Knowledge:** While pretrained knowledge can be beneficial, LLMs may reproduce social biases present in their pretraining data or introduce outdated information, even when such features are not part of the input.

* **Computational Expense:** Large language models require significant computational resources and can be slower compared to classical models, limiting scalability.

## Project Structure

The repository is organized into directories and files to support data preparation, modeling, evaluation, and source code. The following is an overview of the main components:


```
.
├── .env
├── LICENSE
├── README.md
├── data
│   └── GiveMeSomeCredit
│       ├── GiveMeSomeCredit.zip
│       ├── figures
│       ├── processed
│       ├── raw
│       └── results
├── notebooks
│   ├── 01_data_preparation
│   │   ├── 01_download_data.ipynb
│   │   └── 02_data_cleaning.ipynb
│   ├── 02_exploration
│   │   └── 01_eda.ipynb
│   ├── 03_modeling
│   │   ├── 01_logistic_regression.ipynb
│   │   ├── 02_random_forest.ipynb
│   │   └── 03_hgb.ipynb
│   ├── 04_llm_classification
│   │   ├── 01_llm_preparation.ipynb
│   │   ├── 02_local_llm_classification.ipynb
│   │   ├── 03_remote_llm_classification.ipynb
│   │   ├── 04_collate_and_convert_results.ipynb
│   │   └── README.md
│   ├── 05_evaluation
│   │   ├── 01_performance_evaluation.ipynb
│   │   └── 02_llm_evaluation.ipynb
│   └── README.md
└── src
    └── llm_classification
        ├── datasets
        └── utils
```

* `.env` — Environment variables file holding API keys for Hugging Face and Kaggle (**users must create this themselves; see the [./notebooks/README.md](./notebooks/README.md) for details**).
* `LICENSE` — Project license file.
* `README.md` — This documentation file.
* `data/` — Directory for datasets, results, and analysis outputs.
  * `GiveMeSomeCredit/` — Workspace for the Kaggle *Give Me Some Credit* dataset and derived files.
* [`notebooks/`](./notebooks) — Jupyter notebooks organized by project phase:  
  * [`01_data_preparation/`](./notebooks/01_data_preparation) — Data downloading, cleaning, and preprocessing notebooks.
  * [`02_exploration/`](./notebooks/02_exploration) — Exploratory data analysis notebooks.
  * [`03_modeling/`](./notebooks/03_modeling) — Baseline traditional model training notebooks (e.g., logistic regression, random forests).
  * [`04_llm_classification/`](./notebooks/04_llm_classification) — Notebooks for LLM-based classification experiments.
  * [`05_evaluation/`](./notebooks/05_evaluation) — Notebooks focused on performance evaluation and visualization.
* [`src/`](./src) — Python source code for utility scripts, dataset handling, and core functions.

