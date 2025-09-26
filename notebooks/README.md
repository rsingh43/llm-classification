# Notebooks Directory Overview (/notebooks)

This folder contains Jupyter notebooks organized by project phase, covering data preparation, exploration, modeling, LLM-based classification experiments, and evaluation.

## Folder Structure

* [`./01_data_preparation`](./01_data_preparation) &mdash; Notebooks for downloading, cleaning, and preprocessing the dataset.

* [`./02_exploration`](./02_exploration) &mdash; Notebooks focused on exploratory data analysis (EDA), visualizing data, and initial insights.

* [`./03_modeling`](./03_modeling) &mdash; Traditional machine learning models are trained here.

* [`./04_llm_classification`](./04_llm_classification) &mdash; Notebooks for preparing prompts and conducting LLM-based tabular data classification.

* [`./05_evaluation`](./05_evaluation) &mdash; Performance evaluation, comparison of models, and visualization of results.

## Purpose

These notebooks provide a step-by-step guide through the project workflow, from raw data to final model evaluation. They are intended to be run in order, allowing users to reproduce the project pipeline and understand each phase.

## Prerequisites

Before running any notebooks, you need to set up your Kaggle and Hugging Face API tokens. For instructions on how to obtain and configure your tokens, see the following sections:

* [Kaggle API Authentication](#kaggle-api-authentication)
* [Hugging Face API Authentication](#hugging-face-api-authentication)

If these tokens are not properly configured, notebooks that rely on external APIs will not function correctly.

## Kaggle API Authentication

### Step 1: Create a Kaggle Account
* Sign up or log in at [kaggle.com](https://www.kaggle.com/).

### Step 2: Generate API Token
1. Go to Profile → **Account** → **API** → **Create New API Token**.
2. A `kaggle.json` file will be downloaded.

### Step 3: Configure API Token

#### Option 1: Using `kaggle.json` File
1. Move the `kaggle.json` file to the appropriate folder:
   * Linux/macOS: `~/.kaggle/`
   * Windows: `%USERPROFILE%\.kaggle\`
2. Set file permissions (Linux/macOS):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. The Kaggle API will detect this file automatically.

#### Option 2: Using `.env` File
1. Create a .env file in the project root.
2. Add your credentials:
   ```bash
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_api_key
   ```

## Hugging Face API Authentication

### Step 1: Create a Hugging Face Account
* Sign up or log in at [huggingface.co](https://huggingface.co/).

### Step 2: Generate an Access Token
1. Click your profile icon, **Settings**, **Access Tokens**, **New token**.
2. Name the token and select the appropriate scope (usually `read`).
3. Click **Generate** and copy the token (you won’t be able to see it again).

### Step 3: Configure the Access Token

#### Option 1: Using Environment Variable
* Linux/macOS:
  ```bash
  export HF_TOKEN="your_token_here"
  ```
* Windows (PowerShell):
  ```powershell
  setx HF_TOKEN "your_token_here"
  ```

#### Option 2: Using `.env` File
1. Create a `.env` file in the project root.
2. Add your token:
   ```ini
   HF_TOKEN=your_token_here
   ```


