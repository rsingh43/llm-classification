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

* [Kaggle API Setup Instructions](#kaggle-api-setup-instructions)
* [Hugging Face Token Setup Instructions](#hugging-face-token-setup-instructions)

If these tokens are not properly configured, notebooks that rely on external APIs will not function correctly.

## Kaggle API Setup Instructions

### Option 1: Using `kaggle.json` File

1. **Download `kaggle.json` from Kaggle:**

   - Log in to [kaggle.com](https://www.kaggle.com/).  
   - Click your profile picture (top right corner) and select **Account** from the dropdown.  
   - Scroll down to the **API** section.  
   - Click **Create New API Token**.  
   - The `kaggle.json` file will be downloaded automatically.

2. **Place the `kaggle.json` file:**

   - Move the file to the appropriate folder:  
     - Linux/macOS: `~/.kaggle/`  
     - Windows: `%USERPROFILE%\.kaggle\`

   - Set file permissions (Linux/macOS):  
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```

   - The Kaggle API will detect this file automatically when running commands or scripts.

### Option 2: Using Environment Variables with a `.env` File

1. Create a `.env` file in the root of your project directory.

2. Add your Kaggle credentials to the `.env` file:

   ```bash
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_api_key
   ```

## Hugging Face Token Setup Instructions

### Step 1: Create a Hugging Face Account

* Visit [huggingface.co](https://huggingface.co/) and sign up or log in.

### Step 2: Generate an Access Token

1. Click your profile icon (top right corner) and select **Settings**.  
2. In the left sidebar, click **Access Tokens**.  
3. Click **New token**.  
4. Name your token and select the appropriate scopes (usually `read` is sufficient).  
5. Click **Generate**.  
6. Copy the generated token — you won’t be able to see it again.

### Step 3: Set Up Your Token for Use

There are two common ways to configure the Hugging Face token in your environment:

#### Option 1: Using Environment Variable

1. Add your token as an environment variable named `HF_TOKEN`.

   - On Linux/macOS:
     ```bash
     export HF_TOKEN="your_token_here"
     ```
   - On Windows (PowerShell):
     ```powershell
     setx HF_TOKEN "your_token_here"
     ```

#### Option 2: Using a `.env` File

1. Create a `.env` file in your project root directory.  
2. Add your token to the `.env` file:
   ```ini
   HF_TOKEN=your_token_here
   ```
