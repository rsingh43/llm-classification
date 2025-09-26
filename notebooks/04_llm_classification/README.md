# LLM Classification Notebooks (/notebooks/04_llm_classification)

This directory contains Jupyter notebooks for performing LLM-based classification on tabular data.  
These notebooks transform tabular rows into natural language prompts, send them to an LLM, and collect predictions.

## Prerequisites

Before running these notebooks, make sure you have:

* Completed the steps in [`../01_data_preparation`](../01_data_preparation/) to download and preprocess the data.
* Set up your Kaggle and Hugging Face API access tokens as described in the [README.md](../README.md) inside the [/notebooks](..) directory.

Without these prerequisites, the notebooks in this directory will not run correctly.

## Folder Contents

* [`01_llm_prompt_preparation.ipynb`](./01_llm_prompt_preparation.ipynb) &mdash; Generate natural language prompts from the processed dataset.

* [`02_remote_llm_classification.ipynb`](./02_remote_llm_classification.ipynb) &mdash; Use the Hugging Face Inference API to make predictions.

* [`03_local_llm_classification.ipynb`](./03_local_llm_classification.ipynb) &mdash; Use an LLM hosted locally to make predictions.

* [`04_collate_and_convert_results.ipynb`](./04_collate_and_convert_results.ipynb) &mdash; Combine and reformat results from remote/local runs into a standardized format for evaluation.

## Workflow

1. Generate prompts from the processed dataset.
2. Run the classification step with either a remote or local LLM.
3. Collate and convert results for later analysis.

## Notes

* These notebooks **do not** train models; they format data, run inference using LLMs, and save outputs.
* Prompt design plays a major role in performance. See [Prompt Generation Process](#prompt-generation-process) for details on how prompts are constructed.

## Methodology

This project approaches tabular data classification by converting each row into a natural language description. The large language model (LLM) then processes this textual representation to predict the class label and generate a human-readable explanation for its decision.

### Data Conversion to Natural Language

Each row from the tabular dataset is converted into a coherent natural language sentence or paragraph. Feature names and values are combined into descriptive phrases, and any missing values are omitted from the description to allow flexible handling by the LLM.

While it is possible to use an LLM to generate these natural language descriptions dynamically for each data row, this approach can introduce variability. Without strict control over randomness or prompt consistency, the same data instance might produce different prompt texts across runs, potentially impacting reproducibility and downstream classification reliability. Additionally, relying on an LLM for prompt generation adds computational overhead and complexity.

To mitigate these issues, a standardized, fixed-format template for the natural language descriptions was created. This approach ensures that each data row is consistently converted into the same descriptive text, improving reproducibility and simplifying debugging and evaluation. The following section describes the process for obtaining the standardized prompts.

#### Prompt Generation Process

To generate natural language descriptions, an example data row was input into ChatGPT with a prompt requesting a natural language summary of the data. The prompt used was:

> Create a natural language description of this data:  
RevolvingUtilizationOfUnsecuredLines: 0.766127  
age: 45  
NumberOfTime30-59DaysPastDueNotWorse: 2  
DebtRatio: 0.802982  
MonthlyIncome: 9120.0  
NumberOfOpenCreditLinesAndLoans: 13  
NumberOfTimes90DaysLate: 0  
NumberRealEstateLoansOrLines: 6  
NumberOfTime60-89DaysPastDueNotWorse: 0  
NumberOfDependents: 2.0


ChatGPT responded with the following summary:

> The individual is 45 years old with a monthly income of $9,120 and supports 2 dependents. Their revolving utilization of unsecured lines is approximately 76.6%. Their debt ratio stands at about 80.3%. They have a total of 13 open credit lines and loans, including 6 real estate loans or lines. Over the past period, they have had 2 instances of being 30 to 59 days past due, 0 instances of being 60 to 89 days past due, 0 instances of being 90 or more days late.

Next, ChatGPT was prompted to generate a Python function that produces this description given various data attributes as input variables. It was then asked to further refine the function to handle missing values gracefully while still producing coherent descriptions. The final function, `get_nl_description`, is available in the [`01_llm_preparation.ipynb`](./01_llm_preparation.ipynb) notebook.

### Classification with the LLM

Once the dataset rows have been converted into standardized natural language descriptions, each description is extended with a classification question:

> Will this individual experience serious delinquency-defined as being 90 days or more past due-within the next two years? Answer with yes or no only.

The appended question transforms the description of an individual into a complete prompt for the LLM. The prompt is then passed to the model, which generates a response. Alongside the raw response, the model's first token-level probabilities for the "yes" and "no" tokens are also recorded. These probabilities provide an estimate of the model's confidence in its classification and may be used in tasks such as calibration, thresholding, or ensemble evaluation.

### Collating and Saving Results

After obtaining predictions from the LLM, the responses are processed to create a structured output suitable for analysis:

* If the response is exactly "yes" or "no", it is recorded as the predicted label, stored as `1` for yes and `0` for no.  
* Any other response is considered invalid and stored as `-1` to indicate that the model did not produce a valid classification, allowing these cases to be analyzed separately or revisited later.
* Missing values are used to indicate that no prediction was made for a given instance. This allows partial results to be saved, enabling the notebook to be stopped and resumed without losing previously computed predictions.  
* The predicted probability for "yes" is calculated for all LLM responses (including valid and invalid, but not missing) using the token-level probabilities obtained from the LLM during the classification step.


The predicted probability is computed as
```math
p_{\text{yes}} = \frac{P(\text{yes})}{P(\text{yes}) + P(\text{no})}.
```

This predicted probability has the following properties: if the LLM's discrete prediction is "yes", then $p_{\text{yes}} \ge 0.5$; if the prediction is "no", then $p_{\text{yes}} < 0.5$. Even for invalid responses, $p_{\text{yes}}$ can still be computed, enabling probability-based metrics such as ROC AUC to be calculated consistently.

