# Note: This script depends on the following local modules:
# - dataframe_utils.py DataFrame display and utility functions
# - env_paths.py: Provides environment-specific paths (e.g., get_data_dir)
# - kaggle_utils.py: Utilities for downloading datasets from Kaggle

# --- Imports ---
import functools
import logging
import os

import pandas as pd
import sklearn.model_selection
from IPython.display import display, HTML

import dataframe_utils
import env_paths
import kaggle_utils


# --- Logging Configuration ---
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Constants ---
DATASET_NAME = "GiveMeSomeCredit"

# --- Directory Paths ---
DATASET_DIR = os.path.join(env_paths.get_data_dir(), DATASET_NAME)
RAW_DIR = os.path.join(DATASET_DIR, "raw")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")

# --- Filenames ---
TRAINING_DATA_BASENAME = "cs-training.csv"
TESTING_DATA_BASENAME = "cs-test.csv"
SAMPLE_ENTRY_BASENAME = "sampleEntry.csv"
DATA_DICTIONARY_BASENAME = "Data Dictionary.xls"

TRAINING_RESULTS_BASENAME = "training_results.csv"
VALIDATION_RESULTS_BASENAME = "validation_results.csv"

DATA_DESCRIPTIONS_BASENAME = "data_descriptions.csv"
CLASSIFICATION_QUESTIONS_BASENAME = "classification_questions.csv"
REASONINGS_PROMPTS_BASENAME = "reasonings_prompts.csv"
CLASSIFICATION_RESPONSES_BASENAME = "classification_responses.csv"

# --- Public Directories ---
def get_dataset_dir():
	return DATASET_DIR

def get_raw_dir():
	return RAW_DIR

def get_processed_dir():
	return PROCESSED_DIR


# --- Path Helpers ---
def _get_raw_path(filename=None):
	return get_raw_dir() if filename is None else os.path.join(get_raw_dir(), filename)

def _get_processed_path(filename=None):
	return get_processed_dir() if filename is None else os.path.join(get_processed_dir(), filename)


# --- Raw Data Files ---
def get_training_data_path():
	return _get_raw_path(TRAINING_DATA_BASENAME)

def get_testing_data_path():
	return _get_raw_path(TESTING_DATA_BASENAME)

def get_sample_entry_path():
	return _get_raw_path(SAMPLE_ENTRY_BASENAME)

def get_data_dictionary_path():
	return _get_raw_path(DATA_DICTIONARY_BASENAME)


# --- Processed Result Files ---
def get_training_results_path():
	return _get_processed_path(TRAINING_RESULTS_BASENAME)

def get_validation_results_path():
	return _get_processed_path(VALIDATION_RESULTS_BASENAME)


# --- Dataset Download ---
def download_dataset(force=False):
	kaggle_utils.download_competition_dataset(DATASET_NAME, get_dataset_dir(), force=force)
	return get_dataset_dir()


# --- Data Loading ---
def _load_csv_dataframe(filename, verbose=False):
	df = pd.read_csv(filename, index_col=0)
	if verbose:
		dataframe_utils.print_dataframe_info(df, name=filename)
	return df

def _load_xls_dataframe(filename, verbose=False, **read_excel_kwargs):
	sheet_map = pd.read_excel(filename, sheet_name=None, **read_excel_kwargs)
	result = {}
	for sheet_name, df in sheet_map.items():
		if verbose:
			display(HTML(f"<h1>{sheet_name}</h1>"))
			dataframe_utils.print_dataframe_info(df, name=sheet_name)
		result[sheet_name] = df
	return result


# --- Public Loaders ---
def load_training_data(verbose=True):
	return _load_csv_dataframe(get_training_data_path(), verbose=verbose)

def load_testing_data(verbose=True):
	return _load_csv_dataframe(get_testing_data_path(), verbose=verbose)

def load_sample_entry_data(verbose=True):
	return _load_csv_dataframe(get_sample_entry_path(), verbose=verbose)

def load_data_dictionary(verbose=True):
	return _load_xls_dataframe(
		get_data_dictionary_path(), verbose=verbose,
		skiprows=0, header=1, index_col=0
	)


# --- Save/Load Processed Data ---
def save_dataframe(df, filename, append=False, index=True, suppress_logs=False):
	path = _get_processed_path(filename)
	os.makedirs(os.path.dirname(path), exist_ok=True)

	if append and os.path.exists(path):
		df.to_csv(path, mode="a", header=False, index=index)
		if not suppress_logs:
			logger.info(f"Appended {len(df)} rows to existing CSV: {path}")
	else:
		df.to_csv(path, index=index)
		if not suppress_logs:
			logger.info(f"Saved DataFrame to processed directory: {path}")


def load_dataframe(filename, header=None, index_col=0, nrows=None):
	path = _get_processed_path(filename)
	if not os.path.exists(path):
		raise FileNotFoundError(f"File not found in processed directory: {path}")
	return pd.read_csv(path, header=header, index_col=index_col, nrows=nrows)


# --- Create and Save Evaluation Splits ---
def create_and_return_result_splits(suppress_logs=False):
	df = load_training_data()

	ids = df.index
	target = df["SeriousDlqin2yrs"]
	training_ids, validation_ids = sklearn.model_selection.train_test_split(
		ids,
		test_size=0.2,
		stratify=target,
		random_state=0
	)

	training_ids = sorted(training_ids)
	validation_ids = sorted(validation_ids)

	#training_split_df = df.loc[training_ids, ["SeriousDlqin2yrs"]]
	#validation_split_df = df.loc[validation_ids, ["SeriousDlqin2yrs"]]

	#training_split_df.index.name = "ID"
	#validation_split_df.index.name = "ID"

	#multi_columns = pd.MultiIndex.from_tuples([
	#	("target", "SeriousDlqin2yrs")
	#])
	#training_split_df.columns = multi_columns
	#validation_split_df.columns = multi_columns

	training_split_df = pd.DataFrame(index=training_ids)
	validation_split_df = pd.DataFrame(index=validation_ids)
	
	training_split_df.index.name = "Row ID"
	validation_split_df.index.name = "Row ID"
	
	save_dataframe(training_split_df, TRAINING_RESULTS_BASENAME, suppress_logs=suppress_logs)
	save_dataframe(validation_split_df, VALIDATION_RESULTS_BASENAME, suppress_logs=suppress_logs)

	return training_split_df, validation_split_df


# --- Load Training/Validation Result Files ---
def ensure_results_exist():
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			training_exists = os.path.exists(get_training_results_path())
			validation_exists = os.path.exists(get_validation_results_path())

			if not training_exists and not validation_exists:
				create_and_return_result_splits()
			elif training_exists != validation_exists:
				raise FileNotFoundError(
					f"File existence mismatch: {get_training_results_path()} exists: {training_exists}, "
					f"{get_validation_results_path()} exists: {validation_exists}. "
					"Both files must either exist or not exist."
				)

			return func(*args, **kwargs)
		return wrapper
	return decorator

@ensure_results_exist()
def load_training_results():
	tmp_df =  load_dataframe(TRAINING_RESULTS_BASENAME, header=None, index_col=None, nrows=0)
	header = 0 if len(tmp_df.columns) == 1 else [0,1]
	return load_dataframe(TRAINING_RESULTS_BASENAME, header=header, index_col=0)

@ensure_results_exist()
def load_validation_results():
	tmp_df =  load_dataframe(VALIDATION_RESULTS_BASENAME, header=None, index_col=None, nrows=0)
	header = 0 if len(tmp_df.columns) == 1 else [0,1]
	return load_dataframe(VALIDATION_RESULTS_BASENAME, header=header, index_col=0)


# --- Return Evaluation Indicies ---
def get_training_row_ids():
	training_results_df = load_training_results()
	return training_results_df.index.tolist()

def get_validation_row_ids():
	validation_results_df = load_validation_results()
	return validation_results_df.index.tolist()


# --- Update and Merge Results ---
def _group_columns(columns):
	if len(columns[0]) == 1:
		return columns

	from collections import OrderedDict

	grouped = OrderedDict()
	for column in columns:
		top_column = column[0]
		grouped.setdefault(top_column, []).append(column)

	new_columns = []
	for top_column, sub_columns in grouped.items():
		if len(sub_columns[0]) > 1:
			stripped_sub_columns = [c[1:] for c in sub_columns]
			grouped_sub_columns = _group_columns(stripped_sub_columns)
			new_columns.extend([
				(top_column, *sub_column)
				for sub_column in grouped_sub_columns
			])
		else:
			new_columns.extend(sub_columns)
	return new_columns

def _ensure_multilevel(df, inplace=False):
	if not inplace:
		df = df.copy()
	if not isinstance(df.columns, pd.MultiIndex):
		df.columns = pd.MultiIndex.from_arrays([df.columns])
	return df

def _add_top_level_column(df, name, inplace=False):
	df = _ensure_multilevel(df, inplace=inplace)

	df.columns = pd.MultiIndex.from_tuples([
		(name,) + col
		for col in df.columns
	])

	return df

def _pad_levels(df, max_levels, inplace=False, pad_name_template="new_level_{}"):
	if not inplace:
		df = df.copy()

	current_levels = df.columns.nlevels
	n_to_add = max_levels - current_levels

	if n_to_add > 0:
		existing_arrays = [
			df.columns.get_level_values(i)
			for i in range(current_levels)
		]
		new_arrays = [
			[pad_name_template.format(i+1)] * len(df.columns)
			for i in range(n_to_add)
		]
		all_arrays = existing_arrays + new_arrays
		df.columns = pd.MultiIndex.from_arrays(all_arrays)

	return df

def _normalize_multiindex(*dfs, inplace=False):
	dfs = [
		_ensure_multilevel(df, inplace=inplace)
		for df in dfs
	]
	max_levels = max(df.columns.nlevels for df in dfs)
	dfs = [
		_pad_levels(df, max_levels, inplace=inplace)
		for df in dfs
	]
	return dfs if len(dfs) > 1 else dfs[0]


def _update_results_df(training_df, validation_df, new_results_df, inplace=False):
	if not inplace:
		training_df = training_df.copy()
		validation_df = validation_df.copy()
	
	results_ids = set(new_results_df.index)
	training_ids = set(training_df.index)
	validation_ids = set(validation_df.index)

	in_both = results_ids & training_ids & validation_ids
	if in_both:
		raise ValueError(f"IDs appear in both training and validation sets: {in_both}")

	not_found = results_ids - training_ids - validation_ids
	if not_found:
		raise ValueError(f"IDs not found in either training or validation sets: {not_found}")

	def ensure_columns(df, columns):
		missing_cols = [
			column for column in columns
			if column not in df.columns
		]
		for column in missing_cols:
			print("column: ", column)
			df[column] = pd.NA

	training_ids_in_results = list(results_ids & training_ids)
	if training_ids_in_results:
		ensure_columns(training_df, new_results_df.columns)
		training_df.loc[
			training_ids_in_results, new_results_df.columns
		] = new_results_df.loc[training_ids_in_results]

	validation_ids_in_results = list(results_ids & validation_ids)
	if validation_ids_in_results:
		ensure_columns(validation_df, new_results_df.columns)
		validation_df.loc[
			validation_ids_in_results, new_results_df.columns
		] = new_results_df.loc[validation_ids_in_results]

	training_df = training_df[_group_columns(training_df.columns)]
	validation_df = validation_df[_group_columns(validation_df.columns)]
	return training_df, validation_df


# --- Append Model Results to Processed Files ---
def save_train_validation_results(model, new_results_df, suppress_logs=False):
	training_df = load_training_results()
	validation_df = load_validation_results()
	new_results_df = _add_top_level_column(new_results_df, model, inplace=False)

	_normalize_multiindex(
		training_df, validation_df, new_results_df, inplace=True
	)

	updated_training_df, updated_validation_df = _update_results_df(
		training_df, validation_df, new_results_df
	)

	save_dataframe(updated_training_df, TRAINING_RESULTS_BASENAME, suppress_logs=suppress_logs)
	save_dataframe(updated_validation_df, VALIDATION_RESULTS_BASENAME, suppress_logs=suppress_logs)

def load_training_validation_results():
	return load_training_results(), load_validation_results()


# --- ??? ---
def save_data_descriptions(descriptions_df, suppress_logs=False):
	try:
		existing_df = load_data_descriptions()
	except FileNotFoundError:
		existing_df = pd.DataFrame()
	
	combined_df = existing_df.combine_first(descriptions_df)
	save_dataframe(combined_df, DATA_DESCRIPTIONS_BASENAME, suppress_logs=suppress_logs)

def load_data_descriptions():
	return load_dataframe(DATA_DESCRIPTIONS_BASENAME, header=0, index_col=0)

def save_classification_questions(questions_df, suppress_logs=False):
	save_dataframe(questions_df, CLASSIFICATION_QUESTIONS_BASENAME, suppress_logs=suppress_logs)

def load_classification_questions():
	return load_dataframe(CLASSIFICATION_QUESTIONS_BASENAME, header=0, index_col=0)

def save_reasoning_prompts(prompts_df, suppress_logs=False):
	save_dataframe(prompts_df, REASONINGS_PROMPTS_BASENAME, suppress_logs=suppress_logs)

def load_reasoning_prompts():
	return load_dataframe(REASONINGS_PROMPTS_BASENAME, header=0, index_col=0)

def save_classification_responses(responses_df, suppress_logs=True):
	save_dataframe(responses_df, CLASSIFICATION_RESPONSES_BASENAME, append=True, index=False, suppress_logs=suppress_logs)

def load_classification_responses():
	try:
		return load_dataframe(CLASSIFICATION_RESPONSES_BASENAME, header=0, index_col=None)
	except FileNotFoundError:
		return pd.DataFrame(
			columns=[
				"Row ID", "Model", "Description Column", "Classification Question ID",
				"Prediction", "Yes Probability", "No Probability"
			]
		)

