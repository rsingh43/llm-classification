import os

import pandas as pd
import sklearn.model_selection

from ..dataset_manager import DatasetManager
from ..utils import kaggle_utils

class GiveMeSomeCreditDataset(DatasetManager):
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

	def __init__(self, project_root=None):
		super().__init__("GiveMeSomeCredit", project_root=project_root)

	# --- Dataset Download ---
	def download_dataset(self, force=False):
		kaggle_utils.download_competition_dataset(
			self.dataset_name, self.get_dataset_dir(), force=force
		)
		return self.get_dataset_dir()

	# --- Raw Data Loaders ---
	def load_training_data(self, verbose=True):
		return self.load_raw_dataframe(
			self.TRAINING_DATA_BASENAME, header=0, index_col=0, nrows=None, verbose=verbose
		)

	def load_testing_data(self, verbose=True):
		return self.load_raw_dataframe(
			self.TESTING_DATA_BASENAME, header=0, index_col=0, nrows=None, verbose=verbose
		)

	def load_sample_entry_data(self, verbose=True):
		return self.load_raw_dataframe(
			self.SAMPLE_ENTRY_BASENAME, header=0, index_col=0, nrows=None, verbose=verbose
		)

	def load_data_dictionary(self, verbose=True):
		path = self.get_raw_file_path(self.DATA_DICTIONARY_BASENAME)
		return self._load_xls_dataframe(
			path, verbose=verbose,
			skiprows=0, header=1, index_col=0
		)
		
	# --- Results Split Management ---
	def create_and_return_result_splits(self, suppress_logs=False):
		df = self.load_training_data(verbose=False)

		ids = df.index
		target = df["SeriousDlqin2yrs"]
		training_ids, validation_ids = sklearn.model_selection.train_test_split(
			ids, test_size=0.2, stratify=target, random_state=0
		)

		training_split_df = pd.DataFrame(index=sorted(training_ids))
		validation_split_df = pd.DataFrame(index=sorted(validation_ids))

		training_split_df.index.name = "Row ID"
		validation_split_df.index.name = "Row ID"

		self.save_results_dataframe(
			training_split_df, self.TRAINING_RESULTS_BASENAME, suppress_logs=suppress_logs
		)
		self.save_results_dataframe(
			validation_split_df, self.VALIDATION_RESULTS_BASENAME, suppress_logs=suppress_logs
		)

		return training_split_df, validation_split_df

	def ensure_results_exist(self):
		train_path = self.get_results_file_path(self.TRAINING_RESULTS_BASENAME)
		val_path = self.get_results_file_path(self.VALIDATION_RESULTS_BASENAME)

		train_exists = os.path.exists(train_path)
		val_exists = os.path.exists(val_path)

		if not train_exists and not val_exists:
			self.create_and_return_result_splits()
		elif train_exists != val_exists:
			raise FileNotFoundError(
				f"File existence mismatch: {train_path} exists={train_exists}, "
				f"{val_path} exists={val_exists}. Both must either exist or not exist."
			)

	def load_training_results(self, verbose=True):
		self.ensure_results_exist()
		tmp_df = self.load_results_dataframe(
			self.TRAINING_RESULTS_BASENAME, header=None, index_col=None, nrows=0,
			verbose=False
		)
		header = 0 if len(tmp_df.columns) == 1 else [0, 1]
		return self.load_results_dataframe(
			self.TRAINING_RESULTS_BASENAME, header=header, index_col=0,
			verbose=verbose
		)

	def load_validation_results(self, verbose=True):
		self.ensure_results_exist()
		tmp_df = self.load_results_dataframe(
			self.VALIDATION_RESULTS_BASENAME, header=None, index_col=None, nrows=0,
			verbose=False
		)
		header = 0 if len(tmp_df.columns) == 1 else [0, 1]
		return self.load_results_dataframe(
			self.VALIDATION_RESULTS_BASENAME, header=header, index_col=0,
			verbose=verbose
		)

	def get_training_row_ids(self, verbose=False):
		return self.load_training_results(verbose=verbose).index.tolist()

	def get_validation_row_ids(self, verbose=False):
		return self.load_validation_results(verbose=verbose).index.tolist()

	# --- Update and Merge Results ---
	@staticmethod
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
				grouped_sub_columns = GiveMeSomeCreditDataset._group_columns(stripped_sub_columns)
				new_columns.extend([
					(top_column, *sub_column)
					for sub_column in grouped_sub_columns
				])
			else:
				new_columns.extend(sub_columns)
		return new_columns

	@staticmethod
	def _ensure_multilevel(df, inplace=False):
		if not inplace:
			df = df.copy()
		if not isinstance(df.columns, pd.MultiIndex):
			df.columns = pd.MultiIndex.from_arrays([df.columns])
		return df

	@staticmethod
	def _add_top_level_column(df, name, inplace=False):
		df = GiveMeSomeCreditDataset._ensure_multilevel(df, inplace=inplace)

		df.columns = pd.MultiIndex.from_tuples([
			(name,) + col
			for col in df.columns
		])

		return df

	@staticmethod
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

	@staticmethod
	def _normalize_multiindex(*dfs, inplace=False):
		dfs = [
			GiveMeSomeCreditDataset._ensure_multilevel(df, inplace=inplace)
			for df in dfs
		]
		max_levels = max(df.columns.nlevels for df in dfs)
		dfs = [
			GiveMeSomeCreditDataset._pad_levels(df, max_levels, inplace=inplace)
			for df in dfs
		]
		return dfs if len(dfs) > 1 else dfs[0]

	@staticmethod
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

		training_df = training_df[
			GiveMeSomeCreditDataset._group_columns(training_df.columns)
		]
		validation_df = validation_df[
			GiveMeSomeCreditDataset._group_columns(validation_df.columns)
		]
		return training_df, validation_df

	# --- Append Model Results to Processed Files ---
	def save_train_validation_results(self, model, new_results_df, suppress_logs=False, verbose=False):
		training_df = self.load_training_results(verbose=verbose)
		validation_df = self.load_validation_results(verbose=verbose)
		new_results_df = GiveMeSomeCreditDataset._add_top_level_column(
			new_results_df, model, inplace=False
		)

		GiveMeSomeCreditDataset._normalize_multiindex(
			training_df, validation_df, new_results_df, inplace=True
		)   

		updated_training_df, updated_validation_df = GiveMeSomeCreditDataset._update_results_df(
			training_df, validation_df, new_results_df
		)   

		self.save_results_dataframe(
			updated_training_df, self.TRAINING_RESULTS_BASENAME, suppress_logs=suppress_logs
		)
		self.save_results_dataframe(
			updated_validation_df, self.VALIDATION_RESULTS_BASENAME, suppress_logs=suppress_logs
		)

	def load_training_validation_results(self, verbose=False):
		return (
			self.load_training_results(verbose=verbose),
			self.load_validation_results(verbose=verbose),
		)

	# --- Data Description / Q&A ---
	def save_data_descriptions(self, descriptions_df, suppress_logs=False):
		try:
			existing_df = self.load_processed_dataframe(self.DATA_DESCRIPTIONS_BASENAME, header=0, index_col=0)
		except FileNotFoundError:
			existing_df = pd.DataFrame()
		combined_df = existing_df.combine_first(descriptions_df)
		self.save_processed_dataframe(combined_df, self.DATA_DESCRIPTIONS_BASENAME, suppress_logs=suppress_logs)

	def load_data_descriptions(self, verbose=True):
		return self.load_processed_dataframe(
			self.DATA_DESCRIPTIONS_BASENAME, header=0, index_col=0, verbose=verbose
		)

	def save_classification_questions(self, questions_df, suppress_logs=False):
		self.save_processed_dataframe(questions_df, self.CLASSIFICATION_QUESTIONS_BASENAME, suppress_logs=suppress_logs)

	def load_classification_questions(self, verbose=True):
		return self.load_processed_dataframe(
			self.CLASSIFICATION_QUESTIONS_BASENAME, header=0, index_col=0, verbose=verbose
		)

	def save_reasoning_prompts(self, prompts_df, suppress_logs=False):
		self.save_processed_dataframe(prompts_df, self.REASONINGS_PROMPTS_BASENAME, suppress_logs=suppress_logs)

	def load_reasoning_prompts(self, verbose=True):
		return self.load_processed_dataframe(
			self.REASONINGS_PROMPTS_BASENAME, header=0, index_col=0, verbose=verbose
		)

	def save_classification_responses(self, responses_df, suppress_logs=True):
		self.save_results_dataframe(
			responses_df, self.CLASSIFICATION_RESPONSES_BASENAME,
			append=True, index=False, suppress_logs=suppress_logs
		)

	def load_classification_responses(self, verbose=True):
		try:
			return self.load_results_dataframe(
				self.CLASSIFICATION_RESPONSES_BASENAME, header=0, index_col=None, verbose=verbose
			)
		except FileNotFoundError:
			return pd.DataFrame(
				columns=[
					"Row ID", "Model", "Description Column", "Classification Question ID",
					"Prediction", "Yes Probability", "No Probability"
				]
			)
