from abc import ABC, abstractmethod
import logging
import os

import pandas as pd

from .project_paths import ProjectPaths
from .utils import dataframe_utils

class DatasetManager(ABC):
	def __init__(self, dataset_name, project_root=None):
		self.paths = ProjectPaths(project_root=project_root)
		
		self.dataset_name = dataset_name
		
		self.dataset_dir = os.path.join(self.paths.get_data_dir(), dataset_name)
		self.raw_dir = os.path.join(self.dataset_dir, "raw")
		self.processed_dir = os.path.join(self.dataset_dir, "processed")
		self.results_dir = os.path.join(self.dataset_dir, "results")
		self.figures_dir = os.path.join(self.dataset_dir, "figures")
		
		logging.basicConfig(
			level=logging.INFO,
			format="%(asctime)s - %(levelname)s - %(message)s"
		)
		self.logger = logging.getLogger(__name__)

	# --- Optional Methods for Subclass Implementation ---
	def download_dataset(self, force=False):
		raise NotImplementedError(f"{self.dataset_name} does not implement download_dataset.")

	def load_training_data(self, verbose=True):
		raise NotImplementedError(f"{self.dataset_name} does not implement load_training_data.")
	
	def load_validation_data(self, verbose=True):
		raise NotImplementedError(f"{self.dataset_name} does not implement load_validation_data.")

	def load_testing_data(self, verbose=True):
		raise NotImplementedError(f"{self.dataset_name} does not implement load_testing_data.")

	# --- Directory Helpers ---
	def get_dataset_dir(self):
		return self.dataset_dir

	def get_raw_dir(self):
		return self.raw_dir

	def get_processed_dir(self):
		return self.processed_dir

	def get_results_dir(self):
		return self.results_dir

	def get_figures_dir(self):
		return self.figures_dir

	def get_raw_file_path(self, filename=None):
		return self.raw_dir if filename is None else os.path.join(self.raw_dir, filename)

	def get_processed_file_path(self, filename=None):
		return self.processed_dir if filename is None else os.path.join(self.processed_dir, filename)

	def get_results_file_path(self, filename=None):
		return self.results_dir if filename is None else os.path.join(self.results_dir, filename)

	def get_figures_file_path(self, filename=None):
		return self.figures_dir if filename is None else os.path.join(self.figures_dir, filename)

	# --- Private Save/Load Methods ---
	def _save_csv_dataframe(self, df, path, append=False, index=True, suppress_logs=False, log_message=None):
		os.makedirs(os.path.dirname(path), exist_ok=True)
		if append and os.path.exists(path):
			df.to_csv(path, mode="a", header=False, index=index)
			if not suppress_logs:
				self.log(log_message or f"Appended {len(df)} rows to existing CSV: {path}")
		else:
			df.to_csv(path, index=index)
			if not suppress_logs:
				self.log(log_message or f"Saved DataFrame to: {path}")

	def _load_csv_dataframe(self, path, header=None, index_col=0, nrows=None, verbose=False):
		if not os.path.exists(path):
			raise FileNotFoundError(f"File not found: {path}")
		df = pd.read_csv(path, header=header, index_col=index_col, nrows=nrows)
		if verbose:
			dataframe_utils.print_dataframe_info(df, name=os.path.basename(path))
		return df

	def _load_xls_dataframe(self, path, verbose=False, **read_excel_kwargs):
		if verbose:
			try:
				from IPython.display import display, HTML
			except ImportError:
				raise ImportError("IPython is required for verbose display but is not installed.")

		sheet_map = pd.read_excel(path, sheet_name=None, **read_excel_kwargs)
		result = {}
		for sheet_name, df in sheet_map.items():
			if verbose:
				display(HTML(f"<h1>{sheet_name}</h1>"))
				dataframe_utils.print_dataframe_info(df, name=sheet_name)
			result[sheet_name] = df
		return result

	# --- Public Save Methods ---
	def save_processed_dataframe(self, df, filename, append=False, index=True, suppress_logs=False):
		path = self.get_processed_file_path(filename)
		self._save_csv_dataframe(
			df, path, append=append, index=index, suppress_logs=suppress_logs,
			log_message=f"Saved DataFrame to processed directory: {path}"
		)

	def save_results_dataframe(self, df, filename, append=False, index=True, suppress_logs=False):
		path = self.get_results_file_path(filename)
		self._save_csv_dataframe(
			df, path, append=append, index=index, suppress_logs=suppress_logs,
			log_message=f"Saved DataFrame to results directory: {path}"
		)
	
	def save_results_figure(self, fig, filename, suppress_logs=False):
		path = self.get_figures_file_path(filename)
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path, bbox_inches="tight")
		if not suppress_logs:
			self.log(f"Saved figure to figures directory: {path}")

	# --- Public Load Methods ---
	def load_raw_dataframe(self, filename, header=None, index_col=0, nrows=None, verbose=True):
		path = self.get_raw_file_path(filename)
		return self._load_csv_dataframe(
			path, header=header, index_col=index_col, nrows=nrows, verbose=verbose
		)

	def load_processed_dataframe(self, filename, header=None, index_col=0, nrows=None, verbose=True):
		path = self.get_processed_file_path(filename)
		return self._load_csv_dataframe(
			path, header=header, index_col=index_col, nrows=nrows, verbose=verbose
		)

	def load_results_dataframe(self, filename, header=None, index_col=0, nrows=None, verbose=True):
		path = self.get_results_file_path(filename)
		return self._load_csv_dataframe(
			path, header=header, index_col=index_col, nrows=nrows, verbose=verbose
		)

	# --- Logger Helper ---
	def log(self, message, level="info"):
		level = level.lower()
		if level == "debug":
			self.logger.debug(message)
		elif level == "info":
			self.logger.info(message)
		elif level == "warning":
			self.logger.warning(message)
		elif level == "error":
			self.logger.error(message)
		elif level == "critical":
			self.logger.critical(message)
		else:
			raise ValueError(f"Invalid log level: {level}")

