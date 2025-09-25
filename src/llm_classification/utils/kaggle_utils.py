import os

import dotenv
dotenv.load_dotenv()

from . import zip_utils

def download_competition_dataset(competition, dataset_dir, unzip=True, force=False):
	from kaggle.api.kaggle_api_extended import KaggleApi 
	api = KaggleApi()
	api.authenticate()

	os.makedirs(dataset_dir, exist_ok=True)
	dataset_zip = os.path.join(dataset_dir, f"{competition}.zip")
	raw_dir = os.path.join(dataset_dir, "raw")
	os.makedirs(raw_dir, exist_ok=True)  # ensure raw dir exists

	try:
		if not os.path.exists(dataset_zip) or force:
			print("Downloading competition dataset...")
			api.competition_download_files(competition, path=dataset_dir)
			print("Dataset downloaded to:", dataset_zip)
		else:
			print("Zip file already exists at:", dataset_zip)

		if unzip:
			if os.path.exists(dataset_zip):
				print(f"Extracting dataset to: {raw_dir} ...")
				zip_utils.extract_zip(dataset_zip, raw_dir, force)
			else:
				raise FileNotFoundError("Expected ZIP file was not found after download.")

	except Exception as e:
		error_msg = str(e).lower()
		if "not found" in error_msg or "does not exist" in error_msg:
			raise FileNotFoundError(f"Competition '{competition}' not found or inaccessible.")
		elif "not accepted" in error_msg or "accept rules" in error_msg or "forbidden" in error_msg:
			url = f"https://www.kaggle.com/c/{competition}"
			raise PermissionError(
				f"You must accept the competition rules on Kaggle before downloading the data: {url}"
			)
		else:
			status = getattr(e, "status", "unknown")
			raise RuntimeError(f"Kaggle API error (status {status}): {e}")

