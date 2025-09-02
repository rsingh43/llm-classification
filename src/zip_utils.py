from datetime import datetime
import os
import time
import zipfile

def extract_updated(zip_filename, extract_to, force=False):
	with zipfile.ZipFile(zip_filename, "r") as zip_fp:
		for item in zip_fp.infolist():
			extracted_path = os.path.normpath(os.path.join(extract_to, item.filename))

			# Path traversal protection
			if not extracted_path.startswith(os.path.abspath(extract_to)):
				raise Exception(f"Unsafe path in zip file: {item.filename}")

			should_extract = force or (
				not os.path.exists(extracted_path) or
				datetime(*item.date_time) > datetime.fromtimestamp(os.path.getmtime(extracted_path))
			)

			if should_extract:
				os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
				with open(extracted_path, "wb") as out_fp:
					out_fp.write(zip_fp.read(item))
				print(f"Extracted: {item.filename}")
			else:
				print(f"Skipped (up-to-date): {item.filename}")

