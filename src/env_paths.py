import os

# for environments without __file__ (e.g., jupyter)
try:
	CURRENT_FILE = os.path.abspath(__file__)
except NameError:
	CURRENT_FILE = os.getcwd()

SRC_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
PROJECT_NAME = os.path.basename(PROJECT_ROOT)

LOCAL_DATA_FOLDER = "data"
LOCAL_MODEL_FOLDER = "models"

COLAB_BASE_PATH = f"/content/drive/MyDrive/{PROJECT_NAME}"

# cache drive mount status
_drive_mounted = False

def in_colab():
	try:
		return "google.colab" in str(get_ipython())
	except NameError:
		return False

def mount_drive_once():
	global _drive_mounted
	if not _drive_mounted:
		from google.colab import drive
		drive.mount("/content/drive", force_remount=False)
		_drive_mounted = True

def get_data_dir():
	if in_colab():
		mount_drive_once()
		data_dir = os.path.join(COLAB_BASE_PATH, LOCAL_DATA_FOLDER)
	else:
		data_dir = os.path.join(PROJECT_ROOT, LOCAL_DATA_FOLDER)
	os.makedirs(data_dir, exist_ok=True)
	return data_dir

def get_model_dir():
	if in_colab():
		mount_drive_once()
		model_dir = os.path.join(COLAB_BASE_PATH, LOCAL_MODEL_FOLDER)
	else:
		model_dir = os.path.join(PROJECT_ROOT, LOCAL_MODEL_FOLDER)
	os.makedirs(model_dir, exist_ok=True)
	return model_dir

