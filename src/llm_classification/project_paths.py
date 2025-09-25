import pathlib

class ProjectPaths:
	def __init__(self, project_root=None, data_folder="data"):
		try:
			current_file = pathlib.Path(__file__).resolve()
		except NameError:
			current_file = pathlib.Path.cwd()

		# Find 'src' directory
		path = current_file
		while path != path.parent:
			if path.name == "src":
				break
			path = path.parent
		else:
			raise FileNotFoundError(
				"'src' directory not found in any parent directories of the current file."
			)

		# Project root = parent of src
		self.project_root = pathlib.Path(project_root) if project_root else path.parent
		self.project_name = self.project_root.name

		# Paths
		self.data_folder = self.project_root.joinpath(data_folder)
		self.colab_base_path = pathlib.Path("/content/drive/MyDrive").joinpath(self.project_name)

		self._drive_mounted = False

	@staticmethod
	def in_colab():
		try:
			return "google.colab" in str(get_ipython())
		except NameError:
			return False

	def mount_drive_once(self):
		if not self._drive_mounted:
			from google.colab import drive
			drive.mount("/content/drive", force_remount=False)
			self._drive_mounted = True

	def get_data_dir(self):
		if self.in_colab():
			self.mount_drive_once()
			data_dir = self.colab_base_path.joinpath(self.data_folder.name)
		else:
			data_dir = self.data_folder

		data_dir.mkdir(parents=True, exist_ok=True)
		return data_dir

