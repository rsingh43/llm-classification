import functools
import huggingface_hub
import huggingface_hub.utils

def model_supports_task(model_id, task):
	model_info = huggingface_hub.model_info(model_id)
	return task in model_info.tags

def get_model_access_status(model_id, token):
	api = huggingface_hub.HfApi(token=token)
	try:
		info = api.model_info(model_id)
	except Exception as e:
		return {
			"status": "error",
			"raw": str(e)
		}

	gated = getattr(info, "gated", False)
	authorized = getattr(info, "authorized", not gated)

	if not gated:
		status = "public"
	elif authorized:
		status = "gated_authorized"
	else:
		status = "gated_unauthorized"

	return {
		"status": status,
		"gated": gated,
		"authorized": authorized,
		"raw": info
	}

def require_model_access(hf_token_param="token", model_id_param="model_id"):
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			from inspect import signature
			sig = signature(func)
			bound_args = sig.bind(*args, **kwargs)
			bound_args.apply_defaults()

			model_id = bound_args.arguments.get(model_id_param)
			hf_token = bound_args.arguments.get(hf_token_param)

			if model_id is None:
				raise RuntimeError(
					f"Access check failed: Required argument '{model_id_param}' "
					f"is missing when calling '{func.__name__}'."
				)
			if hf_token is None:
				raise RuntimeError(
					f"Access check failed: Required Hugging Face token argument '{hf_token_param}' "
					f"is missing when calling '{func.__name__}'. "
					"This token is necessary for accessing gated or private models."
				)

			result = get_model_access_status(model_id, hf_token)
			model_url = f"https://huggingface.co/{model_id}"

			status = result["status"]
			if status in ("public", "gated_authorized"):
				return func(*args, **kwargs)
			if status == "error":
				raise RuntimeError(
					f"Access check error for model '{model_id}': {result['raw']}"
				)
			if status == "unauthorized_token":
				raise RuntimeError(
					f"Access denied: The provided Hugging Face token is missing, invalid, "
					f"or lacks permissions for model '{model_id}'.\n"
					f"Make sure your token is correct and has the necessary scopes."
				)
			if status == "gated_unauthorized":
				raise RuntimeError(
					f"Access denied: Your token does not have permission to access gated model '{model_id}'.\n"
					f"Please visit {model_url} to request access and agree to the terms.\n"
					"After approval, retry with a token that has access."
				)
			raise RuntimeError(
				f"Access check failed with unexpected status '{status}' for model '{model_id}'."
			)

		return wrapper
	return decorator


@require_model_access(hf_token_param="token", model_id_param="model_id")
def download_huggingface_model(
	model_id, model_dir, force=False, token=None, use_symlinks=True
):
	try:
		model_path = huggingface_hub.snapshot_download(
			repo_id=model_id,
			local_dir=model_dir,
			local_dir_use_symlinks=use_symlinks,
			resume_download=not force,
			token=token
		)
		print("Model downloaded to:", model_path)
		return model_path
	except huggingface_hub.utils.RepositoryNotFoundError:
		print(f"Error: Model repository '{model_id}' not found.")
		raise
	except huggingface_hub.utils.EntryNotFoundError:
		print(f"Error: Some files missing in repository '{model_id}'.")
		raise
	except Exception as e:
		print(f"Unexpected error downloading model '{model_id}': {e}")
		raise

@require_model_access(hf_token_param="hf_token", model_id_param="model_id")
def check_model_access(model_id, hf_token):
	# decorator handles access check, no body needed
	pass

