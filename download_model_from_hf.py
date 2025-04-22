from huggingface_hub import snapshot_download

# Tu repositorio en Hugging Face
repo_id = "NONHUMAN-RESEARCH/demo-isaac-gr00t"

# Carpeta donde quieres guardarlo
local_dir = "outputs"

# Descargar solo los archivos del repositorio, sin cargar el modelo
snapshot_download(repo_id=repo_id, local_dir=local_dir)
