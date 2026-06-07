import os
import yaml
import datetime
import shutil
import torch
import zipfile
from pathlib import Path

class ExperimentManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = yaml.safe_load(open(config_path))

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config.get("experiment_name", "exp")
        self.run_id = f"{exp_name}_{timestamp}"

        output_root = self.config.get("output_dir", "outputs")
        self.run_dir = os.path.join(output_root, self.run_id)
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.model_dir = os.path.join(self.run_dir, "models")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        shutil.copy(config_path, os.path.join(self.run_dir, "config.yaml"))

    def get_log_path(self, filename="train.log"):
        return os.path.join(self.log_dir, filename)

    def save_model(self, model, name="model.pt"):
        save_path = os.path.join(self.model_dir, name)
        torch.save(model.state_dict(), save_path)
        return save_path

    def get_run_dir(self):
        return self.run_dir

    def archive_code(self, exclude_patterns=None):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        
        if exclude_patterns is None:
            exclude_patterns = [
                "outputs", "models", "runs",
                "venv", "env", "conda_env",
                "__pycache__", ".pytest_cache",
                ".git", ".gitignore", ".idea", ".vscode", ".DS_Store",
                "*.pyc", "*.pkl", "*.npz", "*.npy", "*.pt", "*.pth",
                "*.zip", "*.tar.gz", "*.log", "*.csv", "*.xlsx",
                "*.tmp", "*.temp", "*.swp",
                "readme.md", "README.md"
            ]
        
        archive_path = os.path.join(self.run_dir, f"code_archive_{self.run_id}.zip")
        
        dirs_to_archive = [
            os.path.join(self.project_root, "configs"),  
            os.path.join(self.project_root, "scripts"),  
            os.path.join(self.project_root, "src")       
        ]
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for archive_dir in dirs_to_archive:
                if not os.path.exists(archive_dir):
                    print(f"⚠️ {archive_dir} does not exist, skipping archive")
                    continue
                
                for root, dirs, files in os.walk(archive_dir):
                    dirs[:] = [d for d in dirs if not self._matches_pattern(os.path.join(root, d), exclude_patterns)]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        if not self._matches_pattern(file_path, exclude_patterns):
                            rel_path = os.path.relpath(file_path, self.project_root)
                            zipf.write(file_path, rel_path)
        
        print(f"✅ code archive completed! archive path: {archive_path}")
        print(f"    archive content: configs/、scripts/、src/ directories (filtered out useless files)")
        return archive_path
    
    def _matches_pattern(self, path, patterns):
        path = Path(path).as_posix()
        filename = Path(path).name
        
        for pattern in patterns:
            if pattern.startswith("*."):
                suffix = pattern[1:]
                if filename.endswith(suffix):
                    return True
            elif pattern in path.split("/") or filename == pattern:
                return True
        
        return False