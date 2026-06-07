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

        # 生成 run_id
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config.get("experiment_name", "exp")
        self.run_id = f"{exp_name}_{timestamp}"

        output_root = self.config.get("output_dir", "outputs")
        # run 目录
        self.run_dir = os.path.join(output_root, self.run_id)
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.model_dir = os.path.join(self.run_dir, "models")

        # 创建目录
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # 保存一份 config 副本
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
        """
        适配你的项目结构的代码打包功能
        核心打包范围：项目根目录下的 configs/、scripts/、src/ 目录（完整保留结构）
        自动排除 outputs/、models/、runs/ 等大文件目录
        """
        # 项目根目录（根据你的训练脚本位置推导，避免硬编码）
        # 训练脚本在 src/trainer/train.py，所以向上退3级到 Reroute 根目录
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        
        # 默认排除规则（精准适配你的项目结构）
        if exclude_patterns is None:
            exclude_patterns = [
                # 大文件/输出目录（重点排除）
                "outputs", "models", "runs",
                # 虚拟环境
                "venv", "env", "conda_env",
                # 缓存文件
                "__pycache__", ".pytest_cache",
                # 版本控制/编辑器配置
                ".git", ".gitignore", ".idea", ".vscode", ".DS_Store",
                # 二进制/数据文件
                "*.pyc", "*.pkl", "*.npz", "*.npy", "*.pt", "*.pth",
                "*.zip", "*.tar.gz", "*.log", "*.csv", "*.xlsx",
                # 临时文件
                "*.tmp", "*.temp", "*.swp",
                # 只读文档（可选排除）
                "readme.md", "README.md"
            ]
        
        # 压缩包保存路径
        archive_path = os.path.join(self.run_dir, f"code_archive_{self.run_id}.zip")
        
        # 需要打包的核心目录（按你的项目结构）
        dirs_to_archive = [
            os.path.join(self.project_root, "configs"),  # 配置文件
            os.path.join(self.project_root, "scripts"),  # 运行脚本
            os.path.join(self.project_root, "src")       # 核心代码
        ]
        
        # 创建压缩包
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for archive_dir in dirs_to_archive:
                # 跳过不存在的目录（防止报错）
                if not os.path.exists(archive_dir):
                    print(f"⚠️ 目录 {archive_dir} 不存在，跳过打包")
                    continue
                
                # 遍历目录下的所有文件
                for root, dirs, files in os.walk(archive_dir):
                    # 过滤排除目录
                    dirs[:] = [d for d in dirs if not self._matches_pattern(os.path.join(root, d), exclude_patterns)]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        # 过滤排除文件
                        if not self._matches_pattern(file_path, exclude_patterns):
                            # 计算压缩包内的相对路径（保留项目结构）
                            # Example: <project_root>/configs/base.yaml -> configs/base.yaml
                            rel_path = os.path.relpath(file_path, self.project_root)
                            zipf.write(file_path, rel_path)
        
        print(f"✅ 代码打包完成！压缩包路径：{archive_path}")
        print(f"   打包内容：configs/、scripts/、src/ 目录（已过滤无用文件）")
        return archive_path
    
    def _matches_pattern(self, path, patterns):
        """辅助函数：判断路径是否匹配排除模式"""
        # 统一路径格式（适配Linux/Mac）
        path = Path(path).as_posix()
        # 获取文件名（用于后缀匹配）
        filename = Path(path).name
        
        for pattern in patterns:
            # 1. 后缀匹配（如 *.pyc）
            if pattern.startswith("*."):
                suffix = pattern[1:]
                if filename.endswith(suffix):
                    return True
            # 2. 目录/文件名完全匹配（如 __pycache__、outputs）
            elif pattern in path.split("/") or filename == pattern:
                return True
        
        return False