import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import os
import yaml
import matplotlib.pyplot as plt
import argparse

from src.model.model import TrajTransformer, ETARegressor
from src.utils.experiment_manager import ExperimentManager

# ===============================
# Dataset
# ===============================
class ETADataset(Dataset):
    def __init__(self, X, time, y, mask=None):
        self.X_traj = torch.tensor(X, dtype=torch.float32)
        self.X_context = torch.tensor(time, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32) if mask is not None else None

    def __getitem__(self, idx):
        if self.mask is not None:
            return self.X_traj[idx], self.X_context[idx], self.y[idx], self.mask[idx]
        return self.X_traj[idx], self.X_context[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


def to_padding_mask(traj_mask, device):
    """Convert validity mask (1=valid, 0=pad) to padding mask (True=pad)."""
    return ~(traj_mask.bool().to(device))

# ===============================
# Train Epoch
# ===============================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        if len(batch) == 4:
            x_traj, x_context, y, traj_mask = batch
            padding_mask = to_padding_mask(traj_mask, device)
        else:
            x_traj, x_context, y = batch
            padding_mask = None

        x_traj = x_traj.to(device)
        x_context = x_context.to(device)
        y = y.to(device)

        pred = model(x_traj, x_context, padding_mask=padding_mask)
        loss = criterion(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x_traj.size(0)
    
    return total_loss / len(dataloader.dataset)

# ===============================
# Evaluation Epoch
# ===============================
@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        if len(batch) == 4:
            x_traj, x_context, y, traj_mask = batch
            padding_mask = to_padding_mask(traj_mask, device)
        else:
            x_traj, x_context, y = batch
            padding_mask = None

        x_traj = x_traj.to(device)
        x_context = x_context.to(device)
        y = y.to(device)

        pred = model(x_traj, x_context, padding_mask=padding_mask)
        loss = criterion(pred, y)
        total_loss += loss.item() * x_traj.size(0)

    return total_loss / len(dataloader.dataset)

# ===============================
# Test & Metrics & Plotting
# ===============================
@torch.no_grad()
def test_model(model, dataloader, device, scaler_params, output_dir, logger):
    model.eval()
    criterion = nn.L1Loss()
    
    true_vals = []
    pred_vals = []
    
    eta_mean = scaler_params['mean']
    eta_std = scaler_params['std']

    test_loss = 0.0

    for batch in tqdm(dataloader, desc="Testing"):
        if len(batch) == 4:
            x_traj, x_context, y, traj_mask = batch
            padding_mask = to_padding_mask(traj_mask, device)
        else:
            x_traj, x_context, y = batch
            padding_mask = None

        x_traj = x_traj.to(device)
        x_context = x_context.to(device)
        y = y.to(device)

        pred = model(x_traj, x_context, padding_mask=padding_mask)
        loss = criterion(pred, y)
        test_loss += loss.item() * x_traj.size(0)

        y_real = y * eta_std + eta_mean
        pred_real = pred * eta_std + eta_mean
        
        true_vals.append(y_real.cpu().numpy())
        pred_vals.append(pred_real.cpu().numpy())

    true_vals = np.concatenate(true_vals)
    pred_vals = np.concatenate(pred_vals)
    
    mae = np.mean(np.abs(true_vals - pred_vals))
    rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
    mape = np.mean(np.abs((pred_vals - true_vals) / (true_vals + 1e-8))) * 100

    logger.info(f"Test Loss (Norm L1): {test_loss / len(dataloader.dataset):.4f}")
    logger.info(f"TEST MAPE: {mape:.2f}%")
    logger.info(f"TEST RMSE: {rmse:.4f}")
    logger.info(f"TEST MAE:  {mae:.4f}")

    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(true_vals, pred_vals, alpha=0.5, s=10, color='steelblue')
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
    plt.xlabel("Ground Truth ETA (seconds)")
    plt.ylabel("Predicted ETA (seconds)")
    plt.title("ETA Prediction vs. Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    save_path = os.path.join(img_dir, "eta_prediction_scatter.png")
    plt.savefig(save_path, dpi=300)
    logger.info(f"Scatter plot saved to {save_path}")

# ===============================
# Main
# ===============================
def main(config_path):
    exp = ExperimentManager(config_path)
    config = exp.config
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    
    fh = logging.FileHandler(exp.get_log_path("finetune.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(f"===== Start Finetuning Experiment =====")
    logger.info(f"Run ID: {exp.run_id}")
    logger.info(f"Output Dir: {exp.get_run_dir()}")

    logger.info(f"Loading data from {config['data']['npz_path']}...")
    data = np.load(config['data']['npz_path'])
    
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    t_train, t_val, t_test = data['time_encoding_train'], data['time_encoding_val'], data['time_encoding_test']
    eta_train, eta_val, eta_test = data['eta_train'], data['eta_val'], data['eta_test']
    mask_train = data['mask_train'] if 'mask_train' in data else None
    mask_val = data['mask_val'] if 'mask_val' in data else None
    mask_test = data['mask_test'] if 'mask_test' in data else None

    eta_mean = np.mean(eta_train)
    eta_std = np.std(eta_train)
    scaler_params = {'mean': eta_mean, 'std': eta_std}
    logger.info(f"Data Statistics -> ETA Mean: {eta_mean:.2f}, ETA Std: {eta_std:.2f}")

    y_train_norm = (eta_train - eta_mean) / eta_std
    y_val_norm = (eta_val - eta_mean) / eta_std
    y_test_norm = (eta_test - eta_mean) / eta_std

    batch_size = config['train']['batch_size']
    train_loader = DataLoader(ETADataset(X_train, t_train, y_train_norm, mask_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ETADataset(X_val, t_val, y_val_norm, mask_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ETADataset(X_test, t_test, y_test_norm, mask_test), batch_size=batch_size, shuffle=False)

    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    use_driver_emb = config['model'].get('use_driver_emb', True)
    num_drivers = None
    if use_driver_emb:
        with open(config['data']['postman_map_path'], "rb") as f:
            postman_map = pickle.load(f)
        num_drivers = len(postman_map) + 1
        logger.info(f"Using driver embedding ({num_drivers} driver slots)")
    else:
        logger.info("Driver embedding disabled (time-only context)")
    
    encoder = TrajTransformer(
        input_dim=config['model'].get('input_dim', 5), 
        model_dim=config['model'].get('model_dim', 128)
    ).to(device)
    
    pretrained_path = config['model']['pretrained_path']
    if os.path.exists(pretrained_path):
        encoder.load_state_dict(torch.load(pretrained_path, map_location=device))
        logger.info(f"Loaded pretrained encoder from: {pretrained_path}")
    else:
        logger.warning(f"Pretrained model NOT found at {pretrained_path}. Using random initialization.")

    if config['model'].get('freeze_encoder', True):
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder parameters are FROZEN.")
    else:
        logger.info("Encoder parameters are TRAINABLE.")

    model = ETARegressor(
        encoder=encoder,
        num_drivers=num_drivers,      
        time_input_dim=config['model'].get('time_input_dim', 10),        
        driver_emb_dim=config['model'].get('driver_emb_dim', 16),          
        hidden_dim=config['model'].get('hidden_dim', 128),             
        num_attn_heads=config['model'].get('num_attn_heads', 4),
        use_driver_emb=use_driver_emb,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    
    epochs = config['train']['epochs']
    patience = config['train']['patience']
    best_val_loss = float('inf')
    patience_counter = 0

    logger.info("Starting training loop...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_epoch(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            saved_path = exp.save_model(model, "best_model.pt")
            logger.info(f"Validation improved. Model saved to {saved_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    logger.info("\n===== Testing Best Model =====")
    best_model_path = os.path.join(exp.model_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        test_model(model, test_loader, device, scaler_params, exp.get_run_dir(), logger)
    else:
        logger.error("Best model file not found, skipping testing.")

    logger.info("Experiment finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml file")
    args = parser.parse_args()
    
    main(args.config)