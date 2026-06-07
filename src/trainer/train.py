import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os
import yaml
import torch.nn.functional as F

from src.model.model import TrajTransformer
from src.loss.cl_loss import LHInfoNCELoss
from src.dataprocess.TrajectoryContrastiveDataset import TrajectoryContrastiveDataset
from src.utils.experiment_manager import ExperimentManager

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证 CuDNN 结果一致性 (会牺牲一点点速度，但为了复现性值得)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


# ===============================
# InfoNCE loss
# ===============================
def info_nce_loss(features, temperature=0.1):
    features = F.normalize(features, dim=1)
    sim = torch.matmul(features, features.T)

    B = features.shape[0] // 2
    labels = torch.arange(B, device=features.device)
    loss1 = F.cross_entropy(sim[:B] / temperature, labels + B)
    loss2 = F.cross_entropy(sim[B:] / temperature, labels)

    return (loss1 + loss2) / 2

def get_dynamic_cl_weight(current_epoch, total_warmup_epochs, target_lambda):
    """
    线性预热策略
    Args:
        current_epoch: 当前是第几个 epoch (从0开始)
        total_warmup_epochs: 预热总共持续多少个 epoch
        target_lambda: 预热结束后的最终权重
    """
    # 如果已经过了预热期，直接返回目标权重
    if current_epoch >= total_warmup_epochs:
        return target_lambda
    
    # 线性增长: (当前 / 总数) * 目标
    # 第0轮: 0/5 * 0.1 = 0.0
    # 第1轮: 1/5 * 0.1 = 0.02
    return target_lambda * (current_epoch / total_warmup_epochs)

# ===============================
# Train epoch
# ===============================
def train_epoch(model, dataloader, optimizer, lh_loss_fn , device, config, logger, epoch_idx):
    model.train()
    total_loss = total_mae_loss = total_nsp_loss = total_cl_loss = 0.0

    lambda_mae = config["loss"]["lambda_mae"]
    lambda_nsp = config["loss"]["lambda_nsp"]

    target_lambda_cl = config["loss"]["lambda_cl"]
    warmup_epochs =config["loss"]["warmup_epochs"]
    
    current_lambda_cl = get_dynamic_cl_weight(epoch_idx, warmup_epochs, target_lambda_cl)

    if epoch_idx < warmup_epochs: 
        logger.info(f"Epoch {epoch_idx}: CL Warm-up active. Weight: {current_lambda_cl:.5f} (Target: {target_lambda_cl})")

    mae_loss_fn = torch.nn.MSELoss()
    nsp_loss_fn = torch.nn.MSELoss()

    for batch in tqdm(dataloader, desc="Training", leave=False):
        anchor_seq = batch['anchor_seq'].to(device)
        anchor_mask = batch['anchor_mask'].to(device)
        positive_seq = batch['positive_seq'].to(device)
        positive_mask = batch['positive_mask'].to(device)
        # anchor_mask: 1 = 有效, 0 = Padding
        # positive_mask: 1 = 有效, 0 = Padding

        # ===== 对比学习 =====
        loss_cl = torch.tensor(0.0, device=device) # 初始化为0，防止报错
        if current_lambda_cl > 0:
            # 1. 拼接序列 (dim=0) -> (2B, T, D)
            combined_seq = torch.cat([anchor_seq, positive_seq], dim=0)
            # 2. 拼接 Mask (dim=0) -> (2B, T)
            combined_mask_val = torch.cat([anchor_mask, positive_mask], dim=0)
            # 3. 转换 Mask 逻辑
            # PyTorch Transformer 通常要求：True = Padding (被忽略), False = 有效
            # 我的数据是：1 = 有效, 0 = Padding
            # 所以需要取反 (~)
            # 注意：mask 的第 0 位是 CLS，永远应该是“有效”(False)
            # 但既然数据里第 0 位已经是 1 (有效) 了，直接取反就行
            padding_mask = ~(combined_mask_val.bool()) 
            # padding_mask: 1 = 掩码, 0 = 非掩码
            # 4. 传入 Mask 计算 Embedding
            cls_embeddings = model.get_embedding(combined_seq, mask=padding_mask)
            # loss_cl = info_nce_loss(cls_embeddings)
            loss_cl = lh_loss_fn(cls_embeddings)

        # ===== MAE =====
        batch_size, seq_len, _ = anchor_seq.shape
        padding_mask = anchor_mask
        # padding_mask: 1 = 有效, 0 = Padding
        key_padding_mask = ~(anchor_mask.bool())
        # key_padding_mask: 1 = 掩码, 0 = 非掩码
        mae_mask = model.generate_mae_mask(batch_size, seq_len-1, device=device)
        # mae_mask: 1 = 掩码, 0 = 非掩码
        mae_mask = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            mae_mask
        ], dim=1) & padding_mask
        # 只有有效且被掩码的位置才会被用于MAE任务

        out_mae, _ = model(anchor_seq, task="mae", mask_pos=mae_mask, key_padding_mask=key_padding_mask)
        # mask_pos 表示用于MAE任务的掩码位置
        # key_padding_mask 表示自注意力需要忽略的Padding位置

        target_mask = ~torch.isnan(anchor_seq)
        # 忽略 NaN 目标
        final_mask = mae_mask.unsqueeze(-1) & target_mask
        # 只有有效且被掩码且目标非NaN的位置才会被用于计算MAE损失
        loss_mae = mae_loss_fn(out_mae[final_mask], anchor_seq[final_mask])

        # ===== NSP =====
        seq_len_with_cls = seq_len + 1 # +1 为 CLS 预留位置, 序列长度为 T+1
        causal_mask = model.generate_causal_mask(seq_len_with_cls, device=device)
        out_nsp = model(anchor_seq, task='nsp', attn_mask=causal_mask)

        nsp_mask = padding_mask[:, 2:] & padding_mask[:, 1:-1]
        target_to_use = anchor_seq[:, 2:, :4]

        out_to_use = out_nsp[:, 2:seq_len]
        
        # 计算loss
        loss_nsp = nsp_loss_fn(out_to_use[nsp_mask], target_to_use[nsp_mask])

        # ===== 总 loss =====
        loss = lambda_mae * loss_mae + lambda_nsp * loss_nsp + current_lambda_cl * loss_cl

        optimizer.zero_grad()
        loss.backward()
        # debug
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mae_loss += loss_mae.item()
        total_nsp_loss += loss_nsp.item()
        total_cl_loss += loss_cl.item()

    logger.info(f"Training -> MAE: {total_mae_loss/len(dataloader):.4f}, "
                f"NSP: {total_nsp_loss/len(dataloader):.4f}, "
                f"CL:  {total_cl_loss/len(dataloader):.4f}")

    return total_loss / len(dataloader)



# ===============================
# Validation
# ===============================
@torch.no_grad()
def evaluate_epoch(model, dataloader, device, config):
    model.eval()
    mae_loss_fn = torch.nn.MSELoss()
    nsp_loss_fn = torch.nn.MSELoss()

    lambda_mae = config["loss"]["lambda_mae"]
    lambda_nsp = config["loss"]["lambda_nsp"]

    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        x = batch['anchor_seq'].to(device)
        padding_mask = batch['anchor_mask'].to(device)

        batch_size, seq_len, _ = x.shape

        # === MAE ===
        mae_mask = model.generate_span_mask(batch_size, seq_len - 1, device=device)
        mae_mask = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            mae_mask
        ], dim=1) & padding_mask

        out_mae, _ = model(x, task='mae', mask_pos=mae_mask)
        target_mask = ~torch.isnan(x)
        final_mask = mae_mask.unsqueeze(-1) & target_mask
        loss_mae = mae_loss_fn(out_mae[final_mask], x[final_mask])

        # === NSP ===
        seq_len_with_cls = seq_len + 1
        causal_mask = model.generate_causal_mask(seq_len_with_cls, device=device)
        out_nsp = model(x, task='nsp', attn_mask=causal_mask)

        nsp_mask = padding_mask[:, 2:] & padding_mask[:, 1:-1]
        target_to_use = x[:, 2:, :4]
        out_to_use = out_nsp[:, 2:seq_len]

        loss_nsp = nsp_loss_fn(out_to_use[nsp_mask], target_to_use[nsp_mask])

        loss = lambda_mae * loss_mae + lambda_nsp * loss_nsp
        total_loss += loss.item()

    return total_loss / len(dataloader)



# ===============================
# main
# ===============================
def main(config_path):
    set_seed(42)
    # === 加载配置 ===
    config = yaml.safe_load(open(config_path))

    # === 初始化实验管理器 ===
    exp = ExperimentManager(config_path)
    logger = logging.getLogger()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(exp.get_log_path()),
            logging.StreamHandler()
        ]
    )

    # ===== 新增：打包当前代码 =====
    # 调用 archive_code 方法打包代码
    # 可自定义排除规则，比如添加 "data" 目录
    exp.archive_code()
    
    logger = logging.getLogger()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(exp.get_log_path()),
            logging.StreamHandler()
        ]
    )

    logger.info("===== Start Training =====")
    logger.info(f"Experiment: {exp.run_id}")

    # === 加载数据集 ===
    train_dataset = TrajectoryContrastiveDataset(
        npz_path=config["data"]["npz_path"],
        scalers_path=config["data"]["scalers"],
        partition="train",
        max_len=config["data"]["max_len"],
        aug_params=config["aug"]
    )
    val_dataset = TrajectoryContrastiveDataset(
        npz_path=config["data"]["npz_path"],
        scalers_path=config["data"]["scalers"],
        partition="val",
        max_len=config["data"]["max_len"]
    )
    test_dataset = TrajectoryContrastiveDataset(
        npz_path=config["data"]["npz_path"],
        scalers_path=config["data"]["scalers"],
        partition="test",
        max_len=config["data"]["max_len"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], num_workers=4, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config["train"]["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=config["train"]["batch_size"], shuffle=False)

    # === 模型 ===
    device = torch.device(config["train"]["device"])
    model = TrajTransformer(
        input_dim=config["model"]["input_dim"],
        model_dim=config["model"]["model_dim"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    lh_loss_fn = LHInfoNCELoss(
        embedding_dim=config["model"]["model_dim"], # 比如 128
        dim_lorentz=None,      # 默认对半切分
        beta=1.0, 
        gamma_power=0.5,
        temperature=0.05
        ).to(device) # 记得放到 GPU 上

    # === Early stopping ===
    best_val = float("inf")
    patience = config["train"]["patience"]
    patience_counter = 0

    # === Training Loop ===
    for epoch in range(1, config["train"]["epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{config['train']['epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, lh_loss_fn, device, config, logger, epoch)
        val_loss = evaluate_epoch(model, val_loader, device, config)

        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            exp.save_model(model, "best.pt")
            patience_counter = 0
            logger.info("Validation improved → model saved.")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping.")
                break

    # === Test ===
    logger.info("\n===== Testing best model =====")
    model.load_state_dict(torch.load(os.path.join(exp.model_dir, "best.pt")))
    test_loss = evaluate_epoch(model, test_loader, device, config)
    logger.info(f"Test Loss: {test_loss:.4f}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
