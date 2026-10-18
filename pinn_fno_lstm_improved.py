import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import random
from collections import defaultdict
import math

plt.switch_backend('Agg')

# ==========================================
# 1. 优化配置参数
# ==========================================
DATA_DIR = 'Dataset'
SAVE_DIR = 'Results_PINN_Improved'
BATCH_SIZE = 6
EPOCHS = 800
LR = 1e-3
NX, NY = 256, 256
TRAIN_SEQ_LEN = 10
MODES1, MODES2 = 36, 36
WIDTH = 144
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

# 改进的物理约束参数
PHYSICS_START_EPOCH = 1
PHYSICS_WEIGHT_FIXED = 1e-3  # 先使用固定权重试试
PHYSICS_TOLERANCE = 0.05
BOUNDARY_WEIGHT = 1e-4

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# ==========================================
# 2. 频谱卷积层
# ==========================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = (1 / (in_channels * out_channels)) ** 0.5
        
        weights1_real = self.scale * torch.randn(in_channels, out_channels, modes1, modes2) / np.sqrt(modes1 * modes2)
        weights1_imag = self.scale * torch.randn(in_channels, out_channels, modes1, modes2) / np.sqrt(modes1 * modes2)
        self.weights1 = nn.Parameter(torch.complex(weights1_real, weights1_imag))
        
        weights2_real = self.scale * torch.randn(in_channels, out_channels, modes1, modes2) / np.sqrt(modes1 * modes2)
        weights2_imag = self.scale * torch.randn(in_channels, out_channels, modes1, modes2) / np.sqrt(modes1 * modes2)
        self.weights2 = nn.Parameter(torch.complex(weights2_real, weights2_imag))

    def forward(self, x):
        batchsize = x.shape[0]
        size_1, size_2 = x.shape[2], x.shape[3]
        x_ft = torch.fft.rfft2(x, norm='ortho')
        
        out_ft = torch.zeros(batchsize, self.out_channels, size_1, size_2//2 + 1, 
                            dtype=torch.complex64, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        x = torch.fft.irfft2(out_ft, s=(size_1, size_2), norm='ortho')
        return x


# ==========================================
# 3. 通道注意力
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


# ==========================================
# 4. 改进的物理损失（缩放版本）
# ==========================================
class ImprovedPhysicsLoss(nn.Module):
    """改进的物理损失，加入自适应缩放"""
    def __init__(self, dx=1.0/256, dy=1.0/256, tolerance=0.05):
        super(ImprovedPhysicsLoss, self).__init__()
        self.dx = dx
        self.dy = dy
        self.tolerance = tolerance
        # 用于自适应缩放的缓冲区
        self.register_buffer('running_std', torch.ones(1))
        self.momentum = 0.99
        
    def forward(self, preds, mask):
        batch, T, _, H, W = preds.shape
        u = preds[:, :, 0]
        v = preds[:, :, 1]
        
        total_loss_cont = 0.0
        total_loss_bound = 0.0
        
        for t in range(T):
            u_t = u[:, t]
            v_t = v[:, t]
            
            # 计算梯度
            du_dy, du_dx = torch.gradient(u_t, spacing=(self.dy, self.dx), dim=(1, 2))
            dv_dy, dv_dx = torch.gradient(v_t, spacing=(self.dy, self.dx), dim=(1, 2))
            
            # 连续性方程
            divergence = du_dx + dv_dy
            
            mask_2d = mask.squeeze(1)
            
            # 使用相对损失，使其与数据损失在同一数量级
            # 计算散度的RMS作为归一化因子
            div_rms = torch.sqrt(torch.mean((divergence * mask_2d) ** 2) + 1e-8)
            
            # 归一化的散度
            divergence_norm = divergence / (div_rms + 1e-8)
            
            # 使用Smooth L1损失
            loss_cont = F.smooth_l1_loss(
                divergence_norm * mask_2d,
                torch.zeros_like(divergence_norm),
                beta=self.tolerance
            )
            
            # 边界条件（也进行归一化）
            boundary_mask = 1.0 - mask_2d
            
            u_rms = torch.sqrt(torch.mean((u_t * boundary_mask) ** 2) + 1e-8)
            v_rms = torch.sqrt(torch.mean((v_t * boundary_mask) ** 2) + 1e-8)
            
            loss_bound_u = F.smooth_l1_loss(
                u_t * boundary_mask / (u_rms + 1e-8),
                torch.zeros_like(u_t),
                beta=self.tolerance
            )
            loss_bound_v = F.smooth_l1_loss(
                v_t * boundary_mask / (v_rms + 1e-8),
                torch.zeros_like(v_t),
                beta=self.tolerance
            )
            
            total_loss_cont += loss_cont
            total_loss_bound += (loss_bound_u + loss_bound_v) / 2.0
        
        return total_loss_cont / T, total_loss_bound / T


# ==========================================
# 5. 主模型
# ==========================================
class EnhancedFNOLSTM2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=64):
        super(EnhancedFNOLSTM2d, self).__init__()
        self.width = width
        
        self.p = nn.Sequential(
            nn.Linear(3, width),
            nn.GELU(),
            nn.Linear(width, width)
        )
        
        self.fno_conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.fno_conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        
        num_groups = 8
        self.norm1 = nn.GroupNorm(num_groups, width)
        self.norm2 = nn.GroupNorm(num_groups, width)
        
        self.ca1 = ChannelAttention(width)
        self.ca2 = ChannelAttention(width)
        
        self.lstm_gates = nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1)
        self.norm_gates = nn.GroupNorm(num_groups, width * 4)
        
        self.norm_h = nn.GroupNorm(num_groups, width)
        self.norm_c = nn.GroupNorm(num_groups, width)
        
        self.q = nn.Sequential(
            nn.Linear(width, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, sdf, u_seq, teacher_forcing_ratio=0.5):
        batch_size, T, _, y, x = u_seq.shape
        
        h = torch.zeros(batch_size, self.width, y, x, device=u_seq.device)
        c = torch.zeros(batch_size, self.width, y, x, device=u_seq.device)
        
        preds = []
        current_input_u = u_seq[:, 0].clone()
        
        for t in range(T):
            xt = torch.cat([current_input_u, sdf], dim=1)
            xt = xt.permute(0, 2, 3, 1)
            xt = self.p(xt).permute(0, 3, 1, 2)
            
            xt1 = self.fno_conv1(xt)
            xt1 = self.w1(xt) + xt1
            xt1 = self.norm1(xt1)
            xt1 = F.gelu(xt1)
            xt1 = self.ca1(xt1)
            
            xt2 = self.fno_conv2(xt1)
            xt2 = self.w2(xt1) + xt2
            xt2 = self.norm2(xt2)
            xt2 = F.gelu(xt2)
            xt2 = self.ca2(xt2)
            
            combined = torch.cat([xt2, h], dim=1)
            gates = self.lstm_gates(combined)
            gates = self.norm_gates(gates)
            i, f, o, g = torch.split(gates, self.width, dim=1)
            
            new_c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            new_c = self.norm_c(new_c)
            new_h = torch.sigmoid(o) * torch.tanh(new_c)
            new_h = self.norm_h(new_h)
            
            c = new_c
            h = new_h
            
            out = self.q(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            
            if t > 0:
                out = out + preds[-1].squeeze(1) * 0.1
            
            preds.append(out.unsqueeze(1))
            
            if t < T - 1:
                if self.training and np.random.random() < teacher_forcing_ratio:
                    current_input_u = u_seq[:, t+1]
                else:
                    current_input_u = out.detach()
        
        return torch.cat(preds, dim=1)


# ==========================================
# 6. 数据集
# ==========================================
class CFDSeriesDataset2D(Dataset):
    def __init__(self, data_dir, seq_len=10, train=True):
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])
        self.seq_len = seq_len
        self.train = train
        self.samples = []
        
        for f_idx, f_path in enumerate(self.files):
            try:
                data = np.load(f_path, allow_pickle=True)
                t_total = data['u_norm'].shape[0]
                if t_total >= self.seq_len + 1:
                    if self.train:
                        stride = 3
                        for t_start in range(0, t_total - self.seq_len, stride):
                            self.samples.append((f_idx, t_start))
                    else:
                        self.samples.append((f_idx, 0))
            except Exception as e:
                print(f"Error loading {f_path}: {e}")
                
        print(f"{'Train' if train else 'Test'} samples: {len(self.samples)}")
                
    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        f_idx, t_start = self.samples[idx]
        data = np.load(self.files[f_idx], allow_pickle=True)
        
        u_raw = data['u_norm'][t_start : t_start + self.seq_len + 1].copy()
        u_tensor = torch.from_numpy(u_raw).float().permute(0, 3, 1, 2)
        
        sdf_tensor = torch.from_numpy(data['sdf'].copy()).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(data['mask'].copy()).float().unsqueeze(0)
        
        return sdf_tensor, u_tensor, mask_tensor


# ==========================================
# 7. 训练函数（改进版）
# ==========================================
def train_enhanced():
    full_dataset = CFDSeriesDataset2D(DATA_DIR, seq_len=TRAIN_SEQ_LEN, train=True)
    
    if len(full_dataset) == 0:
        print("Error: No data found!")
        return
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = EnhancedFNOLSTM2d(modes1=MODES1, modes2=MODES2, width=WIDTH).to(device)
    physics_loss_fn = ImprovedPhysicsLoss(
        dx=1.0/256, 
        dy=1.0/256, 
        tolerance=PHYSICS_TOLERANCE
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=40, min_lr=1e-6
    )
    
    history = defaultdict(list)
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 200
    
    # 物理权重调度策略
    def get_physics_weight(epoch):
        """渐进式增加物理权重"""
        if epoch < PHYSICS_START_EPOCH:
            return 0.0
        elif epoch < PHYSICS_START_EPOCH + 50:
            # 线性增加
            progress = (epoch - PHYSICS_START_EPOCH) / 50
            return PHYSICS_WEIGHT_FIXED * progress
        else:
            return PHYSICS_WEIGHT_FIXED
    
    # 边界权重
    def get_boundary_weight(epoch):
        return get_physics_weight(epoch) * 0.1
    
    print("\n" + "="*70)
    print("IMPROVED TRAINING WITH FIXED PHYSICS WEIGHTS")
    print(f"Physics starts at epoch: {PHYSICS_START_EPOCH}")
    print(f"Target physics weight: {PHYSICS_WEIGHT_FIXED}")
    print(f"Physics tolerance: {PHYSICS_TOLERANCE}")
    print("="*70 + "\n")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_data_loss = 0.0
        train_physics_loss = 0.0
        train_total_loss = 0.0
        
        current_physics_weight = get_physics_weight(epoch)
        current_boundary_weight = get_boundary_weight(epoch)
        
        for sdf, u_series, mask in train_loader:
            sdf = sdf.to(device)
            u_series = u_series.to(device)
            mask = mask.to(device)
            
            u_input = u_series[:, :-1]
            u_target = u_series[:, 1:]
            
            tf_ratio = max(0.0, 0.5 * (1 - epoch/EPOCHS))
            preds = model(sdf, u_input, teacher_forcing_ratio=tf_ratio)
            
            mask_v = mask.unsqueeze(1)
            data_loss = F.mse_loss(preds * mask_v, u_target * mask_v)
            
            total_loss = data_loss
            
            if current_physics_weight > 0:
                loss_cont, loss_bound = physics_loss_fn(preds, mask)
                physics_loss = loss_cont + loss_bound
                total_loss = data_loss + current_physics_weight * physics_loss
                train_physics_loss += physics_loss.item()
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # 分级梯度裁剪
            if current_physics_weight > 0:
                # 先裁剪数据损失的梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            optimizer.step()
            
            train_data_loss += data_loss.item()
            train_total_loss += total_loss.item()
        
        # 计算平均损失
        avg_train_data_loss = train_data_loss / len(train_loader)
        avg_train_total_loss = train_total_loss / len(train_loader)
        avg_train_physics_loss = train_physics_loss / len(train_loader) if current_physics_weight > 0 else 0.0
        
        # 验证
        model.eval()
        val_data_loss = 0.0
        val_physics_loss = 0.0
        
        with torch.no_grad():
            for sdf, u_series, mask in val_loader:
                sdf = sdf.to(device)
                u_series = u_series.to(device)
                mask = mask.to(device)
                
                u_input = u_series[:, :-1]
                u_target = u_series[:, 1:]
                
                preds = model(sdf, u_input, teacher_forcing_ratio=0.0)
                
                mask_v = mask.unsqueeze(1)
                data_loss = F.mse_loss(preds * mask_v, u_target * mask_v)
                val_data_loss += data_loss.item()
                
                if current_physics_weight > 0:
                    loss_cont, loss_bound = physics_loss_fn(preds, mask)
                    val_physics_loss += (loss_cont.item() + loss_bound.item())
        
        avg_val_data_loss = val_data_loss / len(val_loader)
        avg_val_physics_loss = val_physics_loss / len(val_loader) if current_physics_weight > 0 else 0.0
        
        # 更新学习率
        scheduler.step(avg_val_data_loss)
        
        # 记录历史
        history['train_loss'].append(avg_train_total_loss)
        history['val_loss'].append(avg_val_data_loss)
        history['physics_loss'].append(avg_train_physics_loss)
        history['physics_weight'].append(current_physics_weight)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 打印信息
        current_lr = optimizer.param_groups[0]['lr']
        
        if current_physics_weight > 0:
            phys_str = f"[Phys w={current_physics_weight:.6f}]"
        else:
            phys_str = "[Data only]"
            
        print(f"Epoch {epoch:3d} {phys_str:25s} | "
              f"Train: {avg_train_data_loss:.6f} | "
              f"Val: {avg_val_data_loss:.6f} | "
              f"Phys: {avg_train_physics_loss:.6f} | "
              f"Val_Phys: {avg_val_physics_loss:.6f} | "
              f"LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if avg_val_data_loss < best_val_loss:
            best_val_loss = avg_val_data_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_data_loss,
                'history': dict(history)
            }
            
            torch.save(checkpoint, f"{SAVE_DIR}/best_model.pth")
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
        
        # 定期保存
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_data_loss,
                'history': dict(history)
            }, f"{SAVE_DIR}/checkpoint_epoch_{epoch}.pth")
        
        # 早停
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # 保存最终模型
    torch.save({
        'final_epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': best_val_loss,
        'history': dict(history)
    }, f"{SAVE_DIR}/final_model.pth")
    
    # 绘制训练曲线
    plot_training_history(history, SAVE_DIR)
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final physics weight: {current_physics_weight:.6f}")
    print(f"{'='*70}")


def plot_training_history(history, save_dir):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练和验证损失
    axes[0, 0].plot(history['train_loss'], label='Train Total Loss', alpha=0.7)
    axes[0, 0].plot(history['val_loss'], label='Val Data Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 物理损失和权重
    ax1 = axes[0, 1]
    ax2 = ax1.twinx()
    
    # 过滤掉物理损失为0的epoch
    phys_epochs = [i for i, p in enumerate(history['physics_loss']) if p > 0]
    phys_values = [history['physics_loss'][i] for i in phys_epochs]
    phys_weights = [history['physics_weight'][i] for i in phys_epochs]
    
    if phys_values:
        ax1.plot(phys_epochs, phys_values, 'b-', label='Physics Loss', alpha=0.7)
        ax2.plot(phys_epochs, phys_weights, 'r--', label='Physics Weight', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Physics Loss', color='b')
        ax2.set_ylabel('Physics Weight', color='r')
        ax1.set_title('Physics Loss and Weight')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        ax1.grid(True)
    
    # 学习率
    axes[1, 0].plot(history['lr'], label='Learning Rate', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 验证损失的滑动平均
    if len(history['val_loss']) > 10:
        val_loss_smooth = np.convolve(history['val_loss'], 
                                      np.ones(10)/10, 
                                      mode='valid')
        axes[1, 1].plot(history['val_loss'], alpha=0.3, label='Val Loss (raw)')
        axes[1, 1].plot(range(9, len(history['val_loss'])), 
                       val_loss_smooth, 
                       label='Val Loss (smoothed)', 
                       linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Loss')
        axes[1, 1].set_title('Validation Loss Trend')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_dir}/training_history.png")


if __name__ == "__main__":
    train_enhanced()