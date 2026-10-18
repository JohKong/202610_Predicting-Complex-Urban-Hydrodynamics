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
# 1. 全局静态配置参数
# ==========================================
DATA_DIR = 'Dataset'
BASE_SAVE_DIR = 'Results_Ablation2'
Results_DIR = 'Results_Ablation3'
os.makedirs(Results_DIR, exist_ok=True)

BATCH_SIZE = 6
EPOCHS = 800
LR = 1e-3
NX, NY = 256, 256
TRAIN_SEQ_LEN = 10
MODES1, MODES2 = 36, 36
WIDTH = 144
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

# 物理约束参数（使用改进版本）
PHYSICS_START_EPOCH = 1
PHYSICS_WEIGHT_FIXED = 1e-3
PHYSICS_TOLERANCE = 0.05
BOUNDARY_WEIGHT = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ==========================================
# 2. 消融实验配置管理类
# ==========================================
class AblationConfig:
    def __init__(self, exp_name="Full_Model", use_ca=True, use_gn=True, use_physics=True, 
                 use_tf=True, use_smooth=True, use_lstm=True, lstm_layers=1):
        self.exp_name = exp_name
        self.use_ca = use_ca
        self.use_gn = use_gn
        self.use_physics = use_physics
        self.use_tf = use_tf
        self.use_smooth = use_smooth
        self.use_lstm = use_lstm
        self.lstm_layers = lstm_layers

    def __str__(self):
        return (f"Experiment: {self.exp_name}\n"
                f"  -> Channel Attention: {self.use_ca}\n"
                f"  -> Group Normalization: {self.use_gn}\n"
                f"  -> Physics Constraints: {self.use_physics}\n"
                f"  -> Teacher Forcing: {self.use_tf}\n"
                f"  -> Residual Smoothing: {self.use_smooth}\n"
                f"  -> LSTM: {self.use_lstm} (layers={self.lstm_layers})")


# ==========================================
# 3. 频谱卷积层
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

        out_ft = torch.zeros(batchsize, self.out_channels, size_1, size_2 // 2 + 1,
                             dtype=torch.complex64, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(size_1, size_2), norm='ortho')
        return x


# ==========================================
# 4. 通道注意力
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
# 5. 独立的LSTM模块
# ==========================================
class LSTMCell2d(nn.Module):
    """2D卷积LSTM单元，支持GroupNorm"""
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1, use_gn=True):
        super(LSTMCell2d, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.use_gn = use_gn
        
        # 门控卷积层
        self.gates = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 4, 
                               kernel_size=kernel_size, padding=padding)
        
        # GroupNorm (如果启用)
        if use_gn:
            num_groups = min(8, hidden_channels)
            self.norm_gates = nn.GroupNorm(num_groups, hidden_channels * 4)
            self.norm_c = nn.GroupNorm(num_groups, hidden_channels)
            self.norm_h = nn.GroupNorm(num_groups, hidden_channels)
        else:
            self.norm_gates = nn.Identity()
            self.norm_c = nn.Identity()
            self.norm_h = nn.Identity()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='sigmoid')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, h, c):
        """
        Args:
            x: 输入特征 [B, C_in, H, W]
            h: 隐藏状态 [B, C_hidden, H, W]
            c: 细胞状态 [B, C_hidden, H, W]
        Returns:
            new_h, new_c
        """
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        gates = self.norm_gates(gates)
        
        # 分割为输入门、遗忘门、输出门、候选细胞
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        
        # 状态更新
        new_c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        new_c = self.norm_c(new_c)
        new_h = torch.sigmoid(o) * torch.tanh(new_c)
        new_h = self.norm_h(new_h)
        
        return new_h, new_c


class LSTMStack2d(nn.Module):
    """多层2D卷积LSTM堆叠"""
    def __init__(self, input_channels, hidden_channels, num_layers=1, use_gn=True):
        super(LSTMStack2d, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            self.cells.append(LSTMCell2d(in_ch, hidden_channels, use_gn=use_gn))
    
    def forward(self, x, h_states=None, c_states=None):
        """
        Args:
            x: 输入 [B, C_in, H, W] (单步)
            h_states: 各层隐藏状态列表
            c_states: 各层细胞状态列表
        Returns:
            output: 输出 [B, C_hidden, H, W]
            h_states, c_states: 更新后的状态
        """
        batch_size = x.shape[0]
        
        # 初始化状态
        if h_states is None:
            h_states = [torch.zeros(batch_size, self.hidden_channels, x.shape[2], x.shape[3], 
                                   device=x.device) for _ in range(self.num_layers)]
        if c_states is None:
            c_states = [torch.zeros(batch_size, self.hidden_channels, x.shape[2], x.shape[3],
                                   device=x.device) for _ in range(self.num_layers)]
        
        layer_input = x
        new_h_states = []
        new_c_states = []
        
        for layer_idx, cell in enumerate(self.cells):
            h, c = cell(layer_input, h_states[layer_idx], c_states[layer_idx])
            layer_input = h  # 下一层的输入是当前层的输出
            new_h_states.append(h)
            new_c_states.append(c)
        
        return layer_input, new_h_states, new_c_states


# ==========================================
# 6. 改进的物理损失（缩放版本）
# ==========================================
class ImprovedPhysicsLoss(nn.Module):
    """改进的物理损失，加入自适应缩放"""
    def __init__(self, dx=1.0 / 256, dy=1.0 / 256, tolerance=0.05):
        super(ImprovedPhysicsLoss, self).__init__()
        self.dx = dx
        self.dy = dy
        self.tolerance = tolerance
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

            du_dy, du_dx = torch.gradient(u_t, spacing=(self.dy, self.dx), dim=(1, 2))
            dv_dy, dv_dx = torch.gradient(v_t, spacing=(self.dy, self.dx), dim=(1, 2))

            divergence = du_dx + dv_dy

            mask_2d = mask.squeeze(1)

            div_rms = torch.sqrt(torch.mean((divergence * mask_2d) ** 2) + 1e-8)
            divergence_norm = divergence / (div_rms + 1e-8)

            loss_cont = F.smooth_l1_loss(
                divergence_norm * mask_2d,
                torch.zeros_like(divergence_norm),
                beta=self.tolerance
            )

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
# 7. 主模型 (支持消融开关)
# ==========================================
class EnhancedFNOLSTM2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=64, config=None):
        super(EnhancedFNOLSTM2d, self).__init__()
        self.width = width
        self.config = config if config is not None else AblationConfig()

        # 1. 输入映射
        self.p = nn.Sequential(
            nn.Linear(3, width),
            nn.GELU(),
            nn.Linear(width, width)
        )

        # 2. FNO 块
        self.fno_conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.fno_conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)

        # 归一化层
        num_groups = 8
        if self.config.use_gn:
            self.norm1 = nn.GroupNorm(num_groups, width)
            self.norm2 = nn.GroupNorm(num_groups, width)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # 通道注意力
        if self.config.use_ca:
            self.ca1 = ChannelAttention(width)
            self.ca2 = ChannelAttention(width)

        # 3. LSTM 模块 (根据配置决定)
        if self.config.use_lstm:
            self.lstm = LSTMStack2d(
                input_channels=width,
                hidden_channels=width,
                num_layers=self.config.lstm_layers,
                use_gn=self.config.use_gn
            )
        else:
            # 如果不使用LSTM，用一个简单的3x3卷积替代
            self.feedforward = nn.Sequential(
                nn.Conv2d(width, width, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(width, width, kernel_size=3, padding=1)
            )
            if self.config.use_gn:
                num_groups = min(8, width)
                self.feedforward_norm = nn.GroupNorm(num_groups, width)

        # 4. 输出头
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

        preds = []
        current_input_u = u_seq[:, 0].clone()

        # 初始化LSTM状态（如果使用）
        if self.config.use_lstm:
            h_states = [torch.zeros(batch_size, self.width, y, x, device=u_seq.device) 
                       for _ in range(self.config.lstm_layers)]
            c_states = [torch.zeros(batch_size, self.width, y, x, device=u_seq.device) 
                       for _ in range(self.config.lstm_layers)]
        else:
            h_states = None
            c_states = None

        for t in range(T):
            # 拼接输入
            xt = torch.cat([current_input_u, sdf], dim=1)
            xt = xt.permute(0, 2, 3, 1)
            xt = self.p(xt).permute(0, 3, 1, 2)

            # FNO 块 1
            xt1 = self.fno_conv1(xt) + self.w1(xt)
            xt1 = self.norm1(xt1)
            xt1 = F.gelu(xt1)
            if self.config.use_ca:
                xt1 = self.ca1(xt1)

            # FNO 块 2
            xt2 = self.fno_conv2(xt1) + self.w2(xt1)
            xt2 = self.norm2(xt2)
            xt2 = F.gelu(xt2)
            if self.config.use_ca:
                xt2 = self.ca2(xt2)

            # 时序处理：LSTM 或 前馈网络
            if self.config.use_lstm:
                temporal_out, h_states, c_states = self.lstm(xt2, h_states, c_states)
            else:
                # 简单前馈处理（无时序记忆）
                temporal_out = self.feedforward(xt2)
                if self.config.use_gn:
                    temporal_out = self.feedforward_norm(temporal_out)
                temporal_out = F.gelu(temporal_out)

            # 输出预测
            out = self.q(temporal_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            # 残差平滑
            if self.config.use_smooth and t > 0:
                out = out + preds[-1].squeeze(1) * 0.1

            preds.append(out.unsqueeze(1))

            # Teacher Forcing 逻辑
            if t < T - 1:
                if self.training and np.random.random() < teacher_forcing_ratio:
                    current_input_u = u_seq[:, t + 1]
                else:
                    current_input_u = out.detach()

        return torch.cat(preds, dim=1)


# ==========================================
# 8. 数据集
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

        u_raw = data['u_norm'][t_start: t_start + self.seq_len + 1].copy()
        u_tensor = torch.from_numpy(u_raw).float().permute(0, 3, 1, 2)

        sdf_tensor = torch.from_numpy(data['sdf'].copy()).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(data['mask'].copy()).float().unsqueeze(0)

        return sdf_tensor, u_tensor, mask_tensor


# ==========================================
# 9. 核心消融训练引擎
# ==========================================
def run_ablation_experiment(config: AblationConfig):
    exp_save_dir = os.path.join(BASE_SAVE_DIR, config.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"STARTING EXPERIMENT: {config.exp_name}")
    print(config)
    print("=" * 70 + "\n")

    full_dataset = CFDSeriesDataset2D(DATA_DIR, seq_len=TRAIN_SEQ_LEN, train=True)
    if len(full_dataset) == 0:
        print(f"[{config.exp_name}] Error: No data found!")
        return float('inf')

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = EnhancedFNOLSTM2d(modes1=MODES1, modes2=MODES2, width=WIDTH, config=config).to(device)
    physics_loss_fn = ImprovedPhysicsLoss(dx=1.0 / 256, dy=1.0 / 256, tolerance=PHYSICS_TOLERANCE)

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
        if not config.use_physics:
            return 0.0
        if epoch < PHYSICS_START_EPOCH:
            return 0.0
        elif epoch < PHYSICS_START_EPOCH + 50:
            progress = (epoch - PHYSICS_START_EPOCH) / 50
            return PHYSICS_WEIGHT_FIXED * progress
        else:
            return PHYSICS_WEIGHT_FIXED

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_data_loss = 0.0
        train_physics_loss = 0.0
        train_total_loss = 0.0

        current_physics_weight = get_physics_weight(epoch)

        for sdf, u_series, mask in train_loader:
            sdf = sdf.to(device)
            u_series = u_series.to(device)
            mask = mask.to(device)

            u_input = u_series[:, :-1]
            u_target = u_series[:, 1:]

            # Teacher Forcing 开关
            if config.use_tf:
                tf_ratio = max(0.0, 0.5 * (1 - epoch / EPOCHS))
            else:
                tf_ratio = 0.0

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_data_loss += data_loss.item()
            train_total_loss += total_loss.item()

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

        if epoch % 10 == 0 or epoch == 1:
            print(f"[{config.exp_name}] Epoch {epoch:3d} {phys_str:25s} | "
                  f"Train: {avg_train_data_loss:.6f} | "
                  f"Val: {avg_val_data_loss:.6f} | "
                  f"Phys: {avg_train_physics_loss:.6f} | "
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
                'history': dict(history),
                'config': config.__dict__
            }
            torch.save(checkpoint, f"{exp_save_dir}/best_model.pth")
        else:
            patience_counter += 1
        '''''
        # 定期保存
        if epoch % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_data_loss,
                'history': dict(history)
            }, f"{exp_save_dir}/checkpoint_epoch_{epoch}.pth")
        '''''
        # 早停
        if patience_counter >= max_patience:
            print(f"[{config.exp_name}] Early stopping at epoch {epoch}")
            break
    '''
    # 保存最终模型
    torch.save({
        'final_epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': best_val_loss,
        'history': dict(history),
        'config': config.__dict__
    }, f"{exp_save_dir}/final_model.pth")
    '''
    # 绘制训练曲线
    plot_training_history(history, exp_save_dir, config.exp_name)

    print(f"\n>>> Finished Experiment: {config.exp_name} | Best Val Loss: {best_val_loss:.6f}\n")

    return best_val_loss


def plot_training_history(history, save_dir, exp_name):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 训练和验证损失
    axes[0, 0].plot(history['train_loss'], label='Train Total Loss', alpha=0.7)
    axes[0, 0].plot(history['val_loss'], label='Val Data Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'{exp_name} - Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 物理损失和权重
    ax1 = axes[0, 1]
    ax2 = ax1.twinx()

    phys_epochs = [i for i, p in enumerate(history['physics_loss']) if p > 0]
    phys_values = [history['physics_loss'][i] for i in phys_epochs]
    phys_weights = [history['physics_weight'][i] for i in phys_epochs]

    if phys_values:
        ax1.plot(phys_epochs, phys_values, 'b-', label='Physics Loss', alpha=0.7)
        ax2.plot(phys_epochs, phys_weights, 'r--', label='Physics Weight', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Physics Loss', color='b')
        ax2.set_ylabel('Physics Weight', color='r')
        ax1.set_title(f'{exp_name} - Physics Loss and Weight')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        ax1.grid(True)

    # 学习率
    axes[1, 0].plot(history['lr'], label='Learning Rate', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title(f'{exp_name} - Learning Rate Schedule')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 验证损失的滑动平均
    if len(history['val_loss']) > 10:
        val_loss_smooth = np.convolve(history['val_loss'],
                                      np.ones(10) / 10,
                                      mode='valid')
        axes[1, 1].plot(history['val_loss'], alpha=0.3, label='Val Loss (raw)')
        axes[1, 1].plot(range(9, len(history['val_loss'])),
                        val_loss_smooth,
                        label='Val Loss (smoothed)',
                        linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Loss')
        axes[1, 1].set_title(f'{exp_name} - Validation Loss Trend')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_dir}/training_history.png")


def plot_enhanced_comparison(all_results, save_dir):
    """绘制包含LSTM信息的增强对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    experiments = list(all_results.keys())
    best_losses = [all_results[exp]['best_val_loss'] for exp in experiments]
    final_losses = [all_results[exp]['final_val_loss'] for exp in experiments]
    
    # 获取LSTM信息用于着色
    use_lstm = [all_results[exp]['config']['use_lstm'] for exp in experiments]
    lstm_layers = [all_results[exp]['config']['lstm_layers'] for exp in experiments]
    
    x = np.arange(len(experiments))
    width = 0.35
    
    # 子图1: 损失对比
    axes[0, 0].bar(x - width/2, best_losses, width, label='Best Validation Loss', color='steelblue')
    axes[0, 0].bar(x + width/2, final_losses, width, label='Final Validation Loss', color='coral')
    axes[0, 0].set_xlabel('Experiments')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Comparison Across Ablation Experiments')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(experiments, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 相对改进（按LSTM分组着色）
    full_model_loss = all_results.get('1_Full_Model', {}).get('best_val_loss', 1.0)
    improvements = [(full_model_loss - best_losses[i]) / full_model_loss * 100 
                    for i in range(len(experiments))]
    
    colors = ['#2ecc71' if use_lstm[i] else '#e74c3c' for i in range(len(experiments))]
    axes[0, 1].bar(x, improvements, color=colors, alpha=0.7)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].set_xlabel('Experiments')
    axes[0, 1].set_ylabel('Relative Improvement (%)')
    axes[0, 1].set_title('Relative Improvement vs Full Model (Green=LSTM, Red=No LSTM)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(experiments, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: LSTM层数影响
    lstm_exps = [i for i, exp in enumerate(experiments) if use_lstm[i]]
    lstm_names = [experiments[i] for i in lstm_exps]
    lstm_losses = [best_losses[i] for i in lstm_exps]
    lstm_layers_val = [lstm_layers[i] for i in lstm_exps]
    
    if lstm_exps:
        axes[1, 0].scatter(lstm_layers_val, lstm_losses, s=100, c='blue', alpha=0.6)
        for i, name in enumerate(lstm_names):
            axes[1, 0].annotate(name.split('_')[0], (lstm_layers_val[i], lstm_losses[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 0].set_xlabel('Number of LSTM Layers')
        axes[1, 0].set_ylabel('Best Validation Loss')
        axes[1, 0].set_title('Impact of LSTM Layer Count on Performance')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(range(1, max(lstm_layers_val) + 1))
    
    # 子图4: LSTM vs No LSTM 对比
    lstm_best = [best_losses[i] for i in range(len(experiments)) if use_lstm[i]]
    no_lstm_best = [best_losses[i] for i in range(len(experiments)) if not use_lstm[i]]
    
    data_to_plot = [lstm_best] if not no_lstm_best else [lstm_best, no_lstm_best]
    labels = ['With LSTM'] if not no_lstm_best else ['With LSTM', 'Without LSTM']
    
    bp = axes[1, 1].boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors_box = ['#3498db'] if len(data_to_plot) == 1 else ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 1].set_ylabel('Best Validation Loss')
    axes[1, 1].set_title('Performance Distribution: LSTM vs No LSTM')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/enhanced_comparison_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Enhanced comparison plot saved to {save_dir}/enhanced_comparison_summary.png")


# ==========================================
# 10. 一键运行多组实验
# ==========================================
if __name__ == "__main__":
    # 定义所有要跑的消融对照组
    experiments = [
        # 1. 完整增强模型 (基线)
        AblationConfig(exp_name="1_Full_Model", use_ca=True, use_gn=True, use_physics=True, 
                       use_tf=True, use_smooth=True, use_lstm=True, lstm_layers=1),
        
        # 2. 消融实验：去掉通道注意力机制
        #AblationConfig(exp_name="2_wo_ChannelAttention", use_ca=False, use_gn=True, use_physics=True, 
        #               use_tf=True, use_smooth=True, use_lstm=True, lstm_layers=1),
        
        # 3. 消融实验：去掉组归一化
        #AblationConfig(exp_name="3_wo_GroupNorm", use_ca=True, use_gn=False, use_physics=True, 
        #               use_tf=True, use_smooth=True, use_lstm=True, lstm_layers=1),
        
        # 4. 消融实验：去掉物理损失
        #AblationConfig(exp_name="4_wo_PhysicsLoss", use_ca=True, use_gn=True, use_physics=False, 
        #               use_tf=True, use_smooth=True, use_lstm=True, lstm_layers=1),
        
        # 5. 消融实验：去掉 Teacher Forcing
        #AblationConfig(exp_name="5_wo_TeacherForcing", use_ca=True, use_gn=True, use_physics=True, 
        #               use_tf=False, use_smooth=True, use_lstm=True, lstm_layers=1),
        
        # 6. 消融实验：去掉残差平滑
        #AblationConfig(exp_name="6_wo_Smoothing", use_ca=True, use_gn=True, use_physics=True, 
        #               use_tf=True, use_smooth=False, use_lstm=True, lstm_layers=1),
        
        # 7. 新增：去掉LSTM
        AblationConfig(exp_name="7_wo_LSTM", use_ca=True, use_gn=True, use_physics=True, 
                       use_tf=True, use_smooth=True, use_lstm=False, lstm_layers=1),
        
        # 8. 新增：双层LSTM
        AblationConfig(exp_name="8_LSTM_2layers", use_ca=True, use_gn=True, use_physics=True, 
                       use_tf=True, use_smooth=True, use_lstm=True, lstm_layers=2),
        
        # 9. 新增：三层LSTM
        AblationConfig(exp_name="9_LSTM_3layers", use_ca=True, use_gn=True, use_physics=True, 
                       use_tf=True, use_smooth=True, use_lstm=True, lstm_layers=3),
    ]

    all_results = {}

    # 依次串行执行
    for exp_config in experiments:
        best_loss = run_ablation_experiment(exp_config)

        # 读取最终验证损失
        exp_save_dir = os.path.join(BASE_SAVE_DIR, exp_config.exp_name)
        final_model_path = f"{exp_save_dir}/final_model.pth"
        final_val_loss = best_loss
        if os.path.exists(final_model_path):
            checkpoint = torch.load(final_model_path, map_location='cpu')
            final_val_loss = checkpoint.get('val_loss', best_loss)

        all_results[exp_config.exp_name] = {
            'best_val_loss': best_loss,
            'final_val_loss': final_val_loss,
            'config': exp_config.__dict__
        }

    # 打印汇总表格
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY (WITH LSTM VARIANTS)")
    print("=" * 70)
    print(f"{'Experiment':<30} {'Best Val Loss':<15} {'Final Val Loss':<15} {'LSTM':<12}")
    print("-" * 70)
    for exp_name, results in all_results.items():
        lstm_info = f"{results['config']['use_lstm']}({results['config']['lstm_layers']})"
        print(f"{exp_name:<30} {results['best_val_loss']:<15.6f} {results['final_val_loss']:<15.6f} {lstm_info:<12}")
    print("=" * 70)

    # 绘制对比总结图（增强版）
    plot_enhanced_comparison(all_results, Results_DIR)

    print(f"\nAll ablation experiments completed! Results saved to {Results_DIR}/")