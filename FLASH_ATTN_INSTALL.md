# Flash Attention 2 安装教程

当 `flash_attn` 未安装时，配置已默认使用 `sdpa`，可正常训练。若需启用 FlashAttention2 以加速训练、降低显存，按以下步骤安装。

---

## 方法一：预编译 Wheel（推荐，最快）

### 1. 查看环境信息

```bash
# Python 版本
python -c "import sys; print(sys.version)"

# PyTorch 版本
python -c "import torch; print(torch.__version__)"

# CUDA 版本
python -c "import torch; print(torch.version.cuda)"
nvidia-smi  # 查看驱动支持的 CUDA
```

### 2. 针对当前环境（Python 3.11 + PyTorch 2.6 + CUDA 12.4）

若你的环境为 **Python 3.11**、**PyTorch 2.6.0+cu124**、**CUDA 12.4**、**Linux x86_64**，可直接执行：

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

若上述安装报错（如 ABI 不兼容），可尝试：

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

### 3. 其他环境获取对应 Wheel

访问 [flashattn.dev](https://flashattn.dev/) 或 [flashattn.dev/install](https://flashattn.dev/install)，选择：

- **Platform**: Linux x86_64
- **Python**: 3.10 / 3.11 / 3.12
- **PyTorch**: 2.4 / 2.5 / 2.6 等
- **CUDA**: 11.8 / 12.1 / 12.4 等

复制生成的 `pip install` 命令。

### 4. 安装并验证

```bash
pip install <复制的 wheel URL>
python -c "import flash_attn; print('flash_attn ok')"
```

### 5. 修改配置启用 FlashAttention2

在 `starvla_train_pi.yaml` 中：

```yaml
attn_implementation: flash_attention_2
```

---

## 方法二：从源码编译

适用于没有匹配预编译包的情况。

### 环境要求

- CUDA 12.0+
- PyTorch 2.2+
- GCC、ninja
- 足够内存（建议 16GB+）

### 安装步骤

```bash
# 激活环境
conda activate llava3d_vla_train

# 安装 ninja（若未安装）
pip install ninja packaging

# 编译安装（约 10–30 分钟）
pip install flash-attn --no-build-isolation

# 若内存不足，可限制并行编译
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### 验证

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

---

## 常见问题

### 1. 版本不匹配

确保 PyTorch、CUDA、Python 与 flash-attn 兼容，参考 [flashattn.dev/compatibility](https://flashattn.dev/compatibility)。

### 2. 编译失败

- 检查 CUDA 路径：`echo $CUDA_HOME`
- 使用 `MAX_JOBS=2` 或 `MAX_JOBS=1` 减少并行编译
- 确认 GCC 版本：`gcc --version`（建议 9+）

### 3. 安装后仍报错

- 重启 Python 进程或重新激活 conda 环境
- 确认 `import flash_attn` 无报错

---

## 当前配置说明

- **attn_implementation: sdpa**：无需 flash-attn，可正常训练
- **attn_implementation: flash_attention_2**：需先安装 flash-attn，训练更快、显存更低
