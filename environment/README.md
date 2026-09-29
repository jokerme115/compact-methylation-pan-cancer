# environment/ — 环境记录

## 现有文件

| 文件 | 内容 | 状态 |
|---|---|---|
| `python_version.txt` | `Python 3.12.6`（2026-07-10 环境检查时记录） | 记录值，非锁定 |
| `requirements.txt` | 8 个顶层包名，**无版本号** | ⚠️ 宽约束，非锁定 |
| `environment_notes.md` | PyTorch wheel 选择说明；声明 `.venv_pathmethnet` 未随包交付 | 说明 |

## ⚠️ 已知缺口：依赖未锁定

`requirements.txt` 当前只有包名：

```
numpy  pandas  scipy  scikit-learn  matplotlib  joblib  methylprep  torch
```

这把复现风险留给了下游：同一份代码在不同时间安装会拿到不同版本。公开发表仓库（npj Precision Oncology 的 Code availability）通常要求可复现的环境。

## 如何补齐（在持有 `.venv_pathmethnet` 的机器上执行）

```bash
# 激活项目环境后
python -m pip freeze > environment/pip_freeze.txt

# 若用 conda
conda env export --no-builds > environment/environment.yml

# 记录解释器与关键库的实际版本
python -c "import sys, numpy, pandas, sklearn, torch; \
print(sys.version); print('numpy', numpy.__version__); print('pandas', pandas.__version__); \
print('scikit-learn', sklearn.__version__); print('torch', torch.__version__)" \
  > environment/session_info.txt
```

补齐后请更新 `requirements.txt` 为带版本约束的形式，或在本文件注明以 `pip_freeze.txt` 为准。

## 为什么现在不能补

本机没有项目运行环境。`environment_notes.md` 明确说明 `.venv_pathmethnet` 是机器相关环境、未复制到交付包。**不得凭猜测填写版本号**——错误的锁定比没有锁定更有害。

## 依赖用途

| 包 | 用途 |
|---|---|
| numpy / pandas / scipy | 矩阵与表格运算 |
| scikit-learn | LR panel、ML baselines |
| matplotlib | 稿件图表 |
| joblib | 模型持久化 |
| methylprep | GEO IDAT 预处理（GSE105260 custom IDAT diagnostic） |
| torch | PathMethNet 层级解释模型 |

外部验证数据来自 GEO（GSE56044 / GSE48684 / GSE53051 / GSE105260），TCGA 数据来自 GDC，见根目录 `DATA_SOURCES.md`。
