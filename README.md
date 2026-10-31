# CIFAR-10 数据增强与正则化对比实验

在同一小型 CNN 上比较数据增强与 BN-Dropout 组合：四种配置、三个随机种子、每次30轮。
全部训练与测试已完成；本文是常见方法的受控比较，不宣称提出新算法。

## 真实测试结果

|配置|测试准确率 均值 ± 样本标准差|相对基线|
|---|---:|---:|
|baseline|74.73% ± 0.35%|0.00 个百分点|
|augment|76.13% ± 1.35%|+1.40 个百分点|
|regularized|77.09% ± 0.67%|+2.36 个百分点|
|combined|76.70% ± 0.51%|+1.97 个百分点|

标准差基于种子42、43、44，不是置信区间。组合策略的均值没有超过仅 BN-Dropout 配置。
每个种子使用同一10000张官方测试集；不是120000张相互独立的测试图像。
原始耗时含暂停或系统负载干扰，不能用来排名纯计算效率。

## 目录

- `reproduction/outputs/cifar10/`：训练、评估、分析、环境创建脚本及版本锁定。
- `artifacts/runs/`：十二组验证最优权重、配置、数据索引、原始日志和正式测试结果。
- `artifacts/analysis/`：统计表、混淆矩阵、学习曲线与输入校验记录。
- `artifacts/figures/`：网络图与训练样例，含可复现抽样索引。
- `SHA256SUMS.json`：公开文件校验清单。

数据集、虚拟环境、缓存和优化器续训检查点不随仓库分发。
`analysis_manifest.json` 保留原训练检查点 `last.pt` 的哈希作为审计记录；
公开范围只提供 `best.pt`，不包含这些优化器检查点。包内可获取文件以根目录校验清单为准。

## 环境与复现

已验证环境：Windows 11、Python3.15.2、PyTorch2.14.1+cpu、torchvision0.29.1+cpu，CPU两线程。
请使用 Python3.15。以下从仓库根目录执行：

```powershell
cd reproduction
py -3.15 outputs/cifar10/setup_environment.py
.\work\venv315\Scripts\python.exe outputs/cifar10/run_experiments.py
.\work\venv315\Scripts\python.exe outputs/cifar10/analyze_results.py
```

环境安装使用 PyTorch 官方 CPU 源和 PyPI。脚本只创建项目内的 `work/venv315`，不会覆盖已有环境。
首次训练会通过 torchvision 下载并校验官方 CIFAR-10，数据存到 `reproduction/work/cifar-data`。
归档 MD5 为 `c58f30108f718f92721af3b95e74349a`。所有组使用固定45000/5000训练验证划分。
批处理先完成所有训练，再计算正式测试结果；代码哈希和协议固定，中断后从完整轮次续训。
不同硬件或软件环境不保证逐位一致。

## 只核验提供的模型权重

在 `reproduction` 目录中：

```powershell
.\work\venv315\Scripts\python.exe outputs/cifar10/evaluate_weights.py --runs-dir ../artifacts/runs --variant regularized --seed 42
```

`--variant` 可取 baseline、augment、regularized、combined；`--seed` 可取42、43、44。
`best.pt` 是验证准确率最高轮次的纯状态字典，采用 `weights_only=True` 加载，不需要重新训练。
本地已独立核验 regularized/42 的权重，得到与存档完全相同的准确率和混淆矩阵。
新训练结果与提供的存档分开存放，避免覆盖真实实验记录。

## Windows 运行库说明

原电脑的 Python 安装目录包含较旧的 `msvcp140.dll`，导致导入 PyTorch 失败。
环境脚本在新建虚拟环境内预加载系统运行库，不删除或修改 Windows 系统文件。
如本机系统运行库缺失，需先修复相应运行库；不要复制来源不明的 DLL。

## 解释范围

BN 与 Dropout 作为整体因素，不能单独归因；三种子、单一数据集和固定训练预算不支持普遍优势结论。
训练模式指标含增强或 Dropout，不能直接作为统一评估下的泛化差距。
资料出处见 `reproduction/outputs/cifar10/references.bib`。
