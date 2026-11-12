[English README](../README.md)

# LCGCN

代码和文档采用 MIT 许可证。公开仓库不含第三方数据文件：五个数据集和50份划分的再分发来源授权仍待确认，不能把 GitHub 源码下载当作完整数据包。提供 LCGCN、NativeLCGCN、FastLCGCN 的冻结主表训练配置，不包含基线或历史结果。默认运行会重新训练，不读取预存准确率。

## 1. 解压和环境

**先克隆仓库或解压 GitHub Download ZIP，再安装依赖。无需数据可运行 `python main.py --all --list` 和 `python -m unittest discover -s tests -v`。训练需要已有授权的数据集合；可通过 `--data-dir "数据目录"` 指向旧完整包中的 data，或单独数据包的 LCGCN_data/data。仓库仅提供校验清单 docs/data_manifest.json，不提供公开数据下载。**

解压后在 VS Code 中“文件 → 打开文件夹”，选择直接包含 main.py 的 LCGCN 文件夹：

```text
LCGCN/
  main.py
  README.md
  requirements.txt
  pyproject.toml         标准安装、依赖和命令行配置
  lcgcn/                正式Python包，含models.py、graph_builder.py、data_loader.py
    configs/            450项冻结主表配置
  tests/                无需数据的结构与接口测试
  docs/                 数据说明、统计与来源记录
  data/                 自行提供的授权数据，不在公开仓库内
    manifest.json
    coauthorship/
      cora/             features.pickle、labels.pickle、hypergraph.pickle
        splits/         1.pickle 至 10.pickle
      dblp/             同样包含三个数据文件及 splits/1..10.pickle
    cocitation/
      cora/             同上
      citeseer/         同上
      pubmed/           同上
```

每份划分文件都包含 train/test 节点索引。总计15个数据文件和50个划分文件。

Windows VS Code 安装微软 Python 扩展，打开 LCGCN 文件夹；Ctrl+Shift+P → Python: Select Interpreter → 选择安装依赖的 Python 3.10 解释器。新建终端后先运行 `python -c "import sys; print(sys.executable)"` 确认解释器。不要在 ZIP 中直接运行。

已有环境可直接使用。本地实测 Python 3.10.18、PyTorch 2.7.1+cu128、NumPy 2.2.6、SciPy 1.15.3、scikit-learn 1.7.2，Windows、RTX 5060 Laptop。其他 Python/PyTorch 版本未保证兼容。新环境示例（需预先安装 Python 3.10）：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e .
```

然后在 VS Code 选择 `.venv\Scripts\python.exe`，新开终端。若终端未激活，将后续 `python` 换为 `.\.venv\Scripts\python.exe`；无需修改 PowerShell 执行策略。NVIDIA 用户可将上述 PyTorch 安装行替换为：

```powershell
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

需支持该版本的驱动；CPU 无需显卡。安装来源：[PyTorch 官方历史版本](https://pytorch.org/get-started/previous-versions/)。不需要 torch-scatter、PyG、DGL 或单独编译稀疏扩展。程序启动时检查 PyTorch 稀疏前向/反向运算。

## 2. 准备数据后运行

```powershell
python main.py
```

若数据放在其他位置，使用 `python main.py --data-dir "数据目录"`；默认运行要求项目根目录下已有完整的 data。

也可打开 main.py，点击 VS Code 右上角“运行 Python 文件”。请先选好下述 Python 环境；点击运行采用同样的默认200轮训练。

默认：LCGCN、Cora-CA、split 1、初始化种子 1、真实 200 轮，使用 CUDA（可用时），否则 CPU。通常几十秒至数分钟，取决于硬件；不是完整主表。首次导入 PyTorch 也需要时间。

```powershell
python main.py --check --device cpu
python main.py --smoke --device cpu
python main.py --model native-lcgcn --dataset citeseer --seed 8
python main.py --model fast-lcgcn --dataset pubmed --seed 17 --device cuda
python main.py --all --device cuda
```

`--check` 校验 65 个文件哈希并读取全部 50 份划分；不训练。`--smoke` 真实训练 3 轮，仅验收可运行性。`--all` 串行执行 3 模型 × 5 数据集 × 10 划分 × 3 种子 = 450 次，每次 200 轮；预计数小时，CPU 可更久，应预留数 GB 输出空间。它复现自有模型主表路径，不包括论文基线、消融和其他补充实验。

在任意工作目录可以运行 `python "完整路径/LCGCN/main.py"`。数据默认按 main.py 所在位置寻找，不受终端当前目录影响；自定义位置用 `--data-dir "完整路径/LCGCN/data"`。相对的自定义路径按终端当前目录解析。

## 3. 输出和核验

运行后在终端直接查看结果，无需打开JSON：每项显示模型、数据集、划分、实际初始化种子、轮数、设备、测试准确率（百分比）、测试节点数、训练耗时、重载核验状态和保存位置。训练期间显示第1轮、每20轮及最后一轮的训练损失。

全部任务结束后打印两种明确区分的汇总。第一种是逐次运行描述统计（次数、准确率均值、逐次样本标准差和累计训练时间），该标准差不是论文主表SD。第二种严格采用论文口径：每个划分先平均种子1/8/17，再计算10个划分均值的均值和样本标准差（ddof=1）。只有同一模型、数据集和设备的完整30项200轮结果且无重复时才输出论文统计；不齐全时显示N/A，不用部分运行冒充论文主表。单次训练的逐次SD显示N/A；3轮短跑明确标记SMOKE。程序不读取历史结果，也不自动计算基线配对检验或置信区间。

同样的汇总表保存为 `summary.txt`，详细逐次记录保留在 `summary.json`。`python main.py --check` 也会直接打印数据检查结果和五数据集统计表。

每次运行在 `LCGCN/outputs/train-时间-随机标识/` 新建目录；短跑用 `smoke-` 前缀。每项包含 `history.json`（逐轮训练损失）、`model.pt`、`predictions.npz`、`result.json`（准确率/环境/耗时）、最终图和输入哈希；总索引为 `summary.json`。不会覆盖此前训练。`--output-dir` 可指定输出父目录。

训练后自动重载 checkpoint，核对 logits、预测和重建图。退出码 0 表示本次流程及重载通过；2 表示参数或数据检查失败，或重建数值检查失败（查看终端及 summary）；其他非零码见 traceback。动态图、GPU 与 CPU 浮点差异可能改变排序/图；不能保证跨运行逐位一致或完全复现历史分数。核验失败保留产物调查，不根据准确率择优重跑。

## 4. 冻结协议

Adam，lr 0.01，weight decay 0.0005，dropout 0.5；Citeseer hidden 64，其他 16；200 轮 final checkpoint。种子 1/8/17 对应内部 repeat 9/1/23，图及评估随机流按 repeat 派生。种子来自此前测试表现的探索性选择，并非预先指定。种子 8 部分历史结果复用了早期选参记录；本包按最终冻结参数直接训练，未重复选参。

第一层缓存；LCGCN 第二层使用 ReLU/dropout 后隐藏特征建图，建图分支 detach；评估关闭 dropout 并重建第二层。Native/Fast 两层复用固定图。局部总无向边权为 1，重叠边求和，I 加一次，对称归一化，logits argmax 评估。模块统一命名为 models.py、graph_builder.py、data_loader.py，类名为 LCGCN；名称和数据路径调整不改变算法、数据、划分及训练参数。

根目录打开整个 LCGCN 文件夹，不要只打开 lcgcn 子目录。Pylance 使用项目中的正规包导入，无需设置 extraPaths，也没有关闭缺失导入警告。若 torch 等第三方依赖仍报错，核对 VS Code 选择的解释器与安装依赖的解释器是否一致。

历史核查范围见 [复现说明](REPRODUCIBILITY.md)；未将 3 轮短跑或部分 200 轮运行宣称为 450 次全量复现。依赖安装命令已核对官方来源，本次使用已有环境，未声称在全新 Windows 上重装成功。


## 标准包入口与开发

在项目根目录还可以使用 `python -m lcgcn`；安装 `python -m pip install -e .` 后可使用 `lcgcn` 命令。这三种入口调用同一个训练函数，默认均真实训练200轮。main.py仅保留为VS Code直接运行的简短入口，不含业务逻辑。

源码/可编辑安装默认定位项目内data与outputs。普通wheel安装不携带数据，在任意目录运行时用 `python -m lcgcn --data-dir "完整路径/LCGCN/data" --output-dir "结果目录"`；未指定时读取当前目录的data并写outputs。450项配置随Python包安装。安装命令应在包含pyproject.toml的根目录执行。

推荐Python3.10，本版安装元数据暂限3.10，其他版本尚未验证。开发与CI说明见docs/DEVELOPMENT.md。公开仓库采用 MIT 许可证，Git 默认忽略 data，wheel 也不含数据。旧本地完整 ZIP 保留给已有授权的本地使用，不作为公开附件。第三方数据不随 MIT 重新授权。


本版代码注释、终端提示和主文档统一为英文；本文件仅为中文使用指南。终端的 Paper statistics 采用论文口径，Run summary 为逐次运行描述统计。请在项目根目录执行命令，不要在 docs 目录执行安装命令。
