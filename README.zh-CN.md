<p align="center">
  <img src="assets/brand/cowmata-logo.svg" width="280" alt="COWMATA">
</p>

<h1 align="center">CowTail-4</h1>

<p align="center"><b>论文中十头可用奶牛的尾根惯性行为数据集。</b></p>

<p align="center">
  <a href="README.md"><b>English</b></a>
  &nbsp;·&nbsp;
  <a href="README.zh-CN.md"><b>简体中文</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/zxq309/COWMATA-Pro"><b>COWMATA Pro</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/jiaosir-cow/cowtail/issues"><b>问题</b></a>
</p>

<p align="center">
  <img alt="data licence" src="https://img.shields.io/badge/data-CC%20BY%204.0-2ea44f">
  <img alt="code licence" src="https://img.shields.io/badge/code-MIT-1f6feb">
  <img alt="cows" src="https://img.shields.io/badge/cows-10-555555">
  <img alt="sessions" src="https://img.shields.io/badge/sessions-31-555555">
  <img alt="sampling" src="https://img.shields.io/badge/sampling-50%20Hz-555555">
</p>

这是与 [COWMATA Pro](https://github.com/zxq309/COWMATA-Pro) 配套的公开数据仓库。

**Yang, Zhanhai · Jiao, Tengyu · Zhang, Jianqi · Zhang, Xiangqing · Zhang, Yalong · Jiang, Qinglin · Zhang, Yihui**

## 概览

论文使用十头可用牛。cowA–cowD 是报告中的测试牛：31 个会话、12,736 个互不重叠的 5.12 s 窗口，附九轴原始流。train1–train6 只作额外训练：101 个区间、879 个窗口，见 [`01_data/additional_training/`](01_data/additional_training/README.md)。名册见 [`01_data/study_roles.csv`](01_data/study_roles.csv)。

## 读取

在仓库根目录执行。依赖见 [`requirements.txt`](requirements.txt)（`numpy`、`pandas`）。

```bash
pip install -r requirements.txt
python code/load_example.py            # 第一条行走窗口
python code/load_example.py cowA 100   # 牛编号，行号
```

```python
import numpy as np
import pandas as pd

X = np.load("01_data/acceleration.npy")   # (12736, 3, 256)，报告测试牛，逐窗口归一化
meta = pd.read_csv("01_data/windows.csv", encoding="utf-8-sig")
row = meta.iloc[0]
approx_g = X[int(row.row_index)] * float(row.scale_g)
```

`row_index` 是 `acceleration.npy` 的行号。乘同一行的 `scale_g` 得到近似 *g*。报告测试数组的顺序：cowA 0–6341，cowC 6342–6669，cowD 6670–7223，cowB 7224–12735。单牛数组在 `01_data/<cow>/acceleration.npy`。这四头牛的九轴原始流在 `04_raw_sensors_labels/`，用 `session_id` 对应。

## 论文名册

| 角色 | 动物 |
|---|---|
| 报告测试，M1 | cowA、cowC、cowD |
| 报告测试，M2 | cowB |
| 仅额外训练 | train1–train6 |

M1 测试 cowA、cowC、cowD。M2 测试 cowB。每头报告测试牛都不进入评价它的那个模型的训练数据。train1–train6 按窗口数排序，只是发布用编号。论文不写出这六头的名字。它们不是 `05_six_cows_full` 里的 cowA–cowF。

## 报告测试牛

| | |
|---|---|
| 动物 | cowA–cowD |
| 设备 | 2 台尾根记录仪，在动物之间轮换 |
| 会话 | 31，一天，一个牧场 |
| 通道 | 九轴：加速度、角速度、磁场 |
| 采样 | 50 Hz |
| 窗口 | 256 点，5.12 s，互不重叠 |
| 窗口数 | 12,736 |
| 类别 | 站立、躺卧、行走、采食 |
| 标注 | 三名标注员，再由一名兽医专业研究生复核 |
| 对齐 | 视频与传感器残余 ≤ 0.1 s |

- **站立** — 四肢直立，含站立反刍。
- **躺卧** — 胸骨或侧卧。
- **行走** — 向前移动（198 个窗口）。
- **采食** — 在料槽采食。反刍、舔舐和单纯的头部动作不算采食。

cowD 没有采食标签（`01_data/four_cow_label_and_window_counts.csv` 中 `FEEDING` = 0）。站立与采食重叠时，两个区间都保留。四分类表每个窗口只用一条标签。视频不公开。公开编号为 `DEVICE_NN`、`SESSION_NN` 和相对毫秒。列定义见 [`docs/data_dictionary.md`](docs/data_dictionary.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `01_data/` | 报告测试牛、`additional_training/`、`study_roles.csv` |
| `02_results/` | 归档指标 |
| `03_models_predictions/` | 检查点与逐窗口预测 |
| `04_raw_sensors_labels/` | 报告测试牛的九轴原始流 |
| `05_six_cows_full/` | 较早的导出，不是这十头牛的名册 |
| `code/` | 读取示例与清单生成 |
| `FILE_MANIFEST.json` | 每个文件的大小与 SHA-256 |

四牛 cowA 即六牛 cowA。四牛 cowB 是六牛 cowB 的一个切片。四牛 cowC 是六牛 cowD。四牛 cowD 是六牛 cowE。那六个显示名不是 train1–train6。数量和对照表见 [`05_six_cows_full/README.md`](05_six_cows_full/README.md)。

用 `python verify_and_recompute.py` 核对该快照。

## 引用

```bibtex
@dataset{cowtail4_2026,
  title  = {CowTail-4: A Tail-Mounted Nine-Axis IMU Behaviour Dataset for Dairy Cattle},
  author = {Yang, Zhanhai and Jiao, Tengyu and Zhang, Jianqi and Zhang, Xiangqing and
            Zhang, Yalong and Jiang, Qinglin and Zhang, Yihui},
  year   = {2026},
  url    = {https://github.com/jiaosir-cow/cowtail},
  note   = {Data licence: CC BY 4.0; code licence: MIT}
}
```

同一条记录见 [`CITATION.cff`](CITATION.cff)。论文发表后请同时引用论文。标注软件见 [COWMATA Pro](https://github.com/zxq309/COWMATA-Pro)。

## 许可

数据 [CC BY 4.0](LICENSE)。代码 [MIT](LICENSE-CODE)。COWMATA 名称和 [`assets/brand/`](assets/brand/) 中的文件仍是公司品牌资产，见 [`NOTICE`](NOTICE)。

## 致谢

扬州大学实验牧场伦理委员会已批准。仅尾根外挂。十头可用牛在同一牧场；报告测试记录来自一天。标签只写可观察状态。绝对时间、设备 MAC、原始牛号、视频文件名和标注人姓名已移除。

## 联系

张向前（通讯作者）— jiaosirty@stu.yau.edu.cn

问题请提交 GitHub Issue。
