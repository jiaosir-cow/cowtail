<p align="center">
  <img src="assets/brand/cowmata-logo.svg" width="280" alt="COWMATA">
</p>

<h1 align="center">CowTail-4</h1>

<p align="center"><b>四头奶牛的尾根九轴惯性行为数据集。</b></p>

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
  <img alt="windows" src="https://img.shields.io/badge/windows-12%2C736-555555">
  <img alt="sessions" src="https://img.shields.io/badge/sessions-31-555555">
  <img alt="sampling" src="https://img.shields.io/badge/sampling-50%20Hz-555555">
</p>

这是与 [COWMATA Pro](https://github.com/zxq309/COWMATA-Pro) 配套的公开数据仓库。

**Yang, Zhanhai · Jiao, Tengyu · Zhang, Jianqi · Zhang, Xiangqing · Zhang, Yalong · Jiang, Qinglin · Zhang, Yihui**

## 概览

31 个会话、12,736 个互不重叠的 5.12 s 窗口，单一牧场，一天。尾根传感器为九轴，采样率 50 Hz，每窗 256 点。类别为站立、躺卧、行走、采食。仓库里有九轴原始流、参考区间和模型输入数组。视频不公开。

## 仓库里有什么

- **窗口数组** — `01_data/acceleration.npy`，形状 `(12736, 3, 256)`，逐窗口归一化。
- **窗口表** — `01_data/windows.csv`。`row_index` 对应数组行号，乘同一行的 `scale_g` 得到近似 *g*。
- **九轴原始流** — `04_raw_sensors_labels/`，用 `session_id` 与窗口表对应。
- **六牛现场包** — `05_six_cows_full/`，与上面的四牛命名不是同一套。
- **校验** — `FILE_MANIFEST.json` 给出每个文件的大小和 SHA-256。`python verify_and_recompute.py` 核对快照。

## 数据

| | |
|---|---|
| 动物 | cowA–cowD |
| 设备 | 2 个尾根记录仪，在个体间轮换 |
| 会话 | 31 |
| 通道 | 加速度、角速度、磁场 |
| 采样 | 50 Hz |
| 窗口 | 256 点，5.12 s，互不重叠 |
| 窗口数 | 12,736 |
| 类别 | 站立、躺卧、行走、采食 |
| 标签 | 三名标注员，再经兽医专业研究生复核 |
| 对齐 | 视频与传感器残余误差 ≤ 0.1 s |

- **站立** — 四肢支撑的直立，含站立反刍。
- **躺卧** — 胸卧或侧卧。
- **行走** — 向前的位移（198 个窗口）。
- **采食** — 在料槽摄入饲料。反刍、舔舐和单纯头部活动不算采食。cowD 没有采食标签。

站立与采食重叠时，两条都保留。四分类表每个窗口只用一条标签。公开标识为 `DEVICE_NN`、`SESSION_NN` 和相对毫秒。字段见 [`docs/data_dictionary.md`](docs/data_dictionary.md)。

## 读取

在仓库根目录执行。依赖见 [`requirements.txt`](requirements.txt)（`numpy`、`pandas`）。

```bash
pip install -r requirements.txt
python code/load_example.py            # 第一条行走窗口
python code/load_example.py cowA 100   # 牛编号, 行号
```

```python
import numpy as np
import pandas as pd

X = np.load("01_data/acceleration.npy")   # (12736, 3, 256)，逐窗口归一化
meta = pd.read_csv("01_data/windows.csv", encoding="utf-8-sig")
row = meta.iloc[0]
approx_g = X[int(row.row_index)] * float(row.scale_g)
```

总数组顺序：cowA 0–6341，cowC 6342–6669，cowD 6670–7223，cowB 7224–12735。单牛数组在 `01_data/<cow>/acceleration.npy`。

## 目录

```text
cowtail/
├── 01_data/                  窗口表、单牛数组、尺度
├── 02_results/               归档指标
├── 03_models_predictions/    检查点与逐窗口预测
├── 04_raw_sensors_labels/    九轴原始流与参考区间
├── 05_six_cows_full/         另一份六牛现场数据
├── code/                     加载示例
├── docs/                     字段说明
└── FILE_MANIFEST.json        每个文件的大小与 SHA-256
```

四牛 cowA 即六牛 cowA。四牛 cowB 是六牛 cowB 的一个切片。四牛 cowC 是六牛 cowD。四牛 cowD 是六牛 cowE。对照表见 [`05_six_cows_full/README.md`](05_six_cows_full/README.md)。

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

扬州大学实验牧场伦理委员会已批准。仅尾根外挂。绝对时间、设备 MAC、原始牛号、视频文件名和标注人姓名已移除。

## 联系

张向前（通讯作者）— jiaosirty@stu.yau.edu.cn
