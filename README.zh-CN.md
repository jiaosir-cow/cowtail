<div align="center">
  <img src="assets/brand/cowmata-logo.svg" width="300" alt="COWMATA">

# CowTail-4

**牛尾根九轴惯性传感器行为数据集（完整标注）**

[English](README.md) · [简体中文](README.zh-CN.md)

<img alt="cows" src="https://img.shields.io/badge/cows-4-2f6f4e">
<img alt="sessions" src="https://img.shields.io/badge/sessions-31-2f6f4e">
<img alt="windows" src="https://img.shields.io/badge/windows-12%2C736-1f4e79">
<img alt="rate" src="https://img.shields.io/badge/50%20Hz-256%20samples%20(5.12%20s)-8a5a2b">
<img alt="classes" src="https://img.shields.io/badge/classes-4-444444">
<img alt="data licence" src="https://img.shields.io/badge/data-CC%20BY%204.0-green">
<img alt="code licence" src="https://img.shields.io/badge/code-MIT-blue">

</div>

---

## 📖 数据集简介

CowTail-4 是一份奶牛行为的现场数据集，由**尾根系挂的九轴惯性传感器**采集，包含三类内容：
**原始传感流**、用于生成参考标签的**标注层**，以及实验实际使用的**模型输入窗口**。这样一条
已发表的结果可以追溯到具体的哪一次记录、哪一条标签、哪一个窗口。

数据采集自商业奶牛场，公开发布前已做匿名化处理（设备与会话统一记为 `DEVICE_NN` /
`SESSION_NN`，时间只保留相对毫秒）。**视频不公开**，因为画面中包含可辨识的牧场工作人员和
牧场的动物标识。

本数据集对应论文：*Reusing public collar models for cattle behaviour recognition from tail
acceleration with limited labels*。

## 📊 数据规模

| 项目 | 数值 |
|---|---|
| 动物 | 4 头奶牛（论文中记为 cowA–cowD） |
| 设备 | 2 个尾根记录仪，在不同个体间轮换佩戴 |
| 采集会话 | 31 个 |
| 传感通道 | 9 轴：三轴加速度、三轴角速度、三轴磁场 |
| 采样率 | 50 Hz（原生） |
| 窗口 | 256 点 = 5.12 s，互不重叠 |
| 已标注窗口 | 12,736 个 |
| 行为类别 | 站立 · 躺卧 · 行走 · 采食 |
| 参考真值 | 三名标注员独立标注 + 兽医专业研究生复核 |
| 视频—传感器对齐 | 误差 ≤ 0.1 s |

**类别定义**（参考标签所依据的口径）：

- **站立**——四肢支撑的直立姿态；**包含站立反刍**，反刍不单独设类别。
- **躺卧**——胸卧或侧卧。
- **行走**——向前位移的运动；该类样本稀少（仅 198 个窗口），单类指标不稳定。
- **采食**——在料槽处实际摄入饲料。反刍、舔舐、单纯头部活动**不计**为采食。

> **cowD 没有采食真值。** 因此 cowD 上的任何采食指标都没有意义，论文的四分类评价已将其排除。

## 🗂 目录结构

```
cowtail/
├── 01_data/                  窗口总表、单牛数组、归一化尺度
├── 02_results/              归档的指标与混淆计数
├── 03_models_predictions/            检查点与逐窗口预测
├── 04_raw_sensors_labels/        九轴原始流与参考区间
├── 05_six_cows_full/         扩展的六牛现场数据（训练 + 测试）
├── code/                     加载示例与清单生成脚本
├── docs/                     字段字典与外部数据源说明
├── FILE_MANIFEST.json        每个文件的大小与 SHA-256
├── FINAL_VERIFICATION.json   本快照的核验记录
├── LICENSE                   数据许可（CC BY 4.0）
└── LICENSE-CODE              代码许可（MIT）
```

目录沿用标注软件 COWMATA Pro 的磁盘命名习惯，因此保持中文。逐字段定义见
[`docs/data_dictionary.md`](docs/data_dictionary.md)。

## 🚀 快速开始

```python
import numpy as np
import pandas as pd

X = np.load("01_data/acceleration.npy")          # (12736, 3, 256)，逐窗口归一化
meta = pd.read_csv("01_data/windows.csv", encoding="utf-8-sig")

print(X.shape)                                   # (12736, 3, 256)
print(meta[["display_cow", "session_id", "original_state"]].head())
```

`acceleration.npy` 是**逐窗口归一化**后的三轴加速度，乘回该行的 `scale_g` 才是近似 g 值。
单牛数组在 `01_data/<cow>/acceleration.npy`。九轴原始数据在 `04_raw_sensors_labels/`，通过
`session_id` 与窗口总表对应。

完整示例可从一条已标注窗口追到它所在的原始会话，并校验两条路径描述的是同一段运动：

```bash
python code/load_example.py            # 取第一条 WALKING 窗口
python code/load_example.py cowA 100   # 牛编号, 行号
```

## 📝 标注流程

- 每条记录由三名标注员独立标注。
- 一名兽医专业研究生复核全部标签并裁决分歧。
- 站立与采食同时发生时，该区间**标注两遍**——先站立、后采食，两条标签均保留。
- 视频与传感器时钟每个会话只对齐一次并锁定，残余对齐误差 ≤ 0.1 s。
- 四分类评价对每个窗口只使用一条无歧义的冻结标签；重叠的双标签保留在标注层，但不参与打分。

## ⚠️ 适用范围与局限

- **范围**：4 头牛、单一牧场，四牛集在单日完成采集。不要把这几头牛当作群体样本。
- **类别不平衡**：行走样本严重不足（198 个窗口），其单类 F1 不稳定，引用时必须同时给出样本数。
- **缺失真值**：cowD 无采食真值。
- **匿名化**：绝对时间、设备 MAC、原始牛号、视频文件名、标注人姓名均已移除；除数据所有者外，
  无法将区间重新关联到原始录像。
- **不用于福利诊断**：行为标签描述的是可观察状态，不是健康结论。

## 🔐 完整性校验

`FILE_MANIFEST.json` 记录每个文件的大小与 SHA-256。下载后可这样复核：

```bash
python code/make_manifest.py      # 重新生成
python verify_and_recompute.py              # 只读核验
```

## 📚 引用

使用本数据集请引用论文与数据集记录：

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

## ⚖️ 许可与伦理

- **数据**：CC BY 4.0，见 [`LICENSE`](LICENSE)。
- **代码**：MIT，见 [`LICENSE-CODE`](LICENSE-CODE)。
- **伦理**：全程非侵入（仅尾根外挂佩戴，无采血）。本实验已被扬州大学实验牧场伦理委员会审批通过。

## ✉️ 联系方式

- 张相庆（通讯作者）— zxq1991@mail.nwpu.edu.cn
- 问题与建议：请直接提交 GitHub Issue。
