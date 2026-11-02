<div align="center">
  <img src="assets/brand/cowmata-logo.svg" width="300" alt="COWMATA">

# CowTail-4

**牛尾根九轴惯性传感器行为数据集（完整标注）。**

[English](README.md) · [简体中文](README.zh-CN.md)

<img alt="data licence" src="https://img.shields.io/badge/data-CC%20BY%204.0-green">
<img alt="code licence" src="https://img.shields.io/badge/code-MIT-blue">

</div>

CowTail-4 是论文 *Reusing public collar models for cattle behaviour recognition from tail acceleration with limited labels* 所发布的四头牛尾根数据集。仓库里同时有原始传感流、标注层和模型输入窗口，因此一条已发表结果可以追溯到具体记录、标签和窗口。

## 60 秒开始

在仓库根目录执行。依赖见 [`requirements.txt`](requirements.txt)（`numpy`、`pandas`）：

```bash
pip install -r requirements.txt
python code/load_example.py            # 第一条 WALKING 窗口
python code/load_example.py cowA 100   # 牛编号, 行号
```

也可以直接读数组：

```python
import numpy as np
import pandas as pd

X = np.load("01_data/acceleration.npy")          # (12736, 3, 256)，逐窗口归一化
meta = pd.read_csv("01_data/windows.csv", encoding="utf-8-sig")

print(X.shape)                                   # (12736, 3, 256)
print(meta[["display_cow", "session_id", "original_state"]].head())
row = meta.iloc[0]
approx_g = X[int(row.row_index)] * float(row.scale_g)
```

`acceleration.npy` 按窗口归一化。`01_data/windows.csv` 的 `row_index` 就是这个数组的行号，乘同一行的 `scale_g` 得到近似 *g*。总数组**不是**按 cowA–cowD 排列：cowA（0–6341 行）、cowC（6342–6669）、cowD（6670–7223）、cowB（7224–12735）。单牛数组在 `01_data/<cow>/acceleration.npy`。九轴原始流在 `04_raw_sensors_labels/`，用 `session_id` 与窗口表对应。

## 数据规模

| 项目 | 数值 |
|---|---|
| 动物 | 4 头奶牛（论文中记为 cowA–cowD） |
| 设备 | 2 个尾根记录仪，在不同个体间轮换佩戴 |
| 采集会话 | 31 个，单日，单一牧场 |
| 传感通道 | 9 轴：三轴加速度、三轴角速度、三轴磁场 |
| 采样率 | 50 Hz（原生） |
| 窗口 | 256 点 = 5.12 s，互不重叠 |
| 已标注窗口 | 12,736 个 |
| 行为类别 | 站立 · 躺卧 · 行走 · 采食 |
| 参考真值 | 三名标注员独立标注 + 兽医专业研究生复核 |
| 视频—传感器对齐 | 锁定后残余误差 ≤ 0.1 s |

参考标签的类别定义：

- **站立**——四肢支撑的直立姿态，包含站立反刍。反刍不单独设类。
- **躺卧**——胸卧或侧卧。
- **行走**——向前的位移运动。该类有 198 个窗口，单类指标不稳定。
- **采食**——在料槽处实际摄入饲料。反刍、舔舐和单纯头部活动不计为采食。

cowD 没有采食真值（`01_data/four_cow_label_and_window_counts.csv` 中 `FEEDING` 为 0）。cowD 上的采食指标无法解释，论文的四分类评价已将其排除。

视频不公开。公开文件只使用 `DEVICE_NN` / `SESSION_NN` 和相对毫秒，因为画面中有可辨识的牧场工作人员和牧场动物标识。

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

同一条记录见 [`CITATION.cff`](CITATION.cff)。

## 伦理与局限

- **许可。** 数据 CC BY 4.0（[`LICENSE`](LICENSE)）。代码 MIT（[`LICENSE-CODE`](LICENSE-CODE)）。
- **伦理。** 扬州大学实验牧场伦理委员会已批准。全程非侵入：仅尾根外挂，无采血。
- **范围。** 4 头牛、单一牧场、单日采集。不要把这几头牛当作群体样本。
- **不平衡。** 行走有 198 个窗口。引用行走 F1 时必须同时给出这个样本数。
- **缺失真值。** cowD 无采食真值。
- **匿名化。** 绝对时间、设备 MAC、原始牛号、视频文件名和标注人姓名已移除。除数据所有者外，不能把区间重新连回录像。
- **不是福利诊断。** 标签描述可观察状态，不是健康结论。

## 目录

```
cowtail/
├── 01_data/                  窗口总表、单牛数组、归一化尺度
├── 02_results/               归档的指标与混淆计数
├── 03_models_predictions/    检查点与逐窗口预测
├── 04_raw_sensors_labels/    九轴原始流与参考区间
├── 05_six_cows_full/         另一份现场数据包（见该目录自己的 README）
├── code/                     加载示例与清单生成脚本
├── docs/                     字段字典与外部数据源说明
├── FILE_MANIFEST.json        每个文件的大小与 SHA-256
├── FINAL_VERIFICATION.json   本快照的核验记录
├── LICENSE                   数据许可（CC BY 4.0）
└── LICENSE-CODE              代码许可（MIT）
```

`05_six_cows_full/README.md` 写的是那一份数据自己的规模和划分，不要把它的数字当成上面的四牛表。两边的同名牛也不是同一头：四牛包的 cowB 只是六牛包 cowB 的一个切片，四牛包的 cowC 是六牛包的 cowD，四牛包的 cowD 是六牛包的 cowE。拼接两份数据前先看那份 README 里的对照表。字段定义见 [`docs/data_dictionary.md`](docs/data_dictionary.md)。

## 标注流程

- 每条记录由三名标注员独立标注。
- 一名兽医专业研究生复核全部标签并裁决分歧。
- 站立与采食同时发生时，该区间标注两遍：先站立，后采食，两条都保留。
- 每个会话的视频与传感器时钟只对齐一次并锁定，残余对齐误差 ≤ 0.1 s。
- 四分类评价对每个窗口只使用一条无歧义的冻结标签。重叠的双标签留在标注层，不参与打分。

## 完整性

`FILE_MANIFEST.json` 记录每个文件的大小与 SHA-256。

```bash
python code/make_manifest.py      # 重新生成
python verify_and_recompute.py    # 只读核验
```

## 联系

张相庆（通讯作者）— zxq1991@mail.nwpu.edu.cn

问题请提交 GitHub Issue。
