<p align="center">
  <img src="assets/brand/cowmata-logo.png" width="280" alt="COWMATA">
</p>

<h1 align="center">CowTail</h1>

<p align="center"><b>按行为分类的牛尾根加速度窗口。</b></p>

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
  <img alt="windows" src="https://img.shields.io/badge/windows-29790-555555">
  <img alt="sampling" src="https://img.shields.io/badge/sampling-50%20Hz-555555">
</p>

这是与 [COWMATA Pro](https://github.com/zxq309/COWMATA-Pro) 配套的公开数据仓库。

**Yang, Zhanhai · Jiao, Tengyu · Zhang, Jianqi · Zhang, Xiangqing · Zhang, Yalong · Jiang, Qinglin · Zhang, Yihui**

## 数据

一个文件夹是一种行为。全部标注窗口混在对应文件夹里，文件中没有牛、设备、会话和时间。

| 文件夹 | 行为 | 窗口数 | 文件 |
|---|---|---:|---|
| `站立` | 站立，包含站立反刍 | 9,240 | `windows.npy` |
| `起卧` | 躺卧 | 12,245 | `windows.npy` |
| `行走` | 行走 | 681 | `windows.npy` |
| `采食` | 在料槽采食 | 7,624 | `windows.npy` |

合计 29,790 个窗口。每个 `windows.npy` 是 `float32`，形状 `(窗口数, 3, 256)`：x、y、z 加速度，单位约为 g，50 Hz，5.12 秒，窗口互不重叠。反刍、舔舐和单纯头部活动不算采食。

```bash
pip install -r requirements.txt
python code/load_example.py 行走
```

```python
import numpy as np
x = np.load("行走/windows.npy")   # (681, 3, 256)
```

## 引用

```bibtex
@dataset{cowtail_2026,
  title  = {CowTail: Tail-Root Acceleration Windows by Cattle Behaviour},
  author = {Yang, Zhanhai and Jiao, Tengyu and Zhang, Jianqi and Zhang, Xiangqing and
            Zhang, Yalong and Jiang, Qinglin and Zhang, Yihui},
  year   = {2026},
  url    = {https://github.com/jiaosir-cow/cowtail},
  note   = {Data licence: CC BY 4.0; code licence: MIT}
}
```

同一条记录见 [`CITATION.cff`](CITATION.cff)。

## 许可

数据 [CC BY 4.0](LICENSE)。代码 [MIT](LICENSE-CODE)。COWMATA 名称和 [`assets/brand/`](assets/brand/) 中的文件仍是公司品牌资产，见 [`NOTICE`](NOTICE)。

## 致谢

扬州大学实验牧场伦理委员会已批准。仅尾根外挂。标签只写可观察状态。

## 联系

张相庆（通讯作者）— jiaosirty@stu.yau.edu.cn

问题请提交 GitHub Issue。
