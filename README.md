<p align="center">
  <img src="assets/brand/cowmata-logo.png" width="280" alt="COWMATA">
</p>

<h1 align="center">CowTail</h1>

<p align="center"><b>Tail-root acceleration windows, pooled by behaviour.</b></p>

<p align="center">
  <a href="README.md"><b>English</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/zxq309/COWMATA-Pro"><b>COWMATA Pro</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/jiaosir-cow/cowtail/issues"><b>Questions</b></a>
</p>

<p align="center">
  <img alt="data licence" src="https://img.shields.io/badge/data-CC%20BY%204.0-2ea44f">
  <img alt="code licence" src="https://img.shields.io/badge/code-MIT-1f6feb">
  <img alt="windows" src="https://img.shields.io/badge/windows-29790-555555">
  <img alt="sampling" src="https://img.shields.io/badge/sampling-50%20Hz-555555">
</p>

This is the public data repository released with [COWMATA Pro](https://github.com/zxq309/COWMATA-Pro).

**Yang, Zhanhai · Jiao, Tengyu · Zhang, Jianqi · Zhang, Xiangqing · Zhang, Yalong · Jiang, Qinglin · Zhang, Yihui**

## Data

Each folder is one behaviour. All labelled windows are mixed in that folder. The files do not identify animals, devices, sessions, or time.

| Folder | Behaviour | Windows | File |
|---|---|---:|---|
| `standing` | standing, including standing rumination | 9,240 | `windows.npy` |
| `lying` | lying | 12,245 | `windows.npy` |
| `walking` | walking | 681 | `windows.npy` |
| `feeding` | feeding at the trough | 7,624 | `windows.npy` |

29,790 windows in total. Each array is `float32` with shape `(windows, 3, 256)`: x, y, z acceleration in approximate *g*, 50 Hz, 5.12 s, non-overlapping. Rumination, licking, and head movement alone are not feeding.

```bash
pip install -r requirements.txt
python code/load_example.py walking
```

```python
import numpy as np
x = np.load("walking/windows.npy")   # (681, 3, 256)
```

## Citation

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

The same record is in [`CITATION.cff`](CITATION.cff).

## Licence

Data: [CC BY 4.0](LICENSE). Code: [MIT](LICENSE-CODE). The COWMATA name and the files in [`assets/brand/`](assets/brand/) stay company brand assets. See [`NOTICE`](NOTICE).

## Acknowledgement

Approved by the Ethics Committee of the Experimental Farm of Yangzhou University. External tail-root mounting only. Labels name observable states.

## Contact

Xiangqing Zhang (corresponding author) — jiaosirty@stu.yau.edu.cn

Questions: open a GitHub issue.
