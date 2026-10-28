# External datasets used in the paper

These third-party datasets were used for pretraining / comparison. They are **not
redistributed here** — each has its own licence and canonical host. Use the links.

| Shorthand | Dataset | Source | Licence | Sensor / rate |
|---|---|---|---|---|
| JBB | Japanese Black Beef cow behaviour | Zenodo 10.5281/zenodo.5849025 | CC BY (see record) | neck 3-axis, 25 Hz, 6 cows |
| AcTBeCalf | Pre-weaned calf behaviour | Zenodo 10.5281/zenodo.13259482 | CC BY 4.0 | neck 3-axis, 25 Hz, 30 calves |
| UFRO | db-cow-walking | github.com/WASP-lab/db-cow-walking | none declared | collar 9-axis (BNO055 + MPU9250), 10 Hz, 10 cows |
| WASP | db-cow-mounts | github.com/WASP-lab/db-cow-mounts | none declared | collar 9-axis, 10 Hz, 8 cows |

Notes for reuse:

- The two GitHub repositories declare no licence. Treat them as "all rights
  reserved" until the authors grant permission; cite the accompanying papers.
- Sampling rates, mounting positions (neck vs collar vs tail) and label taxonomies
  differ from this dataset. Cross-dataset transfer requires re-windowing and a label
  mapping; the class names are not interchangeable.
- This release contains only our own field data (six cows, 28,911 windows) and the
  unlabelled 9-axis pool. No third-party recordings are included.
