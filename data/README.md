# Data

| File | Size | Content |
|---|---:|---|
| `energymind_full_joined.csv.gz` | 33 MB | Full dataset: 1,051,200 rows × 67 columns, 60 meters, 7 feeders, 3 nodes, 30-min, year 2025 |
| `sample/energymind_sample.csv.gz` | 0.3 MB | N1, 28 May – 3 June 2025 (10,080 rows, 9 fault types). Built by `scripts/make_sample.py` |

Full description, column dictionary, injection recipes and limits: [`../docs/data_card.md`](../docs/data_card.md).

```python
from energymind.data import load, feature_columns
df = load()               # N1, full year (project scope)
df = load(sample=True)    # N1, sample week
df = load(node=None)      # all 3 nodes (Goal 4 only)
X = df[feature_columns(df)]   # never includes gt_* columns, oracle values or simulator settings
```

All data are **simulated**; no real STEG meter data is included.
