"""Write data/sample/energymind_sample.csv.gz: N1, the 7 days with the most distinct fault types.

Lets reviewers run the notebooks without the full 33 MB file.  Run: python scripts/make_sample.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energymind.data.load import DATA_PATH, SAMPLE_PATH, load  # noqa: E402

df = load(DATA_PATH)
day = df.timestamp.dt.floor("D")
types_per_day = df[df.gt_subtype.isin(["none", "minor_gap"]) == False].groupby(day).gt_subtype.unique()
days = sorted(day.unique())
best = max(range(len(days) - 6),
           key=lambda i: len(set().union(*[set(types_per_day.get(d, [])) for d in days[i:i + 7]])))
start, end = days[best], days[best + 7]
sample = df[(df.timestamp >= start) & (df.timestamp < end)]
SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
sample.to_csv(SAMPLE_PATH, index=False)
print(f"{start:%Y-%m-%d}..{end:%Y-%m-%d}: {len(sample):,} rows, "
      f"{sample.gt_subtype.nunique() - 1} fault types -> {SAMPLE_PATH}")
