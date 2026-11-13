# FINTECH 545 - Assignment 3

`Assignment3_Solutions.pdf` contains the written solutions. The seven CSVs in
`data/` are unchanged copies from the [course assignment](https://github.com/dompazz/FinTech-545-Fall2026/tree/5958d11b51960c18be98f6df2b97c4e80670fb2d/Assignments/Assignment3).

From `Assignment3/`, run these commands with Python 3.10 or later:

```sh
python -m pip install numpy==2.1.3 pandas==2.2.3 scipy==1.15.3 matplotlib==3.9.2
python assignment3.py
```

The script writes the numerical tables and intermediate results to `results.json`
and regenerates the three charts in `figures/`. The American option calculations
may take a few minutes. The option-book simulation uses 1,000,000 ten-day paths
with seed 545. The monthly weight bootstrap uses 1,000 resamples with seed 546.
