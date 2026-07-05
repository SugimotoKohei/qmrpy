# Command Line Interface

qmrpy provides a thin `argparse` CLI for common NIfTI workflows and validation.

## Info

```bash
qmrpy info
```

## Fit NIfTI Data

The first CLI fitting targets are `t2-mono`, `t1rho`, and `mtr`. Input data must
be a NIfTI image with the acquisition dimension last.

```bash
qmrpy fit t2-mono \
  --input sub-01_echoes.nii.gz \
  --output sub-01_t2map.nii.gz \
  --te-ms 10,20,40,80 \
  --mask otsu \
  --n-jobs -1
```

```bash
qmrpy fit t1rho \
  --input sub-01_tsl.nii.gz \
  --output sub-01_t1rho.nii.gz \
  --tsl-ms 0,10,30,60
```

```bash
qmrpy fit mtr \
  --input sub-01_mt_pair.nii.gz \
  --output sub-01_mtr.nii.gz
```

## Validate

```bash
qmrpy validate --suite core --out-dir output/reports/parity_summary
```
