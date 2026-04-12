# FateFormer

Multimodal transformer models for cell fate prediction (`reprogramming` vs `dead-end`) from:
- RNA-seq
- ATAC-seq
- Metabolic flux


### Prepare data

Download from [Zenodo](https://zenodo.org/17864926) and place files in `datasets/`:
- `clones.csv`
- `all_atac_d3_motif.h5ad`
- `flux_labelled.csv`
- `all_rna_d3_labelled.h5ad`
- `all_rna_d3_unlabelled.h5ad`

### Training / fine-tuning

Use notebooks:
- `Model_RNA.ipynb` (train on RNA only)
- `Model_ATAC.ipynb` (train on ATAC only)
- `Model_Flux.ipynb` (train on flux only)
- `Model_Multimodal.ipynb` (train multimodal model)

### Full benchmark

```bash
python model_analysis.py
```

Default behavior:
- creates 4 models: RNA, ATAC, Flux, Multimodal
- uses 5-fold CV
- uses 5 seeds (`[0, 6, 42, 123, 1000]`)
- Total runs: 100
- Writes outputs to: `analysis docs/metrics/`

Outputs:
- `analysis docs/metrics/models/` - trained checkpoints per fold/seed/model
- `analysis docs/metrics/metrics/` - CSV metric summaries
- `analysis docs/metrics/fold_results/` - per-fold serialized results (`.pkl`)

### Plotting and analysis

Open `Plots.ipynb` after `model_analysis.py` completes.



