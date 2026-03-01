# Data Access

This study re-analyses publicly available multi-omics data. No new data were generated.

## Data Sources

| Dataset | Source | Views | Reference |
|---------|--------|-------|-----------|
| MLOmics BRCA | HuggingFace (AIBIC/MLOmics) | mRNA, miRNA, methylation, CNV | MLOmics benchmark; TCGA-BRCA (Cancer Genome Atlas Network, 2012) |
| IBDMDB | iHMP/HMP2 (ibdmdb.org) | MGX, MGX_func, MPX, MBX | Lloyd-Price et al. 2019 |
| CCLE/DepMap | DepMap Portal (24Q2) | mRNA, CNV, proteomics | Ghandi et al. 2019; Nusinow et al. 2020 |
| TCGA-GBM | UCSC Xena Hub | mRNA, methylation, CNV | TCGA Network 2008; Goldman et al. 2020 |

## Download Instructions

### 1) MLOmics BRCA

**Source:** HuggingFace — AIBIC/MLOmics benchmark  
**URL:** https://huggingface.co/datasets/AIBIC/MLOmics  
**Subdirectory:** `Main_Dataset/Classification_datasets/GS-BRCA/Aligned/`

Download:
- `BRCA_mRNA_aligned.csv`
- `BRCA_miRNA_aligned.csv`
- `BRCA_Methy_aligned.csv`
- `BRCA_CNV_aligned.csv`
- `BRCA_label_num.csv`

Clinical labels (PAM50 subtypes):
- `BRCA_clinicalMatrix` from UCSC Xena: https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix

Place in: `data/raw/mlomics/`

### 2) IBDMDB (iHMP/HMP2)

**Source:** Integrative Human Microbiome Project — IBDMDB portal  
**URL:** https://ibdmdb.org/  
**Note:** Files are served via Globus-backed links from the IBDMDB run-products page.

Download:
- `taxonomic_profiles_3.tsv` (metagenomic taxonomy)
- `genefamilies.tsv` (functional gene families; ~3.7 GB)
- `HMP2_proteomics_ecs.tsv` (metaproteomics)
- `HMP2_metabolomics_w_metadata.biom` (metabolomics)
- `hmp2_metadata.csv` (sample metadata, participant IDs, diagnosis)

Place in: `data/raw/ibdmdb/`

### 3) CCLE/DepMap

**Source:** DepMap Portal — Public release 24Q2  
**URL:** https://depmap.org/portal/  
**API:** https://depmap.org/portal/api/download/files (release: "DepMap Public 24Q2")

Download via DepMap portal or API:
- `OmicsExpressionProteinCodingGenesTPMLogp1.csv` (mRNA expression)
- `OmicsCNGene.csv` (copy number)
- `Model.csv` (cell line metadata and tissue annotations)

Proteomics (separate source — Gygi Lab):
- `protein_quant_current_normalized.csv.gz`
- **URL:** https://gygi.hms.harvard.edu/data/ccle/protein_quant_current_normalized.csv.gz

Place in: `data/raw/ccle/`

### 4) TCGA-GBM

**Source:** UCSC Xena Hub — TCGA GBM cohort  
**URL:** https://tcga.xenahubs.net/  
**Cohort:** `TCGA.GBM.sampleMap`

Download:
- `HiSeqV2.gz` (mRNA expression)
- `HumanMethylation450.gz` (methylation beta values)
- `Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz` (CNV)
- `GBM_clinicalMatrix` (subtype labels)

Direct download links:
```
https://tcga.xenahubs.net/download/TCGA.GBM.sampleMap/HiSeqV2.gz
https://tcga.xenahubs.net/download/TCGA.GBM.sampleMap/HumanMethylation450.gz
https://tcga.xenahubs.net/download/TCGA.GBM.sampleMap/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz
https://tcga.xenahubs.net/download/TCGA.GBM.sampleMap/GBM_clinicalMatrix
```

Place in: `data/raw/tcga_gbm/`

## Automated Download

The pipeline includes a download script that retrieves all datasets programmatically:

```bash
python code/compute/00_download/01_download_all_data.py --datasets all
```

This script handles DepMap API queries, Globus-backed IBDMDB links, HuggingFace downloads, and Xena Hub retrieval with automatic integrity checks.

## Verification

```bash
# Check all raw files are present and non-empty
python -c "
import os
expected = {
    'data/raw/mlomics': ['BRCA_mRNA_aligned.csv', 'BRCA_miRNA_aligned.csv',
                          'BRCA_Methy_aligned.csv', 'BRCA_CNV_aligned.csv',
                          'BRCA_label_num.csv', 'BRCA_clinicalMatrix'],
    'data/raw/ibdmdb': ['taxonomic_profiles_3.tsv', 'genefamilies.tsv',
                         'HMP2_proteomics_ecs.tsv', 'HMP2_metabolomics_w_metadata.biom',
                         'hmp2_metadata.csv'],
    'data/raw/ccle':   ['OmicsExpressionProteinCodingGenesTPMLogp1.csv',
                         'OmicsCNGene.csv', 'Model.csv',
                         'protein_quant_current_normalized.csv'],
    'data/raw/tcga_gbm': ['HiSeqV2', 'HumanMethylation450',
                           'Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes',
                           'GBM_clinicalMatrix']
}
for folder, files in expected.items():
    for f in files:
        path = os.path.join(folder, f)
        gz = path + '.gz'
        found = os.path.exists(path) or os.path.exists(gz)
        size = os.path.getsize(path) if os.path.exists(path) else (os.path.getsize(gz) if os.path.exists(gz) else 0)
        status = '✅' if found and size > 0 else '❌'
        print(f'{status} {path} ({size:,} bytes)')
"
```

## Expected Directory Structure

```text
data/raw/
├── mlomics/
│   ├── BRCA_mRNA_aligned.csv
│   ├── BRCA_miRNA_aligned.csv
│   ├── BRCA_Methy_aligned.csv
│   ├── BRCA_CNV_aligned.csv
│   ├── BRCA_label_num.csv
│   └── BRCA_clinicalMatrix
├── ibdmdb/
│   ├── taxonomic_profiles_3.tsv
│   ├── genefamilies.tsv
│   ├── HMP2_proteomics_ecs.tsv
│   ├── HMP2_metabolomics_w_metadata.biom
│   └── hmp2_metadata.csv
├── ccle/
│   ├── OmicsExpressionProteinCodingGenesTPMLogp1.csv
│   ├── OmicsCNGene.csv
│   ├── Model.csv
│   └── protein_quant_current_normalized.csv
└── tcga_gbm/
    ├── HiSeqV2
    ├── HumanMethylation450
    ├── Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes
    └── GBM_clinicalMatrix
```

## Dataset Summary

| Dataset | n | Classes | Task | Views | Role |
|---------|---|---------|------|-------|------|
| MLOmics BRCA | 671 | 5 (PAM50) | Breast cancer subtype | mRNA, miRNA, methylation, CNV | Primary |
| IBDMDB | 155 | 3 (nonIBD, UC, CD) | IBD diagnosis | MGX, MGX_func, MPX, MBX | Primary |
| CCLE/DepMap | 369 | 22 (tissue lineage) | Tissue-of-origin | mRNA, CNV, proteomics | Primary |
| TCGA-GBM | 47 | 4 (GBM subtypes) | GBM subtype | mRNA, methylation, CNV | Sensitivity |

## Sensitivity Views

Two additional views were generated for robustness checks (not separate downloads):

| View | Derived from | Transform |
|------|-------------|-----------|
| IBDMDB MGX_CLR | MGX taxonomy | Centred log-ratio (pseudocount 1e-6) |
| TCGA-GBM methylation_Mval | methylation beta | Logit transform (ε-clipped at 1e-6) |

## Feature Caps

Ultra-high-dimensional views were randomly subsampled (seed = 1, without replacement) to fixed caps **without selecting top-variance features**, preserving the variance distribution:

| View | Original features | Capped to | Seed |
|------|------------------|-----------|------|
| IBDMDB MGX_func | 917,164 | 10,000 | 1 |
| IBDMDB MBX | 81,867 | 20,000 | 1 |
| TCGA-GBM methylation | ~450,000 | 20,000 | 1 |

## References

1. Cancer Genome Atlas Network (2012). Comprehensive molecular portraits of human breast tumours. *Nature* 490, 61–70.

2. Parker, J.S. et al. (2009). Supervised risk predictor of breast cancer based on intrinsic subtypes. *Journal of Clinical Oncology* 27, 1160–1167.

3. Lloyd-Price, J. et al. (2019). Multi-omics of the gut microbial ecosystem in inflammatory bowel diseases. *Nature* 569, 655–662.

4. Ghandi, M. et al. (2019). Next-generation characterization of the Cancer Cell Line Encyclopedia. *Nature* 569, 503–508.

5. Nusinow, D.P. et al. (2020). Quantitative proteomics of the Cancer Cell Line Encyclopedia. *Cell* 180, 387–402.

6. TCGA Research Network (2008). Comprehensive genomic characterization defines human glioblastoma genes and core pathways. *Nature* 455, 1061–1068.

7. Verhaak, R.G.W. et al. (2010). Integrated genomic analysis identifies clinically relevant subtypes of glioblastoma. *Cancer Cell* 17, 98–110.

8. Goldman, M.J. et al. (2020). Visualizing and interpreting cancer genomics data via the Xena platform. *Nature Biotechnology* 38, 675–678.
