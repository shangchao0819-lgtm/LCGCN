# Datasets and provenance

The public repository does not distribute datasets. The frozen experiment expects five preprocessed academic-network hypergraphs and ten supplied train/test splits per dataset. By default, source checkouts read `LCGCN/data`, or use an explicit `--data-dir`. All 65 expected input files are identified by their original SHA256 hashes; do not resample splits when comparing with the frozen experiments.

| CLI name | Nodes | Hyperedges | Features | Classes |
|---|---:|---:|---:|---:|
| dblp | 41302 | 22363 | 1425 | 6 |
| cora-coauthorship | 2708 | 1072 | 1433 | 7 |
| pubmed | 19717 | 7963 | 500 | 3 |
| citeseer | 3312 | 1079 | 3703 | 6 |
| cora-cocitation | 2708 | 1579 | 1433 | 7 |

Features are converted to float32 CSR and row-normalized. Labels are mapped to contiguous integer classes. Serialized hyperedge/member order is retained, with ascending identifiers for set-valued memberships. Splits contain train/test node indices; no extra outer validation set is introduced. Only trusted pickle files should be loaded, because pickle deserialization can execute code.

`docs/data_manifest.json` provides the expected SHA256 hashes relative to the data directory. An authorized local input collection must contain the same manifest as `data/manifest.json`. `docs/dataset_audit.json` records the statistics for all 50 splits. The repository contains input metadata, not the preprocessed inputs, original document collections, or an end-to-end raw-data generation pipeline. It does not bundle historical predictions.

## Redistribution status

The immediate source is the frozen local experimental input collection. A candidate public upstream is the [official HyperGCN repository](https://github.com/malllabiisc/HyperGCN), associated with Yadati et al., *HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs*, NeurIPS 2019.

The upstream [LICENSE](https://github.com/malllabiisc/HyperGCN/blob/master/LICENSE) was inspected on 2026-11-11 and declares Apache-2.0 for that repository. This does not establish a complete redistribution chain for these local preprocessed files or the underlying datasets. Exact upstream file identity, version, preprocessing/split provenance, and third-party rights remain to be confirmed, including the 41,302-node DBLP variant.

**Redistribution clearance is pending for all five datasets.** Consequently, neither raw pickle files nor a complete data ZIP is published here. MIT applies to this project's code, not third-party datasets. Git ignores `data/`, and wheels exclude dataset files. The upstream link above is a provenance lead, not a verified byte-identical download for this experiment. A validated public acquisition pipeline is not yet available.

## Using an existing authorized input collection

1. Obtain permission to use the preprocessed input collection from its provider. This repository currently provides no public dataset archive.
2. Keep the directories `coauthorship/cora`, `coauthorship/dblp`, `cocitation/cora`, `cocitation/citeseer`, and `cocitation/pubmed` under a single data root. Each contains `features.pickle`, `labels.pickle`, `hypergraph.pickle`, and `splits/1.pickle` through `splits/10.pickle`.
3. Keep `manifest.json` at that data root. If it is absent from an otherwise authorized input collection, copy `docs/data_manifest.json` there as `manifest.json`; this supplies checksums only, not missing data.
4. Verify inputs before training:

```shell
python main.py --data-dir "path/to/data" --check --device cpu
python main.py --data-dir "path/to/data" --smoke --device cpu
python main.py --data-dir "path/to/data"
```

An existing authorized `LCGCN_data.zip` extracts to `LCGCN_data/data/`; use that inner directory. An existing complete local bundle has its data under `LCGCN/data/`. The older local archives are not GitHub release assets and are not relicensed by this repository.
