# Source provenance and release decisions

Source: the owner's local `research/Paper_Cutting_Project` folder. Original files were not modified. Original-source SHA-256 hashes for the four adapted code files are stored in `source_manifest.json`.

| Original | Repository treatment |
| --- | --- |
| `four_algorithms.py` | Primary research implementation in `paper_cutting/research.py`; removed embedded industrial demo, made Gurobi optional at import, corrected Phase 1 roll counting |
| `标准列生成.py` | `experiments/legacy/column_generation.py`; removed embedded example data |
| `CG+Diving Heuristic.py` | `experiments/legacy/column_generation_diving.py`; removed embedded example data |
| `3种算法测试.py` | `experiments/legacy/residual_heuristics.py`; removed embedded example data |
| `algorithm.ipynb` | Earlier duplicated implementations and experiment runner; not republished |
| `datapreprocessing.ipynb` | Workflow and demand-unit issue documented in methodology; notebook includes a personal path and factory outputs, so omitted |
| `export_result.csv`, `test_cases.json` | Industrial/derived data omitted |
| Three result spreadsheets and `实验.docx` | Historical results omitted; not fresh reproducible benchmark outputs |
| Two UNC JOURney manuscript DOCX files | Used for project context, not redistributed |
| `four_algorithms readable.pdf` | Redundant code rendering, omitted |
| `运筹学.pdf` | Reference reading, omitted |
| `Cutting_Pattern/`, `cutting-stock/` | Third-party Git repositories; omitted and credited below |

## Third-party reference repositories

Origins verified from their local Git configuration:

- https://github.com/jianyi1995/cutting-stock
- https://github.com/MariuszKorotko/Cutting_Pattern

Their histories and source files are not incorporated. This separation does not prove that all project code is independently authored; review any adapted fragments and applicable terms before selecting an open-source license.

## New repository engineering

The public API, CLI, tests, packaging, synthetic example, documentation and CI configuration were added during repository organization. The standard-library FFD follows the existing algorithm while tracking residual capacities directly. Original research code is preserved apart from the explicit changes above. Internal comments and experimental limitations remain visible instead of rewriting all methods into a new algorithm project.

No credentials or solver license files are included. No remote repository is configured by this package. All original local files remain available in the source folder.
