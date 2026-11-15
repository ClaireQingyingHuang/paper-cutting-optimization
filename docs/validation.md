# Validation record

Validated during repository preparation on Python 3.13.7:

- Eight standard-library unittest cases passed, including 100 seeded randomized feasibility/conservation instances and exhaustive tiny-instance lower-bound comparisons.
- CLI executed successfully on the synthetic example: 9 rolls, 5 patterns, trim 800 / 9000 = 8.89%, no overproduction, lower bound 9.
- All delivered Python source files passed compilation.
- Factory CSV/JSON inputs, original notebooks, manuscript files, spreadsheet outputs, third-party source trees and solver licenses are absent from the release archive.

Not validated in this environment:

- SHP, MSHP, CG and legacy algorithms were not executed. NumPy, PuLP and Gurobi were unavailable; package installation failed because the package index could not be reached.
- Package installation and the GitHub Actions workflow have not been executed. CI targets Python 3.10 and 3.12; the local validation used 3.13.7.
- Historical paper tables, figures and timing claims have not been reproduced. The repository intentionally includes no industrial benchmark claims.

Run the commands in the README in a solver-enabled environment before relying on the experimental methods. The saved example's runtime is only one observed local timing and is not a benchmark.
