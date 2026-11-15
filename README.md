# Paper Cutting Optimization

**Reducing paper-roll consumption while tracking the operational cost of cutting-pattern changes.**

This project compares greedy packing, column generation, and sequential heuristics for the **one-dimensional cutting stock problem**.

Given a master-roll width and quantities of narrower products, the software constructs cutting patterns and reports roll usage, distinct pattern count, trim loss, and runtime. Pattern count is a proxy for setup complexity; the repository does not model a production sequence or measure actual factory savings.

## Quick start — no solver installation required

Requires Python 3.10 or newer. From this repository's root:

```bash
python -m paper_cutting examples/synthetic.json
python -m paper_cutting examples/synthetic.json --output outputs/ffd.json
python -m unittest discover -s tests -v
```

The bundled instance is **synthetic**, with widths and quantities invented for this repository. FFD uses 9 rolls and 5 distinct patterns, with 800 width-units of trim (8.89%). See [example output](examples/ffd_result.json).

For an installable command-line tool:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
paper-cutting examples/synthetic.json
```

Windows users can activate the environment with `.venv\Scripts\activate`.

## Algorithms

| Method | Implementation | Dependencies | Role |
| --- | --- | --- | --- |
| `ffd` | First Fit Decreasing | Standard library | Fast, tested baseline |
| `shp` | Haessler-style sequential heuristic | NumPy | Repeated patterns via aspiration levels |
| `mshp` | Modified SHP: pattern generation + selection | NumPy, PuLP, CBC | Weighted roll/pattern-cost heuristic |
| `cg` | LP column generation + restricted integer master | NumPy, Gurobi | Material-use comparator |

```bash
python -m pip install -e '.[research]'
python -m paper_cutting examples/synthetic.json --method shp
python -m paper_cutting examples/synthetic.json --method mshp

python -m pip install -e '.[research,gurobi]'
python -m paper_cutting examples/synthetic.json --method cg
```

PuLP must have access to CBC; Gurobi requires a usable license. The optional research implementations are retained from the original project and need further solver-backed validation. Their dependency ranges are compatibility targets, not a frozen environment from the original experiments.

**CG here does not prove the global integer optimum:** it optimizes an integer master over generated columns, without branch-and-price. Likewise, `rolls_above_lower_bound` is an absolute difference from a simple analytical bound, not a solver's certified MIP gap.

## Input and output

```json
{
  "id": "synthetic-demo",
  "roll_width": 1000,
  "demands": {"450": 8, "350": 6, "200": 10, "100": 5}
}
```

Widths must be positive integers using the same unit; demand values are **integer piece counts**, not total linear length. Each requested width must fit within the master roll. Zero-demand entries are ignored; an all-zero instance is rejected. The public interface caps demand at 100,000 pieces because the retained implementations expand pieces or rolls in memory.

Output includes pattern frequencies, per-roll trim, overproduction, roll and pattern counts, elapsed time, and a lower bound. An independent validator rejects incomplete or over-width solutions. Overproduction is reported separately from physical trim.

```python
from paper_cutting import solve
result = solve(1000, {450: 8, 350: 6, 200: 10, 100: 5}, method="ffd")
print(result["n_rolls"], result["n_patterns"])
```

## Project layout

| Path | Purpose |
| --- | --- |
| `paper_cutting/core.py` | Validated API, FFD baseline, independent metrics |
| `paper_cutting/research.py` | Adapted original SHP, MSHP, CG and related methods |
| `examples/` | Synthetic input and reproducible FFD output |
| `experiments/legacy/` | Earlier CG, diving and residual-heuristic versions |
| `tests/` | Feasibility, conservation, input validation, tiny exact oracle, CLI |
| `docs/methodology.md` | Mathematical interpretation and limitations |
| `docs/provenance.md` | Source mapping, changes, exclusions and third-party references |
| `docs/validation.md` | What was actually tested |
| `docs/publishing.md` | Repository name, description and upload steps |

## Research context

The accompanying source manuscript is titled *Balancing Material Waste and Operational Efficiency: A Comparative Study of Algorithms for the One-Dimensional Cutting Stock Problem*. It describes random benchmark classes and industrial instances, motivating the tradeoff between roll usage and cutting-pattern changes.

This repository is a curated code release, **not a full reproduction of the manuscript**. Historical industrial results, original orders, manuscript files and notebook outputs are excluded. The current code and manuscript differ in knife constraints and MSHP objective details; see [methodology](docs/methodology.md). No historical performance or savings claims are presented as newly reproduced results.

## Attribution and reuse

The local project contained two third-party reference repositories. Their source is not redistributed here; their recorded origins are credited in [provenance](docs/provenance.md). No blanket open-source license has been selected: confirm ownership of internship code and third-party-derived portions before granting redistribution rights. Original research comments are retained in Chinese where useful for traceability.
