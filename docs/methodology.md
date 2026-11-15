# Methodology and interpretation

## Problem

For width W, product widths w_i and piece demands d_i, a feasible pattern a_j satisfies sum_i w_i a_ij <= W. The classical master minimizes sum_j x_j subject to sum_j a_ij x_j >= d_i and nonnegative integer pattern frequencies x_j.

The public implementation assumes one stock width, zero kerf and no order sequencing, deadlines, inventory reuse or grade changes within an instance. Distinct pattern count measures the variety of setups, not the exact number or duration of machine changes.

## Research-code audit

These observations come from the supplied local code and manuscript, rather than a fresh literature replication.

- CG solves LP masters and knapsack pricing, then a restricted integer master. It is not branch-and-price and does not guarantee a globally optimal integer plan. Its internal iteration cap and solver status handling merit further work.
- The manuscript describes machine-specific knife limits. The supplied code uses MAXR=100 in SHP/MSHP and does not apply a shared knife limit to FFD/CG. The repository therefore makes no machine-feasibility claim beyond width capacity.
- MSHP's selection objective is `W * roll_count + 100 * distinct_pattern_count`. This is a weighted objective, not strict lexicographic minimization. The constants are heuristic weights, not measured currency costs.
- MSHP's original Phase 1 scoring counted pieces rather than rolls. This release changes `sum(len(p) for p in patterns)` to `len(patterns)` and records that change in provenance. Historical numerical results should not be assumed identical after this fix.
- The historical MSHP linking bound evaluates to 1000 for each pattern. Large instances can be constrained artificially by this choice. MSHP also catches solver failures and can use greedy or Phase 1 fallbacks internally. These internal fallbacks are retained; the public interface validates final feasibility but does not certify how each pattern was produced.
- Original compatibility methods have misleading historical names: `solve_residual_ffd` calls SHP and `solve_advanced_column_generation` calls MSHP (or SHP without PuLP). The public CLI does not expose those aliases. Actual earlier residual algorithms live under `experiments/legacy/`.
- Some historical routines read solver values without robust status checks. Optional solver paths remain experimental until exercised with the intended solver environment.

## Metrics

Physical trim = sum over patterns of frequency * (W - sum of piece widths).

Trim rate = physical trim / (roll count * W). Surplus pieces are tracked separately; good trim alone does not imply low overproduction.

The analytical lower bound is the maximum of ceil(sum_i w_i d_i / W) and max_i ceil(d_i / floor(W/w_i)). It is not the CG LP bound. A positive difference from this bound does not establish how far a solution is from the true optimum.

## Data preparation

The original preprocessing notebook groups production records by grammage, week and product length, then sums demand by width. Keeping length fixed preserves the interpretation of demand as piece counts.

A separate exploratory branch aggregates demand * product_length across lengths. Those totals have length units and must not be passed directly as piece counts. The saved `test_cases.json` includes such very large values; it is deliberately excluded from the runnable demonstration. A consistent longitudinal production model or an explicitly justified conversion is needed to use that branch.

## Literature named in the source manuscript

- Gilmore & Gomory (1961), *A Linear Programming Approach to the Cutting-Stock Problem*.
- Haessler (1975), *Controlling Cutting Pattern Changes in One-Dimensional Trim Problems*.
- Martin, Moretti, Gomes-Ruggiero & Salles Neto (2018), *Modification of Haessler’s sequential heuristic procedure for the one-dimensional cutting stock problem with setup cost*.

These identify the research context. Algorithm names do not imply a verified line-by-line reproduction of the cited papers.
