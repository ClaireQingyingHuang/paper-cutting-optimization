"""Validated public interface; FFD uses only the Python standard library."""
from collections import Counter
from math import ceil
from time import perf_counter


def validate_input(roll_width, demands):
    if type(roll_width) is not int or roll_width <= 0:
        raise ValueError('roll_width must be a positive integer')
    if not isinstance(demands, dict) or not demands:
        raise ValueError('demands must be a nonempty mapping')
    for width, quantity in demands.items():
        if type(width) is not int or not 0 < width <= roll_width:
            raise ValueError('every width must be an integer in [1, roll_width]')
        if type(quantity) is not int or quantity < 0:
            raise ValueError('quantities must be nonnegative integer piece counts')
    if not any(demands.values()):
        raise ValueError('at least one positive demand is required')


def summarize(roll_width, demands, bins, method, elapsed):
    """Independently check feasibility and count trim, production and surplus."""
    produced = Counter()
    patterns = Counter()
    for pieces in bins:
        pieces = tuple(int(w) for w in pieces)
        if not pieces or any(w not in demands for w in pieces) or sum(pieces) > roll_width:
            raise ValueError('solver returned an invalid cutting pattern')
        produced.update(pieces)
        patterns[tuple(sorted(pieces, reverse=True))] += 1
    if any(produced[w] < q for w, q in demands.items()):
        raise ValueError('solver returned an incomplete plan')
    n_rolls = sum(patterns.values())
    required_width = sum(w * q for w, q in demands.items())
    trim = n_rolls * roll_width - sum(w * q for w, q in produced.items())
    lower_bound = max(ceil(required_width / roll_width),
                      max(ceil(q / (roll_width // w)) for w, q in demands.items()))
    return {
        'method': method, 'n_rolls': n_rolls, 'n_patterns': len(patterns),
        'trim_loss': trim, 'trim_loss_rate': trim / (n_rolls * roll_width),
        'lower_bound': lower_bound, 'rolls_above_lower_bound': n_rolls - lower_bound,
        'solve_time_seconds': elapsed,
        'overproduction': {str(w): produced[w] - q for w, q in demands.items()},
        'patterns': [{'pieces': list(p), 'frequency': count,
                      'trim_per_roll': roll_width - sum(p)} for p, count in patterns.items()],
    }


def solve(roll_width, demands, method='ffd'):
    """Run FFD or an optional research algorithm; never silently rename fallbacks."""
    validate_input(roll_width, demands)
    demands = {w: q for w, q in demands.items() if q}
    # Historical implementations expand pieces/rolls in memory.
    if sum(demands.values()) > 100_000:
        raise ValueError('demo limit is 100,000 pieces; check demand units or use a batched solver')
    start = perf_counter()
    if method == 'ffd':
        bins, remaining = [], []
        for width in sorted(demands, reverse=True):
            for _ in range(demands[width]):
                for i, available in enumerate(remaining):
                    if available >= width:
                        bins[i].append(width)
                        remaining[i] -= width
                        break
                else:
                    bins.append([width])
                    remaining.append(roll_width - width)
    elif method in ('shp', 'mshp', 'cg'):
        try:
            from .research import CuttingStockAlgorithms, HAS_PULP, grb
        except ImportError as exc:
            raise RuntimeError('Research algorithms need optional dependencies; see README.') from exc
        if method == 'cg' and grb is None:
            raise RuntimeError('CG requires gurobipy and a usable Gurobi license.')
        if method == 'mshp' and not HAS_PULP:
            raise RuntimeError('MSHP requires PuLP and CBC.')
        solver = CuttingStockAlgorithms(roll_width)
        runner = {'shp': solver.solve_haessler_shp, 'mshp': solver.solve_modified_shp,
                  'cg': solver.solve_column_generation}[method]
        bins = runner(demands)['patterns']
    else:
        raise ValueError('method must be ffd, shp, mshp or cg')
    return summarize(roll_width, demands, bins, method, perf_counter() - start)
