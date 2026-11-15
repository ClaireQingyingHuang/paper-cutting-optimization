import argparse
import json
from pathlib import Path
from .core import solve


def main():
    parser = argparse.ArgumentParser(description='Solve a one-dimensional paper cutting instance.')
    parser.add_argument('input', type=Path)
    parser.add_argument('--method', choices=['ffd', 'shp', 'mshp', 'cg'], default='ffd')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text())
        raw = data['demands']
        if not isinstance(raw, dict):
            raise ValueError('demands must be a JSON object')
        demands = {int(w): q for w, q in raw.items()}
        if len(demands) != len(raw):
            raise ValueError('duplicate widths after integer conversion')
        result = solve(data['roll_width'], demands, args.method)
        text = json.dumps(result, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + '\n')
        print(text)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
