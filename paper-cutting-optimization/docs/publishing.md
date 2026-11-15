# Publishing this repository

Suggested name: `paper-cutting-optimization`

Suggested description: `One-dimensional paper cutting-stock optimization: FFD, column generation, and sequential heuristics for material-use and setup tradeoffs.`

Suggested topics: `operations-research`, `cutting-stock`, `optimization`, `supply-chain`, `column-generation`, `python`.

The prepared folder is a local Git repository. No GitHub remote has been created or pushed. A conservative first destination is a private repository while code ownership and the experimental implementation are reviewed.

With GitHub CLI installed and authenticated, run from this folder:

```bash
git add .
git commit -m "Organize paper cutting optimization research project"
gh repo create paper-cutting-optimization --private --source=. --remote=origin --push
```

Alternatively create an empty repository in GitHub, copy its actual remote URL, add it as `origin`, and push the `main` branch. Do not copy the original factory folder into this repository or add reference repositories as if they were original work.

Before making it public, decide the license and code ownership, verify optional solver paths, and decide whether to link an authorized published article. Do not upload unpublished industrial orders or claim the synthetic example reproduces the manuscript.
