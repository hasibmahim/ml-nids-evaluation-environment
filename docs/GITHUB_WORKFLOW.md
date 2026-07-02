# GitHub Workflow

Use `main` as the stable branch.

Create feature branches for changes:

```bash
git checkout -b feature/experiment-runner
```

Open pull requests for review before merging into `main`. For a two-author thesis, both authors should review important changes, especially methodology and result interpretation.

Use meaningful commits:

```bash
git add README.md src tests configs docs
git commit -m "Add ML-NIDS experiment environment"
```

Never commit raw datasets, processed datasets, generated results, trained models, logs, or large CSV/ZIP/Parquet files.
