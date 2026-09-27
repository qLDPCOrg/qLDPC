# Offline lattice-surgery LER experiments

The documentation notebook intentionally uses small Monte Carlo budgets so it can run end to end
in under ten minutes. Use `collect_ler.py` for publication-scale data.

Examples:

```bash
python experiments/lattice_surgery/collect_ler.py \
    --preset bb72 \
    --max-shots 1000000 \
    --max-errors 100 \
    --workers 8 \
    --output bb72_results.csv

python experiments/lattice_surgery/collect_ler.py \
    --preset bb18 \
    --max-shots 10000 \
    --max-errors 100 \
    --workers 8 \
    --output bb18_results.csv

python experiments/lattice_surgery/collect_ler.py \
    --preset steane-joint \
    --max-shots 1000000 \
    --max-errors 200 \
    --workers 8 \
    --output steane_joint_results.csv
```

Each preset writes raw Sinter progress to a run-specific file beside the script unless `--resume`
is supplied. Progress files are ignored by Git and may be resumed after interruption. Run identity,
Git state, physical-error values, collection limits, worker count, and decoder settings are carried
in each task's metadata, so rows from another run cannot be silently relabeled or exported.

The requested `--output` is a small aggregate CSV with those settings and the producing Git state.
For `bb18`, surgery uses 15 rounds while memory uses 9. The `rounds` column records that asymmetry;
compare total failure probabilities with that distinction in mind, or normalize each curve to a
per-cycle rate before plotting.

The `bb18` preset is particularly expensive: existing measurements take roughly 60–100 CPU-seconds
per surgery shot. Expect an overnight or multi-day run depending on worker count and error target.
