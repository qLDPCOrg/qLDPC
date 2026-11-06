# Offline lattice-surgery LER experiments

> [!WARNING]
> Lattice-surgery support in `qldpc` (`qldpc.experimental.surgery`) is experimental and unstable, and its implementation is still pending careful expert review.
> Its API and behavior may change without notice or deprecation.
> Validate results independently, and pin the `qldpc` version if you depend on them.

The [lattice-surgery notebook](../../examples/experimental/lattice_surgery.ipynb) intentionally uses small Monte Carlo budgets so it can run end to end in under ten minutes.
Use `collect_ler.py` for publication-scale data.

Each preset samples a lattice-surgery experiment and a quantum memory baseline under circuit-level depolarizing noise without idling errors, and decodes with BP+LSD.
The module docstring of `collect_ler.py` describes the presets, and its `DECODER_OPTIONS` dictionary sets the decoder options of each preset.
Run `python experiments/lattice_surgery/collect_ler.py --help` to see all options.

To try the script cheaply first, write a small run to a scratch directory; this usually takes under a minute:

```bash
python experiments/lattice_surgery/collect_ler.py \
    --preset steane-joint \
    --p-values 0.002 \
    --max-shots 1000 \
    --max-errors 10 \
    --workers 2 \
    --results-dir /tmp/ler
```

Publication-scale examples, which write their results to `experiments/lattice_surgery/results/`:

```bash
python experiments/lattice_surgery/collect_ler.py \
    --preset bb72 \
    --max-shots 1000000 \
    --max-errors 100 \
    --workers 8

python experiments/lattice_surgery/collect_ler.py \
    --preset bb18 \
    --max-shots 10000 \
    --max-errors 100 \
    --workers 8

python experiments/lattice_surgery/collect_ler.py \
    --preset steane-joint \
    --max-shots 1000000 \
    --max-errors 200 \
    --workers 8
```

By default, each run writes raw Sinter progress to `results/{preset}_{run_id}_progress.csv` and aggregate sample counts to `results/{preset}.csv`, where `results/` is the `--results-dir` (by default, the `results` directory beside the script, which Git ignores).
Pass `--resume` or `--output` to choose either file individually.
An interrupted run resumes from its progress file.
Task identity depends on the preset, tracked Git state, decoder settings, experiment kind, physical-error value, and round count.
Any commit or uncommitted edit to tracked files therefore starts new tasks from scratch, even with the same `--resume` file, so finish a run before changing tracked files.
Collection limits and worker count do not change task identity, so they may be adjusted when resuming.
Unrelated untracked files do not invalidate a resume.

The aggregate CSV file has one row per task and the following columns:

| Column | Meaning |
| --- | --- |
| `preset` | The `--preset` that produced the row |
| `kind` | The experiment: `surgery` or `memory` (`joint_surgery` or `single_memory` for `steane-joint`) |
| `p` | The physical error rate of circuit-level depolarizing noise (without idling errors) |
| `rounds` | The number of syndrome measurement rounds |
| `shots`, `errors`, `discards` | Sinter's counts of sampled shots, logical errors, and discarded shots |
| `seconds` | The CPU time spent sampling and decoding |
| `git_state` | The Git commit that produced the data, with a digest of any uncommitted changes |
| `decoder_settings` | A JSON object with the decoder name (`bp_lsd`) and its options |
| `collection_request` | The p-values, shot and error limits, and worker count of the request |

The logical error rate of a row is `errors / (shots - discards)`.
Rows for other p-values or legacy metadata in a shared resume file are not exported.
For `bb18`, surgery uses 15 rounds while memory uses 9.
The `rounds` column records that asymmetry; compare total failure probabilities with that distinction in mind, or normalize each curve to a per-cycle rate before plotting.

The `bb18` preset is particularly expensive: existing measurements take roughly 60–100 CPU-seconds per surgery shot.
Expect an overnight or multi-day run depending on worker count and error target.
