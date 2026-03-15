# Benchmark results

Performance logs from `cargo bench` for comparison over time.

## Log a new run

From the repo root:

```bash
{ echo "=== kriging-rs performance benchmarks ==="; echo "Run at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo ""; cargo bench 2>&1; } | tee "bench-results/$(date +%Y%m%d-%H%M%S).log"
```

## Compare runs

- Each log is named `YYYYMMDD-HHMMSS.log` and includes a header with the run time.
- Criterion prints time ranges (e.g. `[29.650 µs 31.460 µs 33.264 µs]`) and change vs baseline; use these to compare between log files.
- For side-by-side comparison: `diff bench-results/20260314-192628.log bench-results/<new>.log` or extract the "time:" and "change:" lines with `rg 'time:|change:' bench-results/*.log`.

## Variogram fitting baseline

Variogram fitting uses a 5×5×5 grid search. To save a Criterion baseline: `cargo bench -- --save-baseline grid_fit`; then compare with `cargo bench -- --baseline grid_fit`. You can keep baseline logs in `bench-results/` (or optionally at repo root; `*.log` is gitignored).
