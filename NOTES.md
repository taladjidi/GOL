# GOL — Backlog & Design Notes

Scratch file for ideas we deliberately deferred. Anything here is *not* part of
the current build; it's a menu of follow-up work, roughly ordered by how much it
would help.

## Done in the "prettier + funner" pass

- **Zoom / memory**: deep zoom-out now works. `MIN_CELL_PX` lowered 1.0 → 0.1,
  `MAX_CELL_PX` raised 64 → 128, and the grid cap is derived from a 1 GB total
  memory budget instead of a hard 4M-cell constant.
- **Rules**: the two range sliders are gone, replaced by a famous-rules drop-down
  plus per-neighbor B/S toggles (so non-contiguous rules like Replicator,
  Day&Night, and Seeds are reachable).
- **Graphics**: multiple color palettes (Age/Heatmap), softer rounded cell edges,
  and a toggleable glow/bloom on live cells (radial falloff in `fs_scale`, G key).
- **Gameplay**: live population sparkline, more presets (Pulsar, Acorn, …), a
  Randomize button, and zoom-to-pattern (Fit now frames the live cells).

## Measured

- **Step-kernel cost at the memory cap** (Apple M3 Max, `Mac15,9`; macOS 26.6.2,
  build 25G83). One `gol_step` dispatch over the full cap grid
  (12116×7384 ≈ 89.5M cells), timed headlessly in `tests/metal_test.m`
  (`GOL_BENCH_CAP=1`) as wall-clock around `commit` + `waitUntilCompleted`,
  best of 10 after 3 warmup steps:
  - Untiled (one thread per cell, nine strided global neighbor loads): **~7.2 ms**.
  - Tiled (16×16 threadgroup loading an 18×18 torus halo into threadgroup
    memory, `shaders.metal`): **~5.4 ms** — a ~25% win, so tiling was kept.
- **Implication:** at the cap a single step is ~5.4 ms, i.e. a GPU ceiling of
  ~185 gen/s — so the 600 gen/s slider max is compute-bound (unreachable) at the
  absolute cap; only smaller grids can sustain it.
- **Deferred:** the live in-app 600 gen/s compute-vs-presentation split under
  Instruments. The display session was asleep/off-screen during this work, so the
  headless number above stands in for the decision gate.

## Deferred — big architectural change

- **World / viewport decoupling.** Today the grid *is* the viewport: it resizes
  with the camera (`updateGridForPixelSize`) and is capped by memory. The "real"
  fix for unbounded zoom-out is to keep the world larger than the viewport and
  only simulate what's visible. **Not** doable by naive spatial chunking, because
  in a torus/CA every cell can depend on every other cell (gliders, signals).
  Needs an explicit "world" vs "visible window" split, plus a decision on whether
  the world is finite/torus/infinite. Park until the naive budget is measured to
  be insufficient.

## Deferred — more juice / graphics

- **Birth / death flash**: briefly highlight cells that just changed state.
- **Background grid lines + vignette** for a more "lab" look.
- **Tunable trail persistence**: the fade is hard-coded to `0.94` in
  `shaders.metal:295`; expose it as a control.

## Deferred — gameplay / UX

- **Save / Load** patterns (RLE and/or PNG).
- **Cursor tooltip** near the pointer instead of only in the status bar.
- **Non-wrapping edges** option (the torus is currently hard-coded in both the
  GPU kernel and `gol_step_cpu`).
