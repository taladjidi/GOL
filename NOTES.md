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
  `shaders.metal:269`; expose it as a control.
- **Palette for trails** (currently trails reuse the cell color).

## Deferred — gameplay / UX

- **Step button** (advance exactly one generation, for studying oscillators).
  Not requested in the last pass; the `Space`/`R`/`Z`/`G`/`F` keys exist but
  there's no single-step button.
- **Save / Load** patterns (RLE and/or PNG).
- **Cursor tooltip** near the pointer instead of only in the status bar.
- **Non-wrapping edges** option (the torus is currently hard-coded in both the
  GPU kernel and `gol_step_cpu`).

## Deferred — code quality (correctness/robustness, not features)

- **GPU population count.** `gol_count_alive` runs on the CPU every frame
  (`main.m:2372`) over the whole plane — a full readback-free CPU loop over up to
  ~50M cells. Should be a compute reduction, or at least throttled / sampled.
- **Better RNG.** `gol_randomize` uses `srand(time(NULL))` + `rand()`
  (`gol.c:32`). Fine for now; a proper PRNG (e.g. splitmix64) would be more
  reproducible and higher quality.
- **Toolbar layout.** The toolbar is laid out with a manual x-cursor
  (`main.m:1816+`). Fragile to label-width changes. An `NSStackView`/Auto Layout
  pass would make it robust.
