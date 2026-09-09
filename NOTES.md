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
- **Headless offscreen render no-op (environment quirk, unresolved):** in a bare
  CLI process on this machine, offscreen *render-pass* encodes silently no-op —
  the command buffer completes (status 4) but the target texture stays all-zero,
  with no error. It persists with a full `NSApplication` + window + `MTKView`
  run loop and drawable presentation, for shared and private storage alike;
  compute passes are unaffected. The app's own render path (and its
  `--screenshot`) produces correct output, so the discrepancy is something in the
  app's setup vs a from-scratch harness, not display sleep (screenshots verified
  real content while `pmset` reported the display off). Consequence: shader-side
  UI like the 1.7 brush ring was verified through `--screenshot` output with a
  temporary env-gated cursor injection (all ring pixels landed on the expected
  circle, 1 px wide), not by live pointer interaction — no UI automation is
  available in this session.
- **Mipmap subpixel sampling (1.8) verified by same-soup A/B through
  `--screenshot`:** with a temporary env-gated zoom + RNG-seed injection, the
  same random soup rendered at 4 cells/px through the old single bilinear tap
  vs the new mipmapped tap. High-frequency energy (mean |pixel − neighbor
  mean| over the rendered region) dropped **54.6 → 30.7 per pixel** and the
  field went from isolated sparkle spikes to a smooth density map. The
  non-subpixel path is regression-free: a glider screenshot is byte-identical
  to the pre-change build. The interactive half of the plan's check (panning
  at deep zoom without shimmer) was not exercised — no UI automation.
- **Settings + window persistence (1.9) verified headlessly with temporary
  env-gated hooks:** a launch hook fired all four UI handlers and all five
  values (mode, palette, glow, speed, density) plus the moved window frame
  landed in the defaults domain; a fresh launch restored all five exactly.
  CLI precedence held: `--mode`/`--density` overrode the saved values without
  persisting them (domain unchanged after the run). The read path reaches the
  renderer: a `--screenshot` under saved heatmap/turbo/no-glow settings gave a
  blue-dominated turbo field (72% blue of bright pixels) vs the default
  viridis/age/glow render (47% green). The window-frame autosave round-tripped
  exactly once the display geometry was stable; when the headless display's
  height changed between runs (1084↔1117 pt) AppKit re-centered the window —
  its documented behavior when a saved frame's recorded screen no longer
  matches. Known gap: the View-menu display/palette items call the setters
  directly, so only the sidebar popups (and the glow button/menu item) persist.
  Live quit/relaunch via real UI interaction was not exercised — no UI automation.

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
- **Cursor tooltip** near the pointer instead of only in the side readout panel.
- **Non-wrapping edges** option (the torus is currently hard-coded in both the
  GPU kernel and `gol_step_cpu`).
