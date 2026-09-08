# GOL: Game of Life on the GPU

<p align="center">
  <img src="images/header.gif" width="600" alt="Trails mode: a random soup evolving with phosphor-style persistence">
</p>

A fast, interactive Conway's Game of Life for macOS, rendered entirely with
Metal. The simulation runs in a GPU compute shader over a triple-buffered
16-bit cell grid, with an interactive camera (pan/zoom), a famous-rules picker
with per-neighbor B/S toggles, classic presets, four color palettes, a glow
effect, and three display modes.

## Features

- **GPU simulation**: the step kernel runs on the GPU (with a CPU fallback),
  so large grids stay interactive.
- **Aging cells**: every cell packs its age into 15 bits (up to 32,767
  generations), which the Age and Heatmap display modes visualize.
- **Viewport-scaled grid**: the computational window grows and shrinks with
  what you can see: zoom out and the grid expands to cover the viewport (up to
  a 1 GiB total memory budget, about 12 bytes per cell — tens of millions of
  cells), zoom in and it contracts. Patterns are preserved across resizes.
- **Camera**: scroll to zoom at the cursor, Option-drag (or middle-drag) to
  pan. At sub-pixel zoom levels the renderer switches to linear filtering with
  premultiplied alpha compositing so shrunken patterns stay smooth instead of
  aliasing.
- **Arbitrary rules**: a famous-rules drop-down (Life, HighLife, Day & Night,
  Seeds, Maze, Life w/o Death, Replicator, Diamoeba) plus per-neighbor B/S
  toggles define any birth/survival subset of the 8 neighbors (defaults to
  Conway's B3/S23).
- **Presets**: Glider, Blinker, Block, Beacon, Toad, Pentadecathlon, LWSS,
  R-Pentomino, Heptomino, Pulsar, and Acorn.
- **Display modes**: Age (color by cell age), Trails (phosphor-style
  persistence), and Heatmap (accumulated heat by age).
- **Color palettes**: Viridis, Inferno, Plasma, and Turbo — applied to the
  Age, Trails, and Heatmap modes.
- **Glow / bloom**: a subtle additive glow around live cells, toggleable from
  the toolbar or with `G`.
- **Live stats**: generation, population (with a live trend sparkline), max
  age, and FPS. Population and max age are computed on the GPU as a reduction
  inside the step kernel, so stats cost nothing at high generation rates.

## Screenshots

The **Display** button switches between three renderings of the same
R-Pentomino simulation:

| Age | Trails | Heatmap |
| :---: | :---: | :---: |
| ![Age mode](images/age.png) | ![Trails mode](images/trails.png) | ![Heatmap mode](images/heatmap.png) |

## Requirements

- macOS 12.0 or later, with the Xcode Command Line Tools
  (`xcode-select --install`), which provides `clang`, the Metal SDK, and
  `xcrun`. The binary is built as a universal (arm64 + x86_64) executable.

## Build and Run

```sh
make run        # builds bin/GOL.app (the app bundle) and launches it
```

Other targets:

```sh
make app       # build the bin/GOL.app bundle only (does not launch)
make run-bare  # build and run the bare bin/gol binary (no bundle)
make all       # build only (bin/gol, bin/default.metallib)
make test      # run the CPU and Metal test suites
make dist      # build a distributable dist/GOL-<version>.dmg
make clean     # remove bin/ and build/ (keeps dist/)
make distclean # also remove dist/
```

`bin/GOL.app` is a self-contained bundle: `Contents/MacOS/gol`,
`Contents/Resources/default.metallib`, and (once the icon exists) `GOL.icns`.
The bare `bin/gol` binary still works for scripting and the test suite.

The whole project compiles with `-Weverything` (C, Objective-C, and Metal)
and ships warning-free.

## Launch options

The app accepts command-line flags. Each flag has an environment-variable
equivalent, and a flag takes precedence over the corresponding variable:

| Flag | Env var | Meaning |
| --- | --- | --- |
| `--mode age\|trails\|heatmap` | `GOL_MODE` | Initial display mode (default Age) |
| `--preset NAME` | `GOL_PRESET` | Start with a preset instead of a random soup: `glider`, `blinker`, `block`, `beacon`, `toad`, `pentadecathlon`, `lwss`, `r-pentomino`, `heptomino`, `pulsar`, `acorn` |
| `--palette viridis\|inferno\|plasma\|turbo` | `GOL_PALETTE` | Initial color palette (default Viridis) |
| `--density 0.0-1.0` | `GOL_DENSITY` | Fill probability of the initial soup (default 0.2) |
| `--zoom PX` | `GOL_ZOOM` | Starting pixels per cell (default 6) |
| `--run 0\|1` | `GOL_RUN` | Start running (1) or paused (0, the default) |

Example:

```sh
./bin/gol --mode trails --preset acorn --run 1
```

## Controls

### Mouse (on the canvas)

| Action | Effect |
| --- | --- |
| Left drag | Paint with the active tool (Add or Erase) |
| Right drag | Paint with the opposite of the active tool |
| Scroll wheel | Zoom in/out at the cursor |
| Option + left drag, or middle drag | Pan the camera |
| Hover | Shows cell coordinates and age in the status bar |

### Keyboard

Menu key equivalents, so they work no matter which control has focus.

| Key | Effect |
| --- | --- |
| `Space` | Pause / resume |
| `.` | Advance one generation |
| `r` | Clear the grid |
| `z` | Randomize the grid |
| `g` | Toggle glow / bloom |
| `f` or `0` | Fit the pattern to the window |
| `1` / `2` / `3` | Display mode: Age / Trails / Heatmap |
| Cmd+`+` / Cmd+`-` | Zoom in / out about the view center |
| Cmd+Q | Quit |
| Cmd+W | Close the window (quits) |
| Cmd+M | Minimize |

### Toolbar

- **Rule**: a famous-rules drop-down plus a 2×9 grid of per-neighbor B/S
  toggles. The rule label shows the current rule (e.g. `B3/S23`).
- **Trend**: a live sparkline of population over recent generations.
- **Glow**: toggle the additive glow / bloom effect.
- **Density**: fill probability used by Randomize (default 20%).
- **Speed**: generations per second, 1-600 (default 30). At high rates several
  generations run per frame, so the maximum is no longer bound by refresh rate.
- **Preset**: drop-down of classic patterns; applying one clears the grid and
  places the pattern.
- **Go / Step / Random / Clear**: run/pause, advance one generation, seed a
  random soup, and wipe the grid.
- **Display**: Age, Trails, or Heatmap.
- **Palette**: Viridis, Inferno, Plasma, or Turbo.
- **Tool**: Add or Erase (right-drag always does the opposite).
- **Brush**: paint radius, 0-10 cells.
- **Fit**: rescale the view so the pattern (or whole grid) is visible.

## How It Works

- **Cell format**: one `uint16_t` per cell: bit 0 is alive, bits 1-15 are
  age. A single shared `MTLBuffer` holds three grid planes.
- **Triple buffering**: each generation steps plane *n* → plane *n+1 mod 3*
  in a compute kernel. The renderer always draws the newest plane, and a ring
   of command buffers tracks in-flight work so the CPU never touches a plane
   the GPU is still writing. All mutating operations (paint, randomize,
   clear, preset, resize) drain the command queue first.
- **Rendering**: a small render pass copies the live plane into a cell
  texture (only when it changes), then a full-screen fragment shader scales
  it to the viewport with camera transforms. Trails accumulate in a second
  single-channel texture via a fade-and-max pass, tinted by the active palette.
- **Resize safety**: grid resizes copy the old pattern into a persistent
  scratch buffer with origin-aware region copying, so panning/zooming never
  loses or duplicates pattern data.
- **CPU fallback**: when no Metal step pipeline is available, the same
  simulation runs on the CPU (`src/gol.c`), keeping behavior identical.

## Tests

```sh
make test
```

- `tests/gol_test.c`: rule helpers, cell packing, resize/region copying,
  randomization, and preset geometry.
- `tests/ref_test.c`: validates the CPU simulator against a reference
  implementation, including preset oscillation periods (block, blinker, toad,
  beacon, pentadecathlon) and glider/LWSS translation.
- `tests/metal_test.m`: runs the actual Metal step/trail kernels from the
  built `default.metallib` and compares them against the CPU reference.

## Project Layout

```
src/main.m        App, UI, camera, Metal setup, render loop (Objective-C)
src/gol.c/.h      CPU simulation helpers (rules, packing, resize, presets)
shaders.metal     Step, cell-texture, trail, and scale shaders
tests/            CPU reference tests and Metal kernel tests
packaging/        App bundle Info.plist (and icon source)
Makefile          Warning-free build, test, and run targets
```
