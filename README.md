# GOL: Game of Life on the GPU

[![CI](https://github.com/taladjidi/GOL/actions/workflows/ci.yml/badge.svg)](https://github.com/taladjidi/GOL/actions/workflows/ci.yml)

<p align="center">
  <img src="images/header.gif" width="600" alt="Trails mode: a random soup evolving with phosphor-style persistence">
</p>

A fast, interactive Conway's Game of Life for macOS, rendered with Metal. The
simulation runs in a GPU compute shader over a triple-buffered 16-bit cell
grid, with an interactive camera (pan/zoom), a famous-rules picker with
per-neighbor B/S toggles, classic presets, four color palettes, a glow effect,
and three display modes.

The simulation core is portable and backend-based: the macOS app uses Metal,
with a CPU fallback, while a separate headless Vulkan compute backend supports
Linux/Windows-style hosts.

## Documentation

- [Build guide](docs/build.md)
- [Architecture](docs/architecture.md)
- [Simulation backends](docs/backends.md)

## Features

- **GPU simulation**: the step kernel runs on the GPU (with a CPU fallback),
  so large grids stay interactive.
- **Aging cells**: every cell packs its age into 15 bits (up to 32,767
  generations), which the Age and Heatmap display modes visualize.
- **Viewport-scaled grid**: the computational window grows and shrinks with
  what you can see: zoom out and the grid expands to cover the viewport (up to
  a 1 GiB total memory budget, about 14 bytes per cell — tens of millions of
  cells), zoom in and it contracts. Patterns are preserved across resizes.
- **Camera**: scroll to zoom at the cursor, Option-drag (or middle-drag) to
  pan. At sub-pixel zoom levels the renderer samples a mipmapped cell texture
  with linear filtering and premultiplied alpha compositing, so a dense field
  shrinks into a smooth density map instead of aliasing into noise.
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

For the macOS app:

- macOS 12.0 or later, with the Xcode Command Line Tools
  (`xcode-select --install`), which provides `clang`, the Metal SDK, and
  `xcrun`. Both the app and the bare binary are built as universal
  (arm64 + x86_64) executables.

For the portable core and Vulkan backend, see
[Build](docs/build.md). The non-Darwin build path requires a C compiler and
`make`; Vulkan is required only for `make vulkan-test`.

## Build and Run

```sh
make run        # builds the app bundle and launches it
```

That's all you need for a first run: `make run` assembles `bin/GOL.app`,
a self-contained app bundle, and opens it.

All macOS targets:

| Target | What it does |
| --- | --- |
| `make run` | Build `bin/GOL.app` and launch it |
| `make app` | Build `bin/GOL.app` only (does not launch) |
| `make run-bare` | Build and run the bare `bin/gol` binary (no bundle) |
| `make all` *(default)* | Build `bin/gol` and `bin/default.metallib` only |
| `make test` | Build everything and run the CPU, reference, and Metal test suites |
| `make vulkan-test` | macOS stub; Vulkan tests run on non-Darwin platforms |
| `make dist` | Write a distributable `dist/GOL-<version>.dmg` |
| `make notarize` | Sign with a Developer ID and notarize (see [Distribution](#distribution)) |
| `make install` | Install the `gol` command to `/usr/local/bin` (also builds the app) |
| `make clean` | Remove `bin/` and `build/` (keeps `dist/`) |
| `make distclean` | Also remove `dist/` |

On Linux and other non-Darwin platforms, `make` builds the portable core only
and `make test` runs the CPU/reference suites. `make vulkan-test` builds and
runs the headless Vulkan backend test. See
[Build](docs/build.md) for platform details.

`bin/GOL.app` is a self-contained bundle: `Contents/MacOS/gol`,
`Contents/Resources/default.metallib`, and `GOL.icns`, ad-hoc signed so it
runs on the machine that built it. Inside the bundle the shader library loads
through Metal's default-library lookup; the bare `bin/gol` binary instead finds
`default.metallib` next to itself, so it keeps working for scripting and the
test suite.

The macOS build compiles C, Objective-C, and Metal with `-Weverything` and
ships warning-free. Every push is built and tested by CI on a macOS runner.
The non-Darwin portable core uses standard C warning flags and does not require
the macOS toolchain.

## Distribution

`make dist` writes `dist/GOL-<version>.dmg`: mount the disk image and drag
GOL into Applications.

The bundle is ad-hoc signed, which is enough to run it locally but not enough
to satisfy Gatekeeper on another Mac. To share the app, sign and notarize it
with your Apple Developer credentials:

```sh
make notarize SIGN_ID="Developer ID Application: Your Name (TEAMID)" \
              NOTARY_PROFILE=your-notary-profile
```

`SIGN_ID` is a "Developer ID Application" certificate in your keychain and
`NOTARY_PROFILE` is an `xcrun notarytool store-credentials` profile. The
target re-signs the bundle with the hardened runtime, submits it for
notarization, and staples the ticket, so the app opens without warning on any
macOS 12+ machine.

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
| `--screenshot PATH` | `GOL_SCREENSHOT` | Render a frame to a PNG at `PATH`, then quit (`--gen` and `--size` below apply) |
| `--gen N` | `GOL_GEN` | Generations to run before a screenshot (default 300) |
| `--size N` | `GOL_SIZE` | Screenshot output size, N×N pixels (default 1024) |

Example:

```sh
./bin/gol --mode trails --preset acorn --run 1
```

The flags work with the bundle too: `open bin/GOL.app --args --mode trails`.
Environment variables work with the bare binary, but `open` launches through
LaunchServices and does not pass the shell environment, so use `./bin/gol`
when launching by env var.

The app also remembers the last-used settings: display mode, palette, glow,
speed, and density changed in the UI are saved and restored on the next launch,
along with the window position and size. Explicit flags and environment
variables always take precedence over the remembered values.

## Controls

### Mouse (on the canvas)

| Action | Effect |
| --- | --- |
| Left drag | Paint with the active tool (Add or Erase) |
| Right drag | Paint with the opposite of the active tool |
| Two-finger scroll | Pan the camera |
| Pinch | Zoom in/out at the cursor |
| Cmd + scroll | Zoom in/out at the cursor |
| Option + left drag, or middle drag | Pan the camera |
| Hover | Shows cell coordinates and age in the readout panel right of the canvas, and a ring on the canvas previewing the brush |

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
- **Backend seam**: step and count dispatch is isolated behind `GOLEngine`
  (`src/gol_engine*`) with CPU, Metal, and Vulkan backends; `GOLGrid`
  (`src/gol_grid.c`) describes the shared plane buffer. See
  [Architecture](docs/architecture.md) and
  [Simulation Backends](docs/backends.md).

## Tests

On macOS:

```sh
make test
```

On non-Darwin platforms:

```sh
make test          # CPU and reference tests only
make vulkan-test   # headless Vulkan backend test
```

- `tests/gol_test.c`: rule helpers, cell packing, resize/region copying,
  randomization, and preset geometry.
- `tests/ref_test.c`: validates the CPU simulator against a reference
  implementation, including preset oscillation periods (block, blinker, toad,
  beacon, pentadecathlon) and glider/LWSS translation.
- `tests/metal_test.m`: runs the actual Metal step/trail kernels from the
  built `default.metallib` and compares them against the CPU reference.
- `tests/vulkan_test.c`: runs the Vulkan compute step and compares it against
  the CPU reference across rules, grid sizes, and densities.

## Project Layout

```
src/main.m              App, UI, camera, Metal setup, render loop (Objective-C)
src/gol.c/.h            CPU simulation helpers (rules, packing, resize, presets)
src/gol_grid.c/.h       Grid abstraction over the shared plane buffer
src/gol_engine*         Backend-agnostic step/count engine (CPU, Metal, Vulkan)
shaders.metal           Metal step, cell-texture, trail, and scale shaders
shaders/gol_step.comp   Vulkan compute step shader
tests/                  CPU, reference, Metal, and Vulkan tests
docs/                   Build, architecture, and backend documentation
packaging/              App bundle Info.plist and icon source
Makefile                Platform-aware build, test, run, and distribution targets
.github/                CI workflow (build + test on a macOS runner)
```
