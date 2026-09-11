# Architecture

GOL is split into two main layers:

1. **Presentation layer** — the macOS app: window, UI, camera, rendering,
   input, and frame pacing.
2. **Simulation layer** — a portable C core: grid storage, rules, cell
   packing, and a backend-facing engine interface.

```text
┌─────────────────────────────────────────────────────────────┐
│ macOS app (src/main.m, shaders.metal)                      │
│  AppKit UI · MTKView · camera · render passes · Metal setup │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ GOLEngine API + GOLGrid
                            │
┌───────────────────────────▼─────────────────────────────────┐
│ Portable simulation core (plain C)                          │
│  gol.h/.c        rules, packing, CPU step, presets          │
│  gol_grid.*      plane buffer description                  │
│  gol_engine.*    backend vtable and shared helpers          │
└───────┬───────────────────┬─────────────────────┬───────────┘
        │                   │                     │
   CPU backend          Metal backend         Vulkan backend
 gol_engine_cpu.c    gol_engine_metal.m    gol_engine_vulkan.c
                    shaders.metal         shaders/gol_step.comp
```

The app depends on the simulation core. The simulation core does not depend on
AppKit, Metal, or Vulkan.

## Presentation layer

The macOS app lives mainly in `src/main.m`. It owns:

- the Cocoa window, menus, toolbar, and readout panels
- the `MTKView` draw loop
- camera state: zoom, pan, fit, and cursor position
- paint/randomize/clear/preset operations
- Metal device, queue, pipeline states, buffers, and textures
- display passes: cell texture, trails, heatmap, glow, and full-screen scaling
- the triple-buffered command-buffer ring used to keep rendering ahead of or
  behind simulation safely

The Metal renderer reads the newest completed grid plane and converts it into
display textures. It does not implement the Game of Life rules itself; rule
evaluation is delegated to the simulation engine.

## Simulation core

### `GOLGrid`

`src/gol_grid.h` describes a plane-based grid:

```c
typedef struct GOLGrid {
    size_t planeCells;
    uint16_t *cells;
    void *gpuHandle;
    int w;
    int h;
    int planeCount;
    uint32_t flags;
} GOLGrid;
```

Fields:

- `w`, `h` — active simulation size.
- `planeCells` — capacity/row stride for one plane, in cells.
- `cells` — base pointer to all planes.
- `gpuHandle` — optional backend-specific resource handle.
- `planeCount` — number of planes, normally 3 for the app.
- `flags` — currently `GOL_GRID_FLAG_OWNS_CELLS`, meaning `gol_grid_free`
  frees `cells`.

A plane starts at:

```text
cells + plane * planeCells
```

Active cells inside a plane are indexed:

```text
y * w + x
```

The remaining cells up to `planeCells` are padding/capacity. This lets the app
allocate one large buffer for the maximum supported grid and later shrink the
active `w × h` without reallocating.

### `GOLEngine`

`src/gol_engine.h` defines the backend seam:

```c
struct GOLEngine {
    GOLGrid *grid;
    void *impl;
    GOLRules rules;
    uint32_t flags;
    bool (*stepSingle)(GOLEngine *engine, void *commandBuffer,
                       uint32_t readPlane, uint32_t writePlane);
    bool (*stepRange)(GOLEngine *engine, void *commandBuffer,
                      uint32_t startPlane, uint32_t count);
    void (*count)(GOLEngine *engine, uint32_t plane,
                  uint32_t *alive, uint32_t *maxAge);
    void (*destroyImpl)(GOLEngine *engine);
};
```

The important design choice is that `commandBuffer` is a `void *`. Each backend
interprets it differently:

| Backend | `commandBuffer` means |
| --- | --- |
| CPU | ignored |
| Metal | `id<MTLCommandBuffer>` |
| Vulkan | `VkCommandBuffer` |

This lets the same engine API record asynchronous GPU work or execute
synchronously without the core knowing which graphics API is in use.

### `GOLStats`

Each plane has a 16-byte stats block:

```c
typedef struct {
    uint32_t alive;
    uint32_t maxAge;
    uint32_t pad0;
    uint32_t pad1;
} GOLStats;
```

GPU backends can update stats inside the step kernel. The CPU backend computes
stats by scanning the plane when `count` is called.

## Cell format

Every cell is one `uint16_t`:

```text
bit 0      alive
bits 1-15  age, 0..32767
```

This compact format is shared by CPU, Metal, and Vulkan. It gives enough age
range for the Age, Trails, and Heatmap display modes while keeping the grid
memory small.

## Frame flow in the macOS app

`MTKView` calls `drawInMTKView:`, which calls `tick`. A tick does roughly:

1. Apply any queued paint edits.
2. Update Metal uniforms for camera, display mode, palette, glow, and cursor.
3. Decide how many generations to advance this frame based on the speed slider.
4. If the Metal step pipeline is available and the relevant command-buffer ring
   slots are free, record one or more simulation steps through `GOLEngine`.
5. Render the newest completed plane into the cell texture.
6. Run trails/heatmap/glow/full-screen passes as needed.
7. Present the drawable and commit the command buffer.

The app uses three grid planes and three command-buffer slots. A plane must not
be reused while an older command buffer that touched it is still in flight. If
the GPU is behind, the app renders the latest completed plane but skips
simulation for that frame instead of racing the GPU.

## Backend ownership rules

The current backends are deliberately non-owning for external GPU resources:

- The Metal engine stores unretained references to the pipeline and buffers
  created by the app.
- The Vulkan engine stores caller-provided Vulkan handles and mapped pointers.
- `gol_engine_destroy` frees only the engine's internal implementation struct.

The host application or test harness is responsible for creating and destroying
GPU resources in the correct order. This keeps the portable core simple and
avoids requiring a full graphics API teardown contract.

## Why this split exists

The original app was a single Metal-based program. The backend seam was
introduced to make the simulation portable:

- The CPU implementation remains the reference oracle for tests.
- Metal remains the interactive macOS backend.
- Vulkan can validate the same grid/rule semantics on Linux/Windows without
  pulling AppKit or Metal into the portable core.
- Future hosts can use the Vulkan backend from a headless tool, a benchmark,
  or eventually a native windowed application.

The renderer and simulation are allowed to evolve independently as long as
they agree on the `GOLGrid`, `GOLEngine`, cell packing, and plane-indexing
contracts described in [Simulation Backends](backends.md).
