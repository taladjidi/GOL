# Simulation Backends

A backend implements the `GOLEngine` vtable from
[`src/gol_engine.h`](../src/gol_engine.h). All backends operate on the same
`GOLGrid` and the same packed cell format.

## Common contract

### Cell packing

```text
uint16_t cell:
  bit 0      alive
  bits 1-15  age
```

### Plane layout

A grid contains `planeCount` planes. Each plane has capacity `planeCells`.

```text
plane base   = cells + plane * planeCells
active cell  = plane_base + y * w + x
```

`planeCells` is normally greater than or equal to `w * h`. The extra cells are
padding/capacity, not active simulation state.

### Step semantics

`stepSingle(engine, commandBuffer, readPlane, writePlane)` evaluates the rules
from `readPlane` and writes the next generation to `writePlane`.

`stepRange(engine, commandBuffer, startPlane, count)` evaluates `count`
generations around the plane ring:

```text
step 0: startPlane       -> startPlane + 1
step 1: startPlane + 1   -> startPlane + 2
...
all plane indices modulo planeCount
```

For GPU backends, `readPlane` and `writePlane` must refer to different planes
for a single step. Otherwise the shader may read and write the same memory
within one dispatch.

### Stats

`GOLStats` is one 16-byte block per plane:

```c
uint32_t alive;
uint32_t maxAge;
uint32_t pad0;
uint32_t pad1;
```

GPU backends may accumulate stats during the step. `stepRange` clears stats
before recording its dispatches. `stepSingle` does not clear stats; callers
that need a fresh single-step count should clear the relevant stats block first.

## Backend comparison

| Backend | File | Shader | Command buffer | Primary use |
| --- | --- | --- | --- | --- |
| CPU | `src/gol_engine_cpu.c` | none | ignored | reference implementation and fallback |
| Metal | `src/gol_engine_metal.m` | `shaders.metal` | `MTLCommandBuffer` | macOS app |
| Vulkan | `src/gol_engine_vulkan.c` | `shaders/gol_step.comp` | `VkCommandBuffer` | headless Linux/Windows compute |

## CPU backend

The CPU backend is the portable reference implementation.

- `stepSingle` / `stepRange` call `gol_step_cpu`.
- `commandBuffer` is ignored.
- Execution is synchronous.
- `count` scans the requested plane and returns alive/max-age values.

It is used by:

- unit tests
- reference tests
- the macOS app when a Metal step pipeline cannot be created

## Metal backend

The Metal backend is used by the macOS app.

It is created with non-owning references to app-created resources:

```c
GOLEngine *gol_engine_create_metal(GOLGrid *grid,
                                   void *stepPipeline,
                                   void *gridBuf,
                                   void *uniformsBuf,
                                   void *statsBuf);
```

The backend records a `MTLComputeCommandEncoder` into the provided command
buffer. The step kernel reads one grid plane offset, writes another grid plane
offset, reads uniforms, and updates the stats block for the write plane.

The Metal step shader uses a 16×16 threadgroup and a tiled torus halo to reduce
redundant global memory reads.

The app owns:

- `MTLDevice`
- `MTLCommandQueue`
- pipeline states
- grid buffer
- uniforms buffer
- stats buffer
- command buffers
- render textures

The Metal engine owns only its small internal implementation struct.

## Vulkan backend

The Vulkan backend is a headless compute implementation. It does not create a
window, swapchain, or Vulkan device. The host application or test creates the
Vulkan environment and passes ready-to-use objects through
`GOLVulkanEngineParams`.

```c
GOLEngine *gol_engine_create_vulkan(GOLGrid *grid,
                                    const GOLVulkanEngineParams *params);
```

The params include:

- `VkDevice`
- compute pipeline
- pipeline layout
- descriptor set layout
- descriptor pool
- grid `VkBuffer`
- stats `VkBuffer`
- mapped grid pointer
- mapped stats pointer
- buffer byte sizes
- maximum local workgroup size

The backend is non-owning: the caller creates and destroys the Vulkan objects.

### Buffer layout

The Vulkan step shader treats the packed grid as an array of 32-bit words:

```text
one uint32_t word = two adjacent uint16_t cells
```

This allows atomic updates to a 16-bit cell without atomically rewriting the
whole 32-bit word and clobbering its neighbor.

Requirements:

- `planeCells` must be even so every plane starts on a 32-bit word boundary.
- `planeCount` must be at least 2.
- `readPlane` and `writePlane` must differ for `stepSingle`.

The shader uses two storage buffers:

| Binding | Contents |
| --- | --- |
| 0 | all grid planes, as `uint` words |
| 1 | per-plane stats data |

Push constants provide:

```text
gridW
gridH
readPlane
writePlane
planeCells
birth
survival
```

The dispatch is:

```text
local size  = 8 × 8 × 1
group count = ceil(w / 8) × ceil(h / 8) × 1
```

Each invocation checks bounds before touching grid or stats memory.

### Vulkan synchronization

The engine inserts a buffer barrier around the compute dispatch:

```text
TRANSFER_SRC_ACCESS | TRANSFER_DST_ACCESS
  ->
SHADER_WRITE_ACCESS | SHADER_READ_ACCESS
```

This is intended for use inside a command buffer whose surrounding host code
already handles queue submission and any broader pipeline dependencies.

### Vulkan test

`tests/vulkan_test.c` is a headless harness that:

1. Creates a Vulkan instance and selects a compute-capable device.
2. Creates a queue, device, buffers, descriptor set, and compute pipeline.
3. Loads `build/gol_step.spv`.
4. Runs CPU and Vulkan steps over multiple rules, grid sizes, and densities.
5. Compares the resulting active cells and stats.

Run it with:

```sh
make vulkan-test
```

The test skips gracefully when no Vulkan compute device is available.

## Adding a new backend

A new backend should:

1. Implement the `GOLEngine` callbacks:
   - `stepSingle`
   - `stepRange`
   - `count`
   - `destroyImpl`
2. Provide a `gol_engine_create_<backend>` constructor.
3. Keep backend-specific resource creation outside the portable core.
4. Preserve the packed `uint16_t` cell format.
5. Respect plane indexing:
   - plane base = `plane * planeCells`
   - active cell = `y * w + x`
6. Add a test that compares the backend against `gol_step_cpu`.
7. Document any backend-specific alignment or synchronization requirements on
   this page.
