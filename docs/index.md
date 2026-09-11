# GOL Documentation

GOL is an interactive GPU Game of Life. The shipped app is a macOS Metal
application, but the simulation core is now a portable layer with pluggable
backends:

- **CPU** — reference implementation and fallback.
- **Metal** — the backend used by the macOS app.
- **Vulkan** — a headless compute backend for Linux/Windows-style hosts.

The documentation is split into focused pages:

| Page | Contents |
| --- | --- |
| [Build](build.md) | Platform support, prerequisites, Make targets, Vulkan SDK detection, and test commands |
| [Architecture](architecture.md) | The split between the app/renderer layer and the portable simulation core |
| [Simulation Backends](backends.md) | CPU, Metal, and Vulkan backend contracts, data layout, and synchronization rules |

The [README](../README.md) remains the project overview: features, screenshots,
controls, launch options, and distribution.

## Current status

| Area | Status |
| --- | --- |
| macOS app | Implemented and shipped through `bin/GOL.app` |
| Metal simulation backend | Implemented and used by the app |
| CPU fallback | Implemented and used when a Metal step pipeline is unavailable |
| Portable `GOLGrid` / `GOLEngine` core | Implemented in plain C |
| Vulkan compute backend | Implemented as a headless `GOLEngine` backend; not yet wired into an interactive window |
| CI | Builds and tests the macOS app on every push |

## Reading order

1. Skim [Build](build.md) to choose the right command for your platform.
2. Read [Architecture](architecture.md) to understand why the project is split
   into renderer, grid, engine, and backend layers.
3. Read [Simulation Backends](backends.md) before changing shaders, buffer
   layout, or adding another backend.
