# Build Guide

GOL has two build paths:

| Platform | Default `make` target | What it builds |
| --- | --- | --- |
| macOS | `bin/gol` + `bin/default.metallib` | The full interactive app, Metal shaders, and test binaries when requested |
| Linux / other non-Darwin | core object files only | The portable simulation core, without requiring a Vulkan SDK |

The non-Darwin path is designed for Linux and can also be used on Windows
through a Unix-style environment such as MSYS2.

## macOS

### Prerequisites

- macOS 12.0 or later
- Xcode Command Line Tools:

```sh
xcode-select --install
```

The toolchain provides `clang`, `xcrun`, the Metal SDK, `sips`, `iconutil`,
`codesign`, and `hdiutil`.

### Common commands

```sh
make run        # build bin/GOL.app and launch it
make test       # build and run CPU, reference, and Metal tests
make app        # build the app bundle without launching it
make run-bare   # build and run the bare bin/gol binary
```

### Targets

| Target | What it does |
| --- | --- |
| `make run` | Build `bin/GOL.app` and launch it |
| `make app` | Build `bin/GOL.app` only |
| `make run-bare` | Build and run the bare `bin/gol` binary |
| `make all` *(default)* | Build `bin/gol` and `bin/default.metallib` |
| `make test` | Build and run `gol_test`, `ref_test`, and `metal_test` |
| `make vulkan-test` | Prints a notice; Vulkan tests are non-Darwin only |
| `make dist` | Write `dist/GOL-<version>.dmg` |
| `make notarize` | Sign with a Developer ID and notarize |
| `make install` | Install the `gol` command to `/usr/local/bin` |
| `make clean` | Remove `bin/` and `build/` |
| `make distclean` | Also remove `dist/` |

The macOS build compiles C, Objective-C, and Metal with `-Weverything` and is
expected to be warning-free.

## Linux and other non-Darwin platforms

### Portable core

The minimum build requires only a C11 compiler and `make`:

```sh
make
make test
```

`make test` runs:

- `tests/gol_test.c` — rules, packing, resize/region copying, randomization,
  and presets.
- `tests/ref_test.c` — CPU simulator validation against a reference
  implementation.

This path does **not** build the macOS app and does **not** require Vulkan.

### Vulkan compute backend

To build and run the headless Vulkan implementation test:

```sh
make vulkan-test
```

This builds:

- `build/gol_step.spv` from `shaders/gol_step.comp` using `glslc`
- `build/vulkan_test`, a headless Vulkan test harness
- the required portable core objects

Then it runs:

```sh
./build/vulkan_test build/gol_step.spv
```

If no Vulkan compute device is available, the test prints a diagnostic and
skips gracefully.

### Vulkan SDK detection

The Makefile looks for Vulkan in this order:

1. `pkg-config`:
   - `pkg-config --cflags vulkan`
   - `pkg-config --libs vulkan`
2. If no library flags are found, it falls back to `-lvulkan`.
3. If `VULKAN_SDK` is set, it adds:
   - `-I$(VULKAN_SDK)/include`
   - `-L$(VULKAN_SDK)/lib -lvulkan` when the linker flags are still the
     fallback value

Examples:

```sh
# Use a SDK in a non-default location
VULKAN_SDK=/opt/vulkan make vulkan-test

# Use a specific glslc
GLSLC=/opt/vulkan/bin/glslc make vulkan-test
```

Typical Linux prerequisites for the Vulkan test are therefore:

- a C compiler
- `make`
- `pkg-config` or a `VULKAN_SDK` environment variable
- the Vulkan loader development package
- `glslc` from the Vulkan Tools / SPIR-V tools

## CI

The current GitHub Actions workflow builds and tests the macOS app on a
`macos-14` runner:

```sh
make
make test
```

Linux/Vulkan validation is currently performed locally with:

```sh
make vulkan-test
```

A Linux CI job can be added once the target SDK and driver setup are settled.

## Output layout

Build artifacts are written to gitignored directories:

```text
bin/     app bundle, bare binary, metallib, icon
build/   objects, test binaries, SPIR-V, intermediate artifacts
dist/    distributable archives after make dist
```

`make clean` removes `bin/` and `build/`; `make distclean` also removes
`dist/`.
