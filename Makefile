CC = clang
CFLAGS = -O3 -std=gnu11 -Weverything -Wno-poison-system-directories
OBJCFLAGS = -O3 -std=gnu11 -Weverything -fobjc-arc -fmodules -Wno-poison-system-directories
METALFLAGS = -Weverything -Wno-c++98-compat -Wno-deprecated
FRAMEWORKS = -framework Cocoa -framework Metal -framework MetalKit -framework QuartzCore
METAL_FRAMEWORKS = -framework Metal

all: bin/gol bin/shaders.metallib

bin:
	@mkdir -p bin

build:
	@mkdir -p build

build/gol.o: src/gol.c src/gol.h | build
	$(CC) $(CFLAGS) -Isrc -c src/gol.c -o $@

build/main.o: src/main.m src/gol.h | build
	$(CC) $(OBJCFLAGS) -Isrc -c src/main.m -o $@

build/gol_test.o: tests/gol_test.c src/gol.h | build
	$(CC) $(CFLAGS) -Isrc -c tests/gol_test.c -o $@

build/ref_test.o: tests/ref_test.c src/gol.h tests/refstep.h | build
	$(CC) $(CFLAGS) -Isrc -c tests/ref_test.c -o $@

build/gol_test: build/gol_test.o build/gol.o | build
	$(CC) $(CFLAGS) $^ -o $@

build/ref_test: build/ref_test.o build/gol.o | build
	$(CC) $(CFLAGS) $^ -o $@

build/metal_test.o: tests/metal_test.m src/gol.h tests/refstep.h | build
	$(CC) $(OBJCFLAGS) -Isrc -c tests/metal_test.m -o $@

build/metal_test: build/metal_test.o build/gol.o | build
	$(CC) $(OBJCFLAGS) $^ -o $@ $(METAL_FRAMEWORKS)

bin/gol: build/main.o build/gol.o | bin
	$(CC) $(OBJCFLAGS) $^ -o $@ $(FRAMEWORKS)

bin/shaders.metallib: shaders.metal | bin build
	xcrun -sdk macosx metal $(METALFLAGS) -c shaders.metal -o build/shaders.air
	xcrun -sdk macosx metallib build/shaders.air -o $@

run: all
	./bin/gol

test: build/gol_test build/ref_test build/metal_test bin/shaders.metallib
	./build/gol_test
	./build/ref_test
	./build/metal_test bin/shaders.metallib

clean:
	rm -rf bin build

.PHONY: all run clean test