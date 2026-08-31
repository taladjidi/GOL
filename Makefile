CC = clang
CFLAGS = -O2 -Wall
OBJCFLAGS = -O2 -Wall -fobjc-arc -fmodules
FRAMEWORKS = -framework Cocoa -framework Metal -framework MetalKit -framework QuartzCore

all: bin/gol bin/shaders.metallib

bin:
	@mkdir -p bin

build:
	@mkdir -p build

tests: build/gol_test.o build/gol.o
	$(CC) $(CFLAGS) build/gol_test.o build/gol.o -o build/gol_test

build/gol_test.o: tests/gol_test.c src/gol.h
	$(CC) $(CFLAGS) -Isrc -c tests/gol_test.c -o build/gol_test.o

bin/gol: src/main.m src/gol.c src/gol.h | bin build
	$(CC) $(CFLAGS) -c src/gol.c -o build/gol.o
	$(CC) $(OBJCFLAGS) -c src/main.m -o build/main.o
	$(CC) $(OBJCFLAGS) build/main.o build/gol.o -o $@ $(FRAMEWORKS)

bin/shaders.metallib: shaders.metal | bin build
	xcrun -sdk macosx metal -c shaders.metal -o build/shaders.air
	xcrun -sdk macosx metallib build/shaders.air -o $@

run: all
	./bin/gol

test: tests
	./build/gol_test

clean:
	rm -rf bin build

.PHONY: all run clean test