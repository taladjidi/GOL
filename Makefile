CC = clang
CFLAGS = -O3 -std=gnu11 -Weverything -Wno-poison-system-directories
OBJCFLAGS = -O3 -std=gnu11 -Weverything -fobjc-arc -fmodules -Wno-poison-system-directories
METALFLAGS = -Weverything -Wno-c++98-compat -Wno-deprecated
FRAMEWORKS = -framework Cocoa -framework Metal -framework MetalKit -framework QuartzCore
METAL_FRAMEWORKS = -framework Metal

VERSION = 0.1.0

all: bin/gol bin/default.metallib

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

bin/default.metallib: shaders.metal | bin build
	xcrun -sdk macosx metal $(METALFLAGS) -c shaders.metal -o build/shaders.air
	xcrun -sdk macosx metallib build/shaders.air -o $@

# Assemble the app bundle. The icns is copied only if 3.2 has generated it, so
# the bundle builds (icon-less) before the icon exists.
bin/GOL.app: bin/gol bin/default.metallib packaging/Info.plist | bin
	rm -rf $@
	mkdir -p $@/Contents/MacOS $@/Contents/Resources
	cp bin/gol $@/Contents/MacOS/
	cp bin/default.metallib $@/Contents/Resources/
	[ -f bin/GOL.icns ] && cp bin/GOL.icns $@/Contents/Resources/ || true
	sed 's/$$(VERSION)/$(VERSION)/g' packaging/Info.plist > $@/Contents/Info.plist
	codesign -s - --force --deep $@

app: bin/GOL.app

run: app
	open bin/GOL.app

run-bare: all
	./bin/gol

test: build/gol_test build/ref_test build/metal_test bin/default.metallib
	./build/gol_test
	./build/ref_test
	./build/metal_test bin/default.metallib

clean:
	rm -rf bin build

.PHONY: all app run run-bare clean test
