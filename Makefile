CC = clang
CFLAGS = -O3 -std=gnu11 -Weverything -Wno-poison-system-directories -arch arm64 -arch x86_64 -mmacosx-version-min=12.0 -MMD -MP
OBJCFLAGS = -O3 -std=gnu11 -Weverything -fobjc-arc -fmodules -Wno-poison-system-directories -arch arm64 -arch x86_64 -mmacosx-version-min=12.0 -MMD -MP
METALFLAGS = -Weverything -Wno-c++98-compat -Wno-deprecated -mmacosx-version-min=12.0
FRAMEWORKS = -framework Cocoa -framework Metal -framework MetalKit -framework QuartzCore
METAL_FRAMEWORKS = -framework Metal

VERSION = 0.1.0

.DEFAULT_GOAL := all

all: bin/gol bin/default.metallib

bin:
	@mkdir -p bin

build:
	@mkdir -p build

build/gol.o: src/gol.c | build
	$(CC) $(CFLAGS) -Isrc -c src/gol.c -o $@

build/main.o: src/main.m | build
	$(CC) $(OBJCFLAGS) -Isrc -c src/main.m -o $@

build/gol_test.o: tests/gol_test.c | build
	$(CC) $(CFLAGS) -Isrc -c tests/gol_test.c -o $@

build/ref_test.o: tests/ref_test.c | build
	$(CC) $(CFLAGS) -Isrc -c tests/ref_test.c -o $@

build/gol_test: build/gol_test.o build/gol.o | build
	$(CC) $(CFLAGS) $^ -o $@

build/ref_test: build/ref_test.o build/gol.o | build
	$(CC) $(CFLAGS) $^ -o $@

build/metal_test.o: tests/metal_test.m | build
	$(CC) $(OBJCFLAGS) -Isrc -c tests/metal_test.m -o $@

build/metal_test: build/metal_test.o build/gol.o | build
	$(CC) $(OBJCFLAGS) $^ -o $@ $(METAL_FRAMEWORKS)

bin/gol: build/main.o build/gol.o | bin
	$(CC) $(OBJCFLAGS) $^ -o $@ $(FRAMEWORKS)

bin/default.metallib: shaders.metal | bin build
	xcrun -sdk macosx metal $(METALFLAGS) -c shaders.metal -o build/shaders.air
	xcrun -sdk macosx metallib build/shaders.air -o $@

# 3.2: build the app icon from the committed PNG. sips resizes to each iconset
# size, iconutil packs them into the .icns (a build artifact in gitignored bin/).
bin/GOL.icns: packaging/icon.png | bin build
	rm -rf build/icon.iconset
	mkdir -p build/icon.iconset
	sips -z 16 16     packaging/icon.png --out build/icon.iconset/icon_16x16.png
	sips -z 32 32     packaging/icon.png --out build/icon.iconset/icon_16x16@2x.png
	sips -z 32 32     packaging/icon.png --out build/icon.iconset/icon_32x32.png
	sips -z 64 64     packaging/icon.png --out build/icon.iconset/icon_32x32@2x.png
	sips -z 128 128   packaging/icon.png --out build/icon.iconset/icon_128x128.png
	sips -z 256 256   packaging/icon.png --out build/icon.iconset/icon_128x128@2x.png
	sips -z 256 256   packaging/icon.png --out build/icon.iconset/icon_256x256.png
	sips -z 512 512   packaging/icon.png --out build/icon.iconset/icon_256x256@2x.png
	sips -z 1024 1024 packaging/icon.png --out build/icon.iconset/icon_512x512@2x.png
	iconutil -c icns build/icon.iconset -o $@

# Assemble the app bundle (icon included).
bin/GOL.app: bin/gol bin/default.metallib packaging/Info.plist bin/GOL.icns | bin
	rm -rf $@
	mkdir -p $@/Contents/MacOS $@/Contents/Resources
	cp bin/gol $@/Contents/MacOS/
	cp bin/default.metallib $@/Contents/Resources/
	cp bin/GOL.icns $@/Contents/Resources/
	sed 's/$$(VERSION)/$(VERSION)/g' packaging/Info.plist > $@/Contents/Info.plist
	codesign -s - --force --deep $@

app: bin/GOL.app

run: app
	open bin/GOL.app

run-bare: all
	./bin/gol

# 3.3: Developer ID signing + notarization. Requires SIGN_ID (a "Developer ID
# Application" identity) and NOTARY_PROFILE (a notarytool keychain profile).
notarize: app
	@if [ -z "$(SIGN_ID)" ] || [ -z "$(NOTARY_PROFILE)" ]; then \
		echo "set SIGN_ID and NOTARY_PROFILE"; \
		exit 1; \
	fi
	codesign --force --options runtime --timestamp -s "$(SIGN_ID)" bin/GOL.app
	ditto -c -k --keepParent bin/GOL.app build/GOL.zip
	xcrun notarytool submit build/GOL.zip --keychain-profile "$(NOTARY_PROFILE)" --wait
	xcrun stapler staple bin/GOL.app

# 3.5: distribution archive. Stage the app next to an Applications symlink,
# then write a compressed read-only DMG to dist/ (a gitignored output dir).
dist: app
	rm -rf build/dmg && mkdir -p build/dmg
	cp -R bin/GOL.app build/dmg/
	ln -s /Applications build/dmg/Applications
	mkdir -p dist
	hdiutil create -volname "GOL" -srcfolder build/dmg -ov -format UDZO dist/GOL-$(VERSION).dmg
	rm -rf build/dmg

test: all build/gol_test build/ref_test build/metal_test bin/default.metallib
	./build/gol_test
	./build/ref_test
	./build/metal_test bin/default.metallib

clean:
	rm -rf bin build

# clean removes build outputs but keeps dist/; distclean also drops the archives.
distclean: clean
	rm -rf dist

# Install the bare binary to /usr/local/bin (also builds the app bundle).
install: app /usr/local/bin/gol

/usr/local/bin/gol: bin/gol
	cp $< $@

.PHONY: all app run run-bare notarize dist clean distclean install test

-include $(wildcard build/*.d)
