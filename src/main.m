#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/QuartzCore.h>
#include "gol.h"
#include <math.h>

// Rendering / simulation architecture:
//
// _gridBuf is one shared MTLBuffer containing PLANE_COUNT grid planes.
// Each cell is one uint16_t: bit 0 = alive, bits 8..15 = age.
//
// The active grid is _gridW x _gridH. The buffer is allocated for the maximum
// supported grid size, and each plane uses _planeCells as its row stride.
//
// Frame loop:
//   MTKView calls drawInMTKView: -> tick
//   tick builds one MTLCommandBuffer:
//     1. optional compute step: reads current plane, writes next plane
//     2. render pass: fullscreen quad samples the plane to display
//     3. present drawable and commit
//
// Triple buffering:
//   _frameIndex selects planes in a ring. _cbRing tracks in-flight command
//   buffers. If the oldest buffer for a plane is still active, tick renders
//   the latest submitted plane but skips the simulation step for that frame.
//
// gol_step_cpu() is only a fallback for systems where the Metal compute
// pipeline cannot be created.

static const int INITIAL_GRID_W = 160;
static const int INITIAL_GRID_H = 120;
static const CGFloat CELL_PX = 6.0;
static const int VIEW_W = (int)(INITIAL_GRID_W * CELL_PX);
static const int VIEW_H = (int)(INITIAL_GRID_H * CELL_PX);
static const int BAR_H = 120;
static const int RENDER_SCALE = 1;
enum { PLANE_COUNT = 3 };

typedef struct {
    uint32_t gridW;
    uint32_t gridH;
    uint32_t curOffset;
    uint32_t pad;
    uint8_t birth;
    uint8_t survival;
    uint8_t pad2;
    uint8_t pad3;
} Uniforms;

static MTLSize MakeSize(int w, int h, int d) {
    MTLSize s;
    s.width = (NSUInteger)w;
    s.height = (NSUInteger)h;
    s.depth = (NSUInteger)d;
    return s;
}

static NSTextField *MakeLabel(NSString *s, NSRect f) {
    NSTextField *l = [NSTextField labelWithString:s];
    l.frame = f;
    l.font = [NSFont systemFontOfSize:13];
    l.textColor = [NSColor colorWithSRGBRed:0.82 green:0.87 blue:0.92 alpha:1.0];
    return l;
}

static NSTextField *MakeLabelSmall(NSString *s, NSRect f) {
    NSTextField *l = [NSTextField labelWithString:s];
    l.frame = f;
    l.font = [NSFont systemFontOfSize:11];
    l.textColor = [NSColor colorWithSRGBRed:0.65 green:0.70 blue:0.76 alpha:1.0];
    return l;
}

@interface App : NSObject <NSApplicationDelegate, NSWindowDelegate, MTKViewDelegate>
- (BOOL)setupMetal;
- (void)setupUI;
- (void)randomize;
- (void)goPause:(id)sender;
- (void)reset:(id)sender;
- (void)sliderChanged:(id)sender;
- (void)paintAtEvent:(NSEvent *)e add:(BOOL)add;
- (void)waitLast;
- (uint16_t *)planePointer:(uint32_t)plane;
- (uint16_t *)currentCells;
- (void)rebuildCellTexture;
- (void)updateGridForPixelSize:(CGSize)pixelSize;
- (void)markDirty;
- (void)tick;
- (void)drawInMTKView:(MTKView *)view;
- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size;
- (void)ruleToggled:(id)sender;
- (void)rulePresetClicked:(id)sender;
- (void)applyPreset:(const char *)name;
@end

@interface GOLView : MTKView
@property (nonatomic, weak) App *owner;
@end

@implementation GOLView

- (void)mouseDown:(NSEvent *)e {
    [self.owner paintAtEvent:e add:YES];
}

- (void)mouseDragged:(NSEvent *)e {
    [self.owner paintAtEvent:e add:YES];
}

- (void)rightMouseDown:(NSEvent *)e {
    [self.owner paintAtEvent:e add:NO];
}

- (void)rightMouseDragged:(NSEvent *)e {
    [self.owner paintAtEvent:e add:NO];
}

@end

@implementation App {
    NSWindow *_window;
    GOLView *_mtkView;
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;
    id<MTLLibrary> _library;
    id<MTLRenderPipelineState> _renderPipeline;
    id<MTLRenderPipelineState> _scalePipeline;
    id<MTLComputePipelineState> _stepPipeline;
    id<MTLTexture> _cellTex;
    id<MTLBuffer> _gridBuf;
    id<MTLBuffer> _uniformsBuf;
    id<MTLCommandBuffer> _cbRing[PLANE_COUNT];
    id<MTLCommandBuffer> _lastCB;
    uint64_t _frameIndex;
    uint32_t _displayPlane;
    int _gridW;
    int _gridH;
    int _maxGridW;
    int _maxGridH;
    NSUInteger _planeCells;
    NSUInteger _planeBytes;
    BOOL _running;
    BOOL _dirty;
    uint32_t _gen;
    CFTimeInterval _fpsLast;
    uint32_t _fpsFrames;
    GOLRules _rules;
    NSSlider *_densitySlider;
    NSTextField *_densityPct;
    NSButton *_goButton;
    NSTextField *_genLabel;
    NSTextField *_fpsLabel;
    NSTextField *_ruleLabel;
    NSTextField *_popLabel;
    NSTextField *_maxAgeLabel;
    NSSlider *_speedSlider;
    NSTextField *_speedLabel;
    NSPopUpButton *_presetsPopup;
    NSButton *_ruleToggles[9]; // birth toggles
    NSButton *_ruleTSurv[9];  // survival toggles
    CFTimeInterval _genAccum;
}

- (void)applicationDidFinishLaunching:(NSNotification *)note {
    (void)note;
    _frameIndex = 0;
    _displayPlane = 0;
    _gridW = INITIAL_GRID_W;
    _gridH = INITIAL_GRID_H;
    _gen = 0;
    _dirty = NO;
    _fpsLast = 0;
    _fpsFrames = 0;
    _genAccum = 0;
    _rules = gol_default_rules();
    if (![self setupMetal]) {
        NSLog(@"Metal setup failed");
        [NSApp terminate:nil];
        return;
    }
    [self setupUI];
    [self randomize];
}

- (BOOL)setupMetal {
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) return NO;
    _queue = [_device newCommandQueue];
    if (!_queue) return NO;

    NSString *exeDir = [[NSBundle mainBundle] executablePath];
    exeDir = [exeDir stringByDeletingLastPathComponent];
    NSString *libPath = [exeDir stringByAppendingPathComponent:@"shaders.metallib"];
    NSError *err = nil;
    _library = [_device newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&err];
    if (!_library) {
        NSLog(@"failed to load metallib: %@", err);
        return NO;
    }

    _mtkView = [[GOLView alloc] initWithFrame:NSMakeRect(0, BAR_H, VIEW_W, VIEW_H)
                                          device:_device];
    _mtkView.wantsLayer = YES;
    _mtkView.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    _mtkView.framebufferOnly = YES;
    _mtkView.autoResizeDrawable = YES;
    _mtkView.preferredFramesPerSecond = 120;
    _mtkView.paused = NO;
    _mtkView.owner = self;

    NSScreen *screen = _mtkView.window.screen ?: [NSScreen mainScreen];
    CGFloat screenW = screen.frame.size.width;
    CGFloat screenH = screen.frame.size.height;
    _maxGridW = (int)floor(screenW / CELL_PX) + 16;
    _maxGridH = (int)floor(MAX(0.0, screenH - (CGFloat)BAR_H) / CELL_PX) + 16;
    _maxGridW = MAX(_maxGridW, INITIAL_GRID_W);
    _maxGridH = MAX(_maxGridH, INITIAL_GRID_H);
    NSUInteger need = (NSUInteger)_maxGridW * (NSUInteger)_maxGridH;
    _planeCells = ((need + 7) / 8) * 8;
    _planeBytes = _planeCells * sizeof(uint16_t);

    _gridBuf = [_device newBufferWithLength:(NSUInteger)PLANE_COUNT * _planeBytes
                                    options:MTLResourceStorageModeShared];
    if (!_gridBuf) return NO;

    _uniformsBuf = [_device newBufferWithLength:sizeof(Uniforms) options:MTLResourceStorageModeShared];
    if (!_uniformsBuf) return NO;

    for (int i = 0; i < PLANE_COUNT; i++) {
        _cbRing[i] = nil;
    }

    MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
    rp.vertexFunction = [_library newFunctionWithName:@"vs_main"];
    rp.fragmentFunction = [_library newFunctionWithName:@"fs_main"];
    rp.colorAttachments[0].pixelFormat = _mtkView.colorPixelFormat;
    _renderPipeline = [_device newRenderPipelineStateWithDescriptor:rp error:&err];
    if (!_renderPipeline) {
        NSLog(@"failed to create render pipeline: %@", err);
        return NO;
    }

    MTLRenderPipelineDescriptor *sp = [MTLRenderPipelineDescriptor new];
    sp.vertexFunction = [_library newFunctionWithName:@"vs_main"];
    sp.fragmentFunction = [_library newFunctionWithName:@"fs_scale"];
    sp.colorAttachments[0].pixelFormat = _mtkView.colorPixelFormat;
    _scalePipeline = [_device newRenderPipelineStateWithDescriptor:sp error:&err];
    if (!_scalePipeline) {
        NSLog(@"failed to create scale pipeline: %@", err);
        return NO;
    }

    id<MTLFunction> stepFunc = [_library newFunctionWithName:@"gol_step"];
    _stepPipeline = [_device newComputePipelineStateWithFunction:stepFunc error:&err];
    if (!_stepPipeline) {
        NSLog(@"compute pipeline unavailable: %@; using CPU fallback", err);
        _stepPipeline = nil;
    }

    [self rebuildCellTexture];
    if (!_cellTex) return NO;

    _mtkView.delegate = self;
    return YES;
}

- (void)setupUniformsGPU {
    if (!_uniformsBuf) return;
    Uniforms *u = (Uniforms *)[_uniformsBuf contents];
    u->birth = (uint8_t)_rules.birth;
    u->survival = (uint8_t)_rules.survival;
}

- (void)updateRuleUI {
    uint8_t b = _rules.birth;
    uint8_t s = _rules.survival;
    for (int i = 0; i < 9; i++) {
        BOOL onB = (b >> i) & 1u;
        BOOL onS = (s >> i) & 1u;
        if (_ruleToggles[i]) {
            _ruleToggles[i].state = onB ? NSControlStateValueOn : NSControlStateValueOff;
        }
        if (_ruleTSurv[i]) {
            _ruleTSurv[i].state = onS ? NSControlStateValueOn : NSControlStateValueOff;
        }
    }
    
    // Build B/S label
    NSMutableString *label = [NSMutableString string];
    [label appendString:@"B"];
    for (int i = 0; i < 9; i++) {
        if ((b >> i) & 1u) [label appendFormat:@"%d", i];
    }
    [label appendString:@"/S"];
    for (int i = 0; i < 9; i++) {
        if ((s >> i) & 1u) [label appendFormat:@"%d", i];
    }
    if (_ruleLabel) {
        _ruleLabel.stringValue = label;
    }
    
    [self setupUniformsGPU];
}

- (void)applyPreset:(const char *)name {
    [self waitLast];
    for (int i = 0; i < PLANE_COUNT; i++) {
        _cbRing[i] = nil;
    }
    _frameIndex = 0;
    _displayPlane = 0;
    _gen = 0;
    uint16_t *cells = [self currentCells];
    if (!cells) return;
    gol_apply_preset(cells, _gridW, _gridH, name);
    uint16_t *base = (uint16_t *)[_gridBuf contents];
    for (uint32_t p = 1; p < (uint32_t)PLANE_COUNT; p++) {
        memset(base + (size_t)p * (size_t)_planeCells, 0, _planeBytes);
    }
    if (_genLabel) {
        _genLabel.stringValue = @"Gen 0";
    }
    [self markDirty];
}

- (void)rulePresetClicked:(id)sender {
    NSButton *btn = (NSButton *)sender;
    NSString *preset = btn.title;
    if ([preset isEqualToString:@"Glider"]) {
        [self applyPreset:"glider"];
    } else if ([preset isEqualToString:@"Blinker"]) {
        [self applyPreset:"blinker"];
    } else if ([preset isEqualToString:@"Block"]) {
        [self applyPreset:"block"];
    } else if ([preset isEqualToString:@"Beacon"]) {
        [self applyPreset:"beacon"];
    } else if ([preset isEqualToString:@"Toad"]) {
        [self applyPreset:"toad"];
    } else if ([preset isEqualToString:@"Pentadecathlon"]) {
        [self applyPreset:"pentadecathlon"];
    } else if ([preset isEqualToString:@"LWSS"]) {
        [self applyPreset:"lwss"];
    } else if ([preset isEqualToString:@"R-Pentomino"]) {
        [self applyPreset:"r-pentomino"];
    } else if ([preset isEqualToString:@"Heptomino"]) {
        [self applyPreset:"heptomino"];
    }
}

- (void)ruleToggled:(id)sender {
    NSButton *btn = (NSButton *)sender;
    NSInteger tag = btn.tag;
    
    // Tags 0-8 = birth, 10-18 = survival (skip 9)
    if (tag >= 0 && tag <= 8) {
        _rules.birth = (uint8_t)((_rules.birth & ~(1u << tag)) | ((btn.state == NSControlStateValueOn) ? (1u << tag) : 0));
    } else if (tag >= 10 && tag <= 18) {
        int n = tag - 10;
        _rules.survival = (uint8_t)((_rules.survival & ~(1u << n)) | ((btn.state == NSControlStateValueOn) ? (1u << n) : 0));
    }
    
    [self updateRuleUI];
    [self randomize];
}

- (void)randomize {
    double d = _densitySlider ? _densitySlider.doubleValue : 0.2;
    if (d < 0.0) d = 0.0;
    if (d > 1.0) d = 1.0;
    [self waitLast];
    for (int i = 0; i < PLANE_COUNT; i++) {
        _cbRing[i] = nil;
    }
    _frameIndex = 0;
    _displayPlane = 0;
    _gen = 0;
    _genAccum = 0;
    uint16_t *cells = [self currentCells];
    if (!cells) return;
    gol_randomize(cells, _planeCells, _gridW, _gridH, d);
    uint16_t *base = (uint16_t *)[_gridBuf contents];
    for (uint32_t p = 1; p < (uint32_t)PLANE_COUNT; p++) {
        memset(base + (size_t)p * (size_t)_planeCells, 0, _planeBytes);
    }
    if (_densityPct) {
        _densityPct.stringValue = [NSString stringWithFormat:@"%d%%", (int)lround(d * 100.0)];
    }
    if (_genLabel) {
        _genLabel.stringValue = @"Gen 0";
    }
    [self markDirty];
}

- (void)goPause:(id)sender {
    (void)sender;
    _running = !_running;
    _goButton.title = _running ? @"Pause" : @"Go";
    if (_running) {
        _mtkView.paused = NO;
    } else if (!_dirty) {
        _mtkView.paused = YES;
    }
}

- (void)reset:(id)sender {
    (void)sender;
    [self randomize];
}

- (void)sliderChanged:(id)sender {
    if (sender == _speedSlider) {
        if (_speedLabel) {
            int val = (int)_speedSlider.doubleValue;
            _speedLabel.stringValue = [NSString stringWithFormat:@"%d gen/s", val];
        }
        return;
    }
    [self randomize];
}

- (void)paintAtEvent:(NSEvent *)e add:(BOOL)add {
    if (!_mtkView) return;
    [self waitLast];
    NSPoint pt = [_mtkView convertPoint:[e locationInWindow] fromView:nil];
    CGFloat viewH = _mtkView.bounds.size.height;
    double yTop = [_mtkView isFlipped] ? pt.y : (viewH - pt.y);
    int col = (int)floor(pt.x / CELL_PX);
    int row = (int)floor(yTop / CELL_PX);
    if (col < 0 || col >= _gridW || row < 0 || row >= _gridH) return;
    uint16_t *cells = [self currentCells];
    if (!cells) return;
    static const int kBrush[5][2] = {
        {0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}
    };
    for (int k = 0; k < 5; k++) {
        int cx = col + kBrush[k][0];
        int cy = row + kBrush[k][1];
        gol_set_plane(cells, _gridW, _gridH, cx, cy, add);
    }
    [self markDirty];
}

- (void)waitLast {
    if (_lastCB) {
        [_lastCB waitUntilCompleted];
        if (_lastCB.error) {
            NSLog(@"command buffer error: %@", _lastCB.error);
        }
        _lastCB = nil;
    }
}

- (uint16_t *)planePointer:(uint32_t)plane {
    if (!_gridBuf) return nil;
    return (uint16_t *)[_gridBuf contents] + (size_t)plane * (size_t)_planeCells;
}

- (uint16_t *)currentCells {
    return [self planePointer:(uint32_t)(_frameIndex % (uint32_t)PLANE_COUNT)];
}

- (void)updateGridForPixelSize:(CGSize)pixelSize {
    if (!_gridBuf || !_mtkView) return;
    if (pixelSize.width < 1.0 || pixelSize.height < 1.0) return;

    CGFloat scale = _mtkView.window ? _mtkView.window.backingScaleFactor : 1.0;
    if (scale < 1.0) scale = 1.0;

    int newW = (int)floor(pixelSize.width / (CELL_PX * scale));
    int newH = (int)floor(pixelSize.height / (CELL_PX * scale));
    newW = MAX(8, MIN(newW, _maxGridW));
    newH = MAX(8, MIN(newH, _maxGridH));

    if ((NSUInteger)newW * (NSUInteger)newH > _planeCells) {
        double shrink = sqrt((double)_planeCells / ((double)newW * (double)newH));
        newW = MAX(8, (int)floor(newW * shrink));
        newH = MAX(8, (int)floor(newH * shrink));
    }

    if (newW == _gridW && newH == _gridH) return;

    [self waitLast];
    for (int i = 0; i < PLANE_COUNT; i++) {
        _cbRing[i] = nil;
    }

    int oldW = _gridW;
    int oldH = _gridH;
    uint16_t *base = (uint16_t *)[_gridBuf contents];
    uint16_t *oldPlane = base + (size_t)_displayPlane * (size_t)_planeCells;
    uint16_t *tmp = calloc(_planeCells, sizeof(uint16_t));
    if (!tmp) return;

    gol_resize_copy(oldPlane, oldW, oldH, tmp, newW, newH, _planeCells);
    _gridW = newW;
    _gridH = newH;

    memcpy(base, tmp, _planeCells * sizeof(uint16_t));
    for (uint32_t p = 1; p < (uint32_t)PLANE_COUNT; p++) {
        memset(base + (size_t)p * (size_t)_planeCells, 0, _planeBytes);
    }
    free(tmp);

    _frameIndex = 0;
    _displayPlane = 0;
    [self rebuildCellTexture];
}

- (void)rebuildCellTexture {
    if (!_device || !_mtkView) return;
    NSUInteger w = (NSUInteger)MAX(1, _gridW) * (NSUInteger)RENDER_SCALE;
    NSUInteger h = (NSUInteger)MAX(1, _gridH) * (NSUInteger)RENDER_SCALE;
    w = MIN(w, (NSUInteger)16384);
    h = MIN(h, (NSUInteger)16384);
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:_mtkView.colorPixelFormat
                                                                                 width:w
                                                                                 height:h
                                                                             mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModePrivate;
    _cellTex = [_device newTextureWithDescriptor:d];
}

- (void)markDirty {
    _dirty = YES;
    if (_mtkView) {
        _mtkView.paused = NO;
    }
}

static NSButton *makeToggle(NSString *text, NSRect frame, BOOL on, NSInteger tag, id target, SEL action) {
    NSButton *btn = [[NSButton alloc] initWithFrame:frame];
    btn.bezelStyle = NSBezelStyleRounded;
    btn.state = on ? NSControlStateValueOn : NSControlStateValueOff;
    btn.tag = tag;
    btn.allowsMixedState = NO;
    btn.title = @"";
    btn.wantsLayer = YES;
    btn.layer.backgroundColor = on ?
        [NSColor colorWithSRGBRed:0.3 green:0.85 blue:0.4 alpha:1.0].CGColor :
        [NSColor colorWithSRGBRed:0.22 green:0.25 blue:0.32 alpha:1.0].CGColor;
    btn.layer.cornerRadius = frame.size.width / 2.0;
    btn.target = target;
    btn.action = action;
    
    NSTextField *label = [NSTextField labelWithString:text];
    label.frame = NSMakeRect(0, 0, frame.size.width, frame.size.height);
    label.font = [NSFont systemFontOfSize:10 weight:NSFontWeightMedium];
    label.textColor = on ? [NSColor whiteColor] : [NSColor colorWithSRGBRed:0.55 green:0.60 blue:0.68 alpha:1.0];
    label.alignment = NSTextAlignmentCenter;
    label.backgroundColor = [NSColor clearColor];
    label.selectable = NO;
    label.editable = NO;
    [btn addSubview:label];
    
    return btn;
}

- (void)setupUI {
    NSRect content = NSMakeRect(0, 0, VIEW_W, VIEW_H + BAR_H);
    _window = [[NSWindow alloc] initWithContentRect:content
                                            styleMask:(NSWindowStyleMaskTitled |
                                                        NSWindowStyleMaskClosable |
                                                        NSWindowStyleMaskMiniaturizable |
                                                        NSWindowStyleMaskResizable)
                                              backing:NSBackingStoreBuffered
                                                defer:NO];
    _window.releasedWhenClosed = NO;
    [_window setTitle:@"Game of Life"];
    [_window setContentSize:content.size];
    [_window setBackgroundColor:[NSColor colorWithSRGBRed:0.04 green:0.05 blue:0.08 alpha:1.0]];
    [_window setDelegate:self];
    [_window center];

    NSScreen *screen = _window.screen ?: [NSScreen mainScreen];
    _window.minSize = NSMakeSize(500.0, 350.0 + (CGFloat)BAR_H);
    _window.maxSize = NSMakeSize(screen.frame.size.width, screen.frame.size.height);

    NSView *contentView = [_window contentView];
    _mtkView.frame = NSMakeRect(0, BAR_H, VIEW_W, VIEW_H);
    _mtkView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    [contentView addSubview:_mtkView];

    NSView *bar = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, VIEW_W, BAR_H)];
    bar.wantsLayer = YES;
    bar.autoresizingMask = NSViewWidthSizable;
    bar.layer.backgroundColor = [NSColor colorWithSRGBRed:0.07 green:0.08 blue:0.11 alpha:1.0].CGColor;
    [contentView addSubview:bar];
    
    CGFloat x = 12;
    CGFloat y = 8;
    CGFloat btnW = 24;
    CGFloat btnH = 24;
    CGFloat btnGap = 4;
    
    // FPS label
    _fpsLabel = MakeLabel(@"FPS --", NSMakeRect(x, y, 80, 18));
    [bar addSubview:_fpsLabel];
    x += 90;
    
    // Rule section
    [bar addSubview:MakeLabelSmall(@"B", NSMakeRect(x, y - 2, 14, 22))];
    x += 18;
    for (int i = 0; i < 9; i++) {
        _ruleToggles[i] = makeToggle([NSString stringWithFormat:@"%d", i],
                                      NSMakeRect(x, y - 2, btnW, btnH),
                                      (_rules.birth >> i) & 1u,
                                      i, self, @selector(ruleToggled:));
        [bar addSubview:_ruleToggles[i]];
        x += btnW + btnGap;
    }
    
    x += 8;
    [bar addSubview:MakeLabelSmall(@"S", NSMakeRect(x, y - 2, 14, 22))];
    x += 18;
    for (int i = 0; i < 9; i++) {
        _ruleTSurv[i] = makeToggle([NSString stringWithFormat:@"%d", i],
                                    NSMakeRect(x, y - 2, btnW, btnH),
                                    (_rules.survival >> i) & 1u,
                                    i + 10, self, @selector(ruleToggled:));
        [bar addSubview:_ruleTSurv[i]];
        x += btnW + btnGap;
    }
    
    x += 8;
    _ruleLabel = MakeLabelSmall(@"B3/S23", NSMakeRect(x, y - 2, 100, 22));
    [bar addSubview:_ruleLabel];
    x += 110;
    
    _popLabel = MakeLabelSmall(@"Pop: 0", NSMakeRect(x, y - 2, 90, 22));
    [bar addSubview:_popLabel];
    x += 100;
    
    _maxAgeLabel = MakeLabelSmall(@"Age: 0", NSMakeRect(x, y - 2, 70, 22));
    [bar addSubview:_maxAgeLabel];
    x += 80;
    
    // --- Second row ---
    y = 38;
    x = 12;
    
    [bar addSubview:MakeLabelSmall(@"Density", NSMakeRect(x, y, 55, 18))];
    x += 60;
    
    _densitySlider = [[NSSlider alloc] initWithFrame:NSMakeRect(x, y - 3, 100, 20)];
    _densitySlider.minValue = 0.0;
    _densitySlider.maxValue = 1.0;
    _densitySlider.doubleValue = 0.2;
    _densitySlider.allowsTickMarkValuesOnly = NO;
    _densitySlider.numberOfTickMarks = 0;
    _densitySlider.continuous = YES;
    _densitySlider.target = self;
    _densitySlider.action = @selector(sliderChanged:);
    [bar addSubview:_densitySlider];
    x += 110;
    
    _densityPct = MakeLabelSmall(@"20%", NSMakeRect(x, y, 35, 18));
    [bar addSubview:_densityPct];
    x += 45;
    
    [bar addSubview:MakeLabelSmall(@"Speed", NSMakeRect(x, y, 45, 18))];
    x += 50;
    
    _speedSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(x, y - 3, 120, 20)];
    _speedSlider.minValue = 1.0;
    _speedSlider.maxValue = 120.0;
    _speedSlider.doubleValue = 30.0;
    _speedSlider.allowsTickMarkValuesOnly = NO;
    _speedSlider.numberOfTickMarks = 6;
    _speedSlider.continuous = YES;
    _speedSlider.target = self;
    _speedSlider.action = @selector(sliderChanged:);
    [bar addSubview:_speedSlider];
    x += 130;
    
    _speedLabel = MakeLabelSmall(@"30 gen/s", NSMakeRect(x, y, 65, 18));
    [bar addSubview:_speedLabel];
    x += 75;
    
    [bar addSubview:MakeLabelSmall(@"Preset", NSMakeRect(x, y, 45, 18))];
    x += 50;
    
    _presetsPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(x, y - 3, 120, 24)
                                                 pullsDown:NO];
    [_presetsPopup addItemWithTitle:@"Glider"];
    [_presetsPopup addItemWithTitle:@"Blinker"];
    [_presetsPopup addItemWithTitle:@"Block"];
    [_presetsPopup addItemWithTitle:@"Beacon"];
    [_presetsPopup addItemWithTitle:@"Toad"];
    [_presetsPopup addItemWithTitle:@"Pentadecathlon"];
    [_presetsPopup addItemWithTitle:@"LWSS"];
    [_presetsPopup addItemWithTitle:@"R-Pentomino"];
    [_presetsPopup addItemWithTitle:@"Heptomino"];
    [_presetsPopup setTarget:self];
    [_presetsPopup setAction:@selector(rulePresetClicked:)];
    [bar addSubview:_presetsPopup];
    x += 130;
    
    _goButton = [[NSButton alloc] initWithFrame:NSMakeRect(x, y - 3, 65, 24)];
    _goButton.title = @"Go";
    _goButton.bezelStyle = NSBezelStyleRounded;
    _goButton.target = self;
    _goButton.action = @selector(goPause:);
    [bar addSubview:_goButton];
    x += 75;
    
    NSButton *resetButton = [[NSButton alloc] initWithFrame:NSMakeRect(x, y - 3, 65, 24)];
    resetButton.title = @"Reset";
    resetButton.bezelStyle = NSBezelStyleRounded;
    resetButton.target = self;
    resetButton.action = @selector(reset:);
    [bar addSubview:resetButton];
    x += 75;
    
    _genLabel = MakeLabelSmall(@"Gen 0", NSMakeRect(x, y, 90, 18));
    [bar addSubview:_genLabel];
    x += 100;
    
    [bar addSubview:MakeLabelSmall(@"L:add  R:erase", NSMakeRect(x, y, 120, 18))];
    
    [self updateRuleUI];
    [_window makeKeyAndOrderFront:nil];
}


- (void)drawInMTKView:(MTKView *)view {
    (void)view;
    [self tick];
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    (void)view;
    [self updateGridForPixelSize:size];
    [self markDirty];
}

- (void)tick {
    if (!_mtkView || !_queue || !_renderPipeline || !_scalePipeline ||
        !_uniformsBuf || !_gridBuf) return;

    // If paused and nothing changed, stop asking the display link for work.
    if (!_running && !_dirty) {
        _mtkView.paused = YES;
        return;
    }

    // 1. Ask MTKView for the next drawable and create this frame's command buffer.
    id<CAMetalDrawable> drawable = _mtkView.currentDrawable;
    if (!drawable) return;
    id<MTLCommandBuffer> cb = [_queue commandBuffer];
    if (!cb) return;

    // 2. Choose the grid planes for this frame using the triple-buffer ring.
    uint64_t f = _frameIndex;
    uint32_t slot = (uint32_t)(f % (uint32_t)PLANE_COUNT);
    uint32_t curPlane = (uint32_t)(f % (uint32_t)PLANE_COUNT);
    uint32_t renderPlane = _displayPlane;
    BOOL willStep = NO;

    // Speed control: accumulate time, advance generations based on slider
    CFTimeInterval now = CFAbsoluteTimeGetCurrent();
    
    // Update uniforms BEFORE any GPU work so shaders see correct rules
    Uniforms *u = (Uniforms *)[_uniformsBuf contents];
    u->gridW = (uint32_t)_gridW;
    u->gridH = (uint32_t)_gridH;
    u->pad = 0;
    u->birth = (uint8_t)_rules.birth;
    u->survival = (uint8_t)_rules.survival;
    u->pad2 = 0;
    u->pad3 = 0;
    
    if (_running) {
        double genPerSec = _speedSlider ? _speedSlider.doubleValue : 30.0;
        if (genPerSec < 1.0) genPerSec = 1.0;
        double genInterval = 1.0 / genPerSec;
        _genAccum += (now - _fpsLast > 0 && _fpsLast > 0) ? (now - _fpsLast) : genInterval;
        
        // Advance multiple generations if needed (e.g., slow computer catching up)
        int gensToAdvance = (int)floor(_genAccum / genInterval);
        gensToAdvance = MIN(gensToAdvance, 5); // cap at 5 per frame
        _genAccum -= gensToAdvance * genInterval;
        
        if (gensToAdvance > 0) {
            // We only advance one generation per frame on GPU (single command buffer)
            // For multiple gens, we'd need a different approach. Just advance 1.
            gensToAdvance = 1;
        }

        if (gensToAdvance > 0 && _stepPipeline) {
            id<MTLCommandBuffer> oldest = _cbRing[slot];
            BOOL oldestDone = !oldest ||
                              oldest.status == MTLCommandBufferStatusCompleted ||
                              oldest.status == MTLCommandBufferStatusError;
            if (oldestDone) {
                uint32_t writePlane = (curPlane + 1) % (uint32_t)PLANE_COUNT;
                NSUInteger curByte = (NSUInteger)curPlane * _planeBytes;
                NSUInteger writeByte = (NSUInteger)writePlane * _planeBytes;
                const NSUInteger TGW = 16;
                const NSUInteger TGH = 16;
                NSUInteger gx = ((NSUInteger)_gridW + TGW - 1) / TGW;
                NSUInteger gy = ((NSUInteger)_gridH + TGH - 1) / TGH;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:_stepPipeline];
                [enc setBuffer:_gridBuf offset:curByte atIndex:0];
                [enc setBuffer:_gridBuf offset:writeByte atIndex:1];
                [enc setBuffer:_uniformsBuf offset:0 atIndex:2];
                [enc dispatchThreadgroups:MakeSize((int)gx, (int)gy, 1)
                  threadsPerThreadgroup:MakeSize((int)TGW, (int)TGH, 1)];
                [enc endEncoding];
                renderPlane = writePlane;
                _displayPlane = writePlane;
                willStep = YES;
            }
        } else if (gensToAdvance > 0) {
            // CPU fallback
            uint32_t writePlane = (curPlane + 1) % (uint32_t)PLANE_COUNT;
            uint16_t *cur = [self planePointer:curPlane];
            uint16_t *write = [self planePointer:writePlane];
            if (cur && write) {
                gol_step_cpu(cur, write, _gridW, _gridH, _rules);
                renderPlane = writePlane;
                _displayPlane = writePlane;
                willStep = YES;
            }
        }
    }

    // 4. Update curOffset for render pass (grid/plane may have changed)
    u->curOffset = (uint32_t)renderPlane * (uint32_t)_planeCells;

    // Re-render the small cell texture only when the visible grid changed.
    BOOL needCell = willStep || _dirty;
    if (needCell && _cellTex && _cellTex.width > 0 && _cellTex.height > 0) {
        MTLRenderPassDescriptor *crp = [MTLRenderPassDescriptor renderPassDescriptor];
        crp.colorAttachments[0].texture = _cellTex;
        crp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        crp.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> cenc = [cb renderCommandEncoderWithDescriptor:crp];
        [cenc setRenderPipelineState:_renderPipeline];
        MTLViewport cvp;
        cvp.originX = 0.0;
        cvp.originY = 0.0;
        cvp.width = (double)_cellTex.width;
        cvp.height = (double)_cellTex.height;
        cvp.znear = 0.0;
        cvp.zfar = 1.0;
        [cenc setViewport:cvp];
        [cenc setFragmentBuffer:_gridBuf offset:0 atIndex:0];
        [cenc setFragmentBuffer:_uniformsBuf offset:0 atIndex:1];
        [cenc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        [cenc endEncoding];
    }

    // 5. Scale the cell texture into the drawable.
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = drawable.texture;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> rend = [cb renderCommandEncoderWithDescriptor:rp];
    [rend setRenderPipelineState:_scalePipeline];
    MTLViewport vp;
    vp.originX = 0.0;
    vp.originY = 0.0;
    vp.width = (double)drawable.texture.width;
    vp.height = (double)drawable.texture.height;
    vp.znear = 0.0;
    vp.zfar = 1.0;
    [rend setViewport:vp];
    [rend setFragmentBuffer:_uniformsBuf offset:0 atIndex:0];
    if (_cellTex) {
        [rend setFragmentTexture:_cellTex atIndex:0];
    }
    [rend drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [rend endEncoding];

    // 6. Present and commit.
    [cb presentDrawable:drawable];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
        if (b.error) {
            NSLog(@"command buffer error: %@", b.error);
        }
    }];
    [cb commit];

    // 7. Update bookkeeping after the frame has been submitted.
    if (willStep) {
        _frameIndex = f + 1;
        _gen++;
    }
    if (needCell && _cellTex) {
        _dirty = NO;
    }
    _cbRing[slot] = cb;
    _lastCB = cb;

    if (_genLabel) {
        _genLabel.stringValue = [NSString stringWithFormat:@"Gen %u", _gen];
    }

    // Update stats
    int alive = 0;
    int maxAge = 0;
    gol_count_alive([self currentCells], _gridW, _gridH, &alive, &maxAge);
    if (_popLabel) {
        _popLabel.stringValue = [NSString stringWithFormat:@"Pop: %d", alive];
    }
    if (_maxAgeLabel) {
        _maxAgeLabel.stringValue = [NSString stringWithFormat:@"MaxAge: %d", maxAge];
    }

    _fpsFrames++;
    _fpsLast = now;
    if (_fpsFrames >= 10) {
        CFTimeInterval dt = now - (_fpsLast - 0.5);
        if (dt > 0) {
            int fps = (int)lround((double)_fpsFrames / dt);
            if (_fpsLabel) {
                _fpsLabel.stringValue = [NSString stringWithFormat:@"FPS %d", fps];
            }
        }
        _fpsFrames = 0;
        _fpsLast = now;
    }

    if (!_running && !_dirty) {
        _mtkView.paused = YES;
    }
}

- (BOOL)windowShouldClose:(NSWindow *)sender {
    (void)sender;
    [NSApp terminate:nil];
    return NO;
}

@end

int main(int argc, const char *argv[]) {
    (void)argc;
    (void)argv;
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        App *app = [[App alloc] init];
        [NSApp setDelegate:app];
        [NSApp activateIgnoringOtherApps:YES];
        [NSApp run];
    }
    return 0;
}
