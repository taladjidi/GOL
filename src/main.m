#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/QuartzCore.h>
#include "gol.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Rendering / simulation architecture:
//
// gridBuf is one shared MTLBuffer containing PLANE_COUNT grid planes.
// Each cell is one uint16_t: bit 0 = alive, bits 1..15 = age.
//
// The active grid is gridW x gridH. The buffer is allocated for the maximum
// supported grid size, and each plane uses planeCells as its row stride.
//
// Frame loop:
//   MTKView calls drawInMTKView: -> tick
//   tick builds one MTLCommandBuffer:
//     1. optional compute step: reads current plane, writes next plane
//     2. render pass: fullscreen quad samples the plane to display
//     3. present drawable and commit
//
// Triple buffering:
//   frameIndex selects planes in a ring. cb0/cb1/cb2 track in-flight command
//   buffers. If the oldest buffer for a plane is still active, tick renders
//   the latest submitted plane but skips the simulation step for that frame.
//
// gol_step_cpu() is only a fallback for systems where the Metal compute
// pipeline cannot be created.

static const int INITIAL_GRID_W = 160;
static const int INITIAL_GRID_H = 120;
static const CGFloat CELL_PX = 6.0;
static const CGFloat MIN_CELL_PX = 1.0;
static const CGFloat MAX_CELL_PX = 64.0;
static const int VIEW_W = (int)(INITIAL_GRID_W * CELL_PX);
static const int VIEW_H = (int)(INITIAL_GRID_H * CELL_PX);
static const int BAR_H = 120;
static const int RENDER_SCALE = 1;
static const int PLANE_COUNT = 3;
static const NSUInteger MAX_PLANE_CELLS = 4000000;
static const int MAX_TEXTURE_SIZE = 16384;
static const uint32_t DISPLAY_AGE = 0u;
static const uint32_t DISPLAY_TRAILS = 1u;
static const uint32_t DISPLAY_HEATMAP = 2u;

#define GOL_MIN(A, B) ((A) < (B) ? (A) : (B))
#define GOL_MAX(A, B) ((A) > (B) ? (A) : (B))

typedef struct {
    uint32_t gridW;
    uint32_t gridH;
    uint32_t curOffset;
    uint32_t pad;
    uint8_t birth;
    uint8_t survival;
    uint8_t pad2;
    uint8_t pad3;
    float viewScaleX;
    float viewScaleY;
    float viewOffsetX;
    float viewOffsetY;
    float viewWidth;
    float viewHeight;
    uint32_t displayMode;
    uint32_t pad4;
} Uniforms;

static MTLSize MakeSize(int w, int h, int d) {
    MTLSize s;
    s.width = (NSUInteger)w;
    s.height = (NSUInteger)h;
    s.depth = (NSUInteger)d;
    return s;
}

static int FloorInt(double value) {
    double floored = floor(value);
    return (int)floored;
}

static int LroundInt(double value) {
    double shifted = value + (value >= 0.0 ? 0.5 : -0.5);
    double floored = floor(shifted);
    return (int)floored;
}

static int CeilIntStable(double value) {
    double shifted = value - 1e-6;
    double ceiled = ceil(shifted);
    return (int)ceiled;
}

static double ClampDouble(double value, double lo, double hi) {
    return GOL_MIN(GOL_MAX(value, lo), hi);
}

static NSScreen *ScreenForWindow(NSWindow *window) {
    if (window != nil && window.screen != nil) {
        return window.screen;
    }
    return [NSScreen mainScreen];
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

@class App;
@class GOLView;

@interface GOLRangeSlider : NSView
@property (nonatomic, assign) int minValue;
@property (nonatomic, assign) int maxValue;
@property (nonatomic, assign) int defaultMin;
@property (nonatomic, assign) int defaultMax;
@property (nonatomic, assign) int draggingHandle;
@property (nonatomic, copy) void (^rangeChanged)(void);
- (instancetype)initWithFrame:(NSRect)frame
                     minValue:(int)minValue
                          max:(int)maxValue
                   defaultMin:(int)defaultMin
                   defaultMax:(int)defaultMax;
- (void)setRange:(int)minValue max:(int)maxValue;
@end

@implementation GOLRangeSlider

@synthesize minValue = _minValue;
@synthesize maxValue = _maxValue;
@synthesize defaultMin = _defaultMin;
@synthesize defaultMax = _defaultMax;
@synthesize draggingHandle = _draggingHandle;
@synthesize rangeChanged = _rangeChanged;

- (instancetype)initWithFrame:(NSRect)frame
                     minValue:(int)minValue
                          max:(int)maxValue
                   defaultMin:(int)defaultMin
                   defaultMax:(int)defaultMax {
    if ((self = [super initWithFrame:frame])) {
        self.minValue = minValue;
        self.maxValue = maxValue;
        self.defaultMin = defaultMin;
        self.defaultMax = defaultMax;
        self.draggingHandle = -1;
        self.rangeChanged = nil;
        self.wantsLayer = YES;
        self.layer.backgroundColor = [NSColor clearColor].CGColor;
        self.toolTip = @"Drag handles to set rule range";
    }
    return self;
}

- (void)setRange:(int)minValue max:(int)maxValue {
    int t;
    self.minValue = GOL_MIN(GOL_MAX(minValue, 0), 8);
    self.maxValue = GOL_MIN(GOL_MAX(maxValue, 0), 8);
    if (self.minValue > self.maxValue) {
        t = self.minValue;
        self.minValue = self.maxValue;
        self.maxValue = t;
    }
    [self setNeedsDisplay:YES];
}

- (void)drawRect:(NSRect)dirtyRect {
    NSRect bounds;
    CGFloat trackY;
    CGFloat trackH;
    CGFloat trackX;
    CGFloat trackW;
    NSBezierPath *trackPath;
    CGFloat defX1;
    CGFloat defX2;
    NSBezierPath *defPath;
    CGFloat actX1;
    CGFloat actX2;
    NSBezierPath *actPath;
    int i;
    CGFloat lx;
    NSString *label;
    NSDictionary *attrs;
    NSSize textSize;
    NSPoint textPoint;
    int handleIndex;
    int val;
    CGFloat hx;
    CGFloat handleR;
    NSBezierPath *handlePath;

    (void)dirtyRect;
    [super drawRect:dirtyRect];

    bounds = [self bounds];
    trackY = bounds.size.height * 0.5;
    trackH = 6.0;
    trackX = 10.0;
    trackW = bounds.size.width - 20.0;

    trackPath = [NSBezierPath bezierPathWithRoundedRect:NSMakeRect(trackX, trackY - trackH * 0.5, trackW, trackH) xRadius:3.0 yRadius:3.0];
    [[NSColor colorWithSRGBRed:0.15 green:0.18 blue:0.22 alpha:1.0] setFill];
    [trackPath fill];

    if (self.defaultMin <= self.defaultMax) {
        defX1 = trackX + (self.defaultMin / 8.0) * trackW;
        defX2 = trackX + (self.defaultMax / 8.0) * trackW;
        defPath = [NSBezierPath bezierPathWithRoundedRect:NSMakeRect(defX1, trackY - trackH * 0.5 - 2, defX2 - defX1, trackH + 4) xRadius:4.0 yRadius:4.0];
        [[NSColor colorWithSRGBRed:0.3 green:0.6 blue:0.35 alpha:0.25] setFill];
        [defPath fill];
    }

    actX1 = trackX + (self.minValue / 8.0) * trackW;
    actX2 = trackX + (self.maxValue / 8.0) * trackW;
    actPath = [NSBezierPath bezierPathWithRoundedRect:NSMakeRect(actX1, trackY - trackH * 0.5, actX2 - actX1, trackH) xRadius:3.0 yRadius:3.0];
    [[NSColor colorWithSRGBRed:0.25 green:0.7 blue:0.35 alpha:0.6] setFill];
    [actPath fill];

    for (i = 0; i <= 8; i++) {
        lx = trackX + (i / 8.0) * trackW;
        label = [NSString stringWithFormat:@"%d", i];
        attrs = @{ NSFontAttributeName: [NSFont systemFontOfSize:9],
                   NSForegroundColorAttributeName: (i >= self.minValue && i <= self.maxValue) ? [NSColor whiteColor] : [NSColor colorWithSRGBRed:0.5 green:0.55 blue:0.6 alpha:1.0] };
        textSize = [label sizeWithAttributes:attrs];
        textPoint = NSMakePoint(lx - textSize.width * 0.5, trackY + 8);
        [label drawAtPoint:textPoint withAttributes:attrs];
    }

    handleR = 8.0;
    for (handleIndex = 0; handleIndex < 2; handleIndex++) {
        val = (handleIndex == 0) ? self.minValue : self.maxValue;
        hx = trackX + (val / 8.0) * trackW;
        handlePath = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(hx - handleR, trackY - handleR, handleR * 2, handleR * 2)];
        [[NSColor colorWithSRGBRed:0.9 green:0.92 blue:0.95 alpha:1.0] setFill];
        [handlePath fill];
        [[NSColor colorWithSRGBRed:0.4 green:0.45 blue:0.5 alpha:1.0] setStroke];
        handlePath.lineWidth = 1.5;
        [handlePath stroke];
    }
}

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (BOOL)becomeFirstResponder {
    return YES;
}

- (void)mouseDown:(NSEvent *)e {
    NSPoint pt;
    CGFloat trackX;
    CGFloat trackW;
    CGFloat minHx;
    CGFloat maxHx;
    CGFloat distMin;
    CGFloat distMax;

    [self becomeFirstResponder];
    pt = [self convertPoint:[e locationInWindow] fromView:nil];
    trackX = 10.0;
    trackW = [self bounds].size.width - 20.0;

    minHx = trackX + (self.minValue / 8.0) * trackW;
    maxHx = trackX + (self.maxValue / 8.0) * trackW;

    distMin = fabs(pt.x - minHx);
    distMax = fabs(pt.x - maxHx);

    if (distMin < distMax && distMin < 15.0) {
        self.draggingHandle = 0;
    } else if (distMax < 15.0) {
        self.draggingHandle = 1;
    }
}

- (void)mouseDragged:(NSEvent *)e {
    NSPoint pt;
    CGFloat trackX;
    CGFloat trackW;
    double valD;
    int val;

    if (self.draggingHandle < 0) {
        return;
    }

    pt = [self convertPoint:[e locationInWindow] fromView:nil];
    trackX = 10.0;
    trackW = [self bounds].size.width - 20.0;

    valD = (pt.x - trackX) / trackW * 8.0;
    val = LroundInt(valD);
    val = GOL_MIN(GOL_MAX(val, 0), 8);

    if (self.draggingHandle == 0) {
        self.minValue = GOL_MIN(val, self.maxValue);
    } else {
        self.maxValue = GOL_MAX(val, self.minValue);
    }

    [self setNeedsDisplay:YES];

    if (self.rangeChanged != nil) {
        self.rangeChanged();
    }
}

- (void)mouseUp:(NSEvent *)e {
    (void)e;
    self.draggingHandle = -1;
}

@end

@interface App : NSObject <NSApplicationDelegate, NSWindowDelegate, MTKViewDelegate>
@property (nonatomic, strong) NSWindow *window;
@property (nonatomic, strong) GOLView *mtkView;
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
@property (nonatomic, strong) id<MTLLibrary> library;
@property (nonatomic, strong) id<MTLRenderPipelineState> renderPipeline;
@property (nonatomic, strong) id<MTLRenderPipelineState> scalePipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> stepPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> trailStepPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> trailClearPipeline;
@property (nonatomic, strong) id<MTLTexture> cellTex;
@property (nonatomic, strong) id<MTLTexture> trailTex;
@property (nonatomic, strong) id<MTLBuffer> gridBuf;
@property (nonatomic, strong) id<MTLBuffer> uniformsBuf;
@property (nonatomic, strong) id<MTLCommandBuffer> cb0;
@property (nonatomic, strong) id<MTLCommandBuffer> cb1;
@property (nonatomic, strong) id<MTLCommandBuffer> cb2;
@property (nonatomic, strong) id<MTLCommandBuffer> lastCB;
@property (nonatomic, assign) uint64_t frameIndex;
@property (nonatomic, assign) uint32_t displayPlane;
@property (nonatomic, assign) int gridW;
@property (nonatomic, assign) int gridH;
@property (nonatomic, assign) int maxGridW;
@property (nonatomic, assign) int maxGridH;
@property (nonatomic, assign) NSUInteger planeCells;
@property (nonatomic, assign) NSUInteger planeBytes;
@property (nonatomic, assign) uint16_t *resizeTmp;
@property (nonatomic, assign) BOOL running;
@property (nonatomic, assign) BOOL dirty;
@property (nonatomic, assign) BOOL needsGridResize;
@property (nonatomic, assign) uint32_t gen;
@property (nonatomic, assign) CFTimeInterval fpsWindowStart;
@property (nonatomic, assign) CFTimeInterval simLastTime;
@property (nonatomic, assign) uint32_t fpsFrames;
@property (nonatomic, assign) GOLRules rules;
@property (nonatomic, strong) NSSlider *densitySlider;
@property (nonatomic, strong) NSTextField *densityPct;
@property (nonatomic, strong) NSButton *goButton;
@property (nonatomic, strong) NSTextField *genLabel;
@property (nonatomic, strong) NSTextField *fpsLabel;
@property (nonatomic, strong) NSTextField *ruleLabel;
@property (nonatomic, strong) NSTextField *popLabel;
@property (nonatomic, strong) NSTextField *maxAgeLabel;
@property (nonatomic, strong) NSSlider *speedSlider;
@property (nonatomic, strong) NSTextField *speedLabel;
@property (nonatomic, strong) GOLRangeSlider *birthRangeView;
@property (nonatomic, strong) GOLRangeSlider *survRangeView;
@property (nonatomic, assign) CFTimeInterval genAccum;
@property (nonatomic, assign) CGFloat cellPx;
@property (nonatomic, assign) CGFloat viewOffsetX;
@property (nonatomic, assign) CGFloat viewOffsetY;
@property (nonatomic, assign) int tool;
@property (nonatomic, assign) int brushRadius;
@property (nonatomic, assign) BOOL panning;
@property (nonatomic, assign) CGFloat lastPanX;
@property (nonatomic, assign) CGFloat lastPanY;
@property (nonatomic, assign) uint32_t displayMode;
@property (nonatomic, strong) NSPopUpButton *displayPopup;
@property (nonatomic, strong) NSSegmentedControl *toolControl;
@property (nonatomic, strong) NSSlider *brushSlider;
@property (nonatomic, strong) NSTextField *brushLabel;
@property (nonatomic, strong) NSTextField *zoomLabel;
@property (nonatomic, strong) NSTextField *hoverLabel;
@property (nonatomic, strong) NSTextField *hintLabel;
@property (nonatomic, strong) id keyMonitor;

- (BOOL)setupMetal;
- (void)setupUI;
- (void)randomize;
- (void)goPause:(id)sender;
- (void)clear:(id)sender;
- (void)clear;
- (void)fit:(id)sender;
- (void)fitView;
- (void)updateZoomLabel;
- (void)updateHintLabel;
- (void)toolChanged:(id)sender;
- (void)displayChanged:(id)sender;
- (void)clearTrailNow;
- (void)beginPanAtEvent:(NSEvent *)e;
- (void)panWithEvent:(NSEvent *)e;
- (void)endPan;
- (void)zoomAtEvent:(NSEvent *)e;
- (void)hoverAtEvent:(NSEvent *)e;
- (void)hoverExited;
- (void)updateHoverAtPoint:(NSPoint)pt;
- (NSPoint)topPointForEvent:(NSEvent *)e;
- (BOOL)gridCellAtPoint:(NSPoint)pt col:(int *)outCol row:(int *)outRow;
- (BOOL)handleKey:(NSEvent *)event;
- (void)sliderChanged:(id)sender;
- (void)paintAtEvent:(NSEvent *)e add:(BOOL)add;
- (void)waitLast;
- (void)waitAll;
- (uint16_t *)planePointer:(uint32_t)plane;
- (uint16_t *)currentCells;
- (void)rebuildCellTexture;
- (void)updateGridForPixelSize:(CGSize)pixelSize;
- (void)requestGridResize;
- (void)markDirty;
- (void)tick;
- (void)drawInMTKView:(MTKView *)view;
- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size;
- (void)ruleSliderChanged:(id)sender;
- (void)rulePresetClicked:(id)sender;
- (void)applyPreset:(const char *)name;
- (void)updateRuleUI;
- (id<MTLCommandBuffer>)cbAt:(uint32_t)plane;
- (void)setCB:(id<MTLCommandBuffer>)cb at:(uint32_t)plane;
- (void)clearCBRing;
@end

@interface GOLView : MTKView
@property (nonatomic, weak) App *owner;
@end

@implementation GOLView

@synthesize owner = _owner;

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (BOOL)becomeFirstResponder {
    return YES;
}

- (void)mouseDown:(NSEvent *)e {
    App *owner = self.owner;
    [self becomeFirstResponder];
    if (owner == nil) {
        return;
    }
    if (([e modifierFlags] & NSEventModifierFlagOption) != 0) {
        [owner beginPanAtEvent:e];
    } else {
        [owner paintAtEvent:e add:(owner.tool == 0)];
    }
}

- (void)mouseDragged:(NSEvent *)e {
    App *owner = self.owner;
    if (owner == nil) {
        return;
    }
    if (owner.panning) {
        [owner panWithEvent:e];
    } else {
        [owner paintAtEvent:e add:(owner.tool == 0)];
    }
}

- (void)mouseUp:(NSEvent *)e {
    App *owner = self.owner;
    (void)e;
    [owner endPan];
    [self becomeFirstResponder];
}

- (void)rightMouseDown:(NSEvent *)e {
    App *owner = self.owner;
    [self becomeFirstResponder];
    if (owner != nil) {
        [owner paintAtEvent:e add:(owner.tool == 1)];
    }
}

- (void)rightMouseDragged:(NSEvent *)e {
    App *owner = self.owner;
    if (owner != nil) {
        [owner paintAtEvent:e add:(owner.tool == 1)];
    }
}

- (void)otherMouseDown:(NSEvent *)e {
    App *owner = self.owner;
    [self becomeFirstResponder];
    if (owner != nil) {
        [owner beginPanAtEvent:e];
    }
}

- (void)otherMouseDragged:(NSEvent *)e {
    App *owner = self.owner;
    if (owner != nil && owner.panning) {
        [owner panWithEvent:e];
    }
}

- (void)otherMouseUp:(NSEvent *)e {
    App *owner = self.owner;
    (void)e;
    [owner endPan];
    [self becomeFirstResponder];
}

- (void)scrollWheel:(NSEvent *)e {
    App *owner = self.owner;
    if (owner != nil) {
        [owner zoomAtEvent:e];
    }
}

- (void)mouseMoved:(NSEvent *)e {
    App *owner = self.owner;
    if (owner != nil) {
        [owner hoverAtEvent:e];
    }
}

- (void)mouseExited:(NSEvent *)e {
    App *owner = self.owner;
    (void)e;
    if (owner != nil) {
        [owner hoverExited];
    }
}

@end

@implementation App

@synthesize window = _window;
@synthesize mtkView = _mtkView;
@synthesize device = _device;
@synthesize queue = _queue;
@synthesize library = _library;
@synthesize renderPipeline = _renderPipeline;
@synthesize scalePipeline = _scalePipeline;
@synthesize stepPipeline = _stepPipeline;
@synthesize trailStepPipeline = _trailStepPipeline;
@synthesize trailClearPipeline = _trailClearPipeline;
@synthesize cellTex = _cellTex;
@synthesize trailTex = _trailTex;
@synthesize gridBuf = _gridBuf;
@synthesize uniformsBuf = _uniformsBuf;
@synthesize cb0 = _cb0;
@synthesize cb1 = _cb1;
@synthesize cb2 = _cb2;
@synthesize lastCB = _lastCB;
@synthesize frameIndex = _frameIndex;
@synthesize displayPlane = _displayPlane;
@synthesize gridW = _gridW;
@synthesize gridH = _gridH;
@synthesize maxGridW = _maxGridW;
@synthesize maxGridH = _maxGridH;
@synthesize planeCells = _planeCells;
@synthesize planeBytes = _planeBytes;
@synthesize resizeTmp = _resizeTmp;
@synthesize running = _running;
@synthesize dirty = _dirty;
@synthesize needsGridResize = _needsGridResize;
@synthesize gen = _gen;
@synthesize fpsWindowStart = _fpsWindowStart;
@synthesize simLastTime = _simLastTime;
@synthesize fpsFrames = _fpsFrames;
@synthesize rules = _rules;
@synthesize densitySlider = _densitySlider;
@synthesize densityPct = _densityPct;
@synthesize goButton = _goButton;
@synthesize genLabel = _genLabel;
@synthesize fpsLabel = _fpsLabel;
@synthesize ruleLabel = _ruleLabel;
@synthesize popLabel = _popLabel;
@synthesize maxAgeLabel = _maxAgeLabel;
@synthesize speedSlider = _speedSlider;
@synthesize speedLabel = _speedLabel;
@synthesize birthRangeView = _birthRangeView;
@synthesize survRangeView = _survRangeView;
@synthesize genAccum = _genAccum;
@synthesize cellPx = _cellPx;
@synthesize viewOffsetX = _viewOffsetX;
@synthesize viewOffsetY = _viewOffsetY;
@synthesize tool = _tool;
@synthesize brushRadius = _brushRadius;
@synthesize panning = _panning;
@synthesize lastPanX = _lastPanX;
@synthesize lastPanY = _lastPanY;
@synthesize displayMode = _displayMode;
@synthesize displayPopup = _displayPopup;
@synthesize toolControl = _toolControl;
@synthesize brushSlider = _brushSlider;
@synthesize brushLabel = _brushLabel;
@synthesize zoomLabel = _zoomLabel;
@synthesize hoverLabel = _hoverLabel;
@synthesize hintLabel = _hintLabel;
@synthesize keyMonitor = _keyMonitor;

- (void)applyLaunchConfig {
    NSArray<NSString *> *args;
    NSString *mode = nil;
    NSString *preset = nil;
    const char *env;
    double val;
    NSUInteger i;
    args = [[NSProcessInfo processInfo] arguments];
    for (i = 1; i < args.count; i++) {
        NSString *a = args[i];
        if ([a hasPrefix:@"-"]) continue;
        if (i + 1 < args.count) {
            NSString *n = args[i + 1];
            if ([a caseInsensitiveCompare:@"--mode"] == NSOrderedSame) { mode = n; i++; }
            else if ([a caseInsensitiveCompare:@"--preset"] == NSOrderedSame) { preset = n; i++; }
        }
    }
    if (mode == nil && (env = getenv("GOL_MODE")) != NULL) mode = [NSString stringWithUTF8String:env];
    if (preset == nil && (env = getenv("GOL_PRESET")) != NULL) preset = [NSString stringWithUTF8String:env];
    if ((env = getenv("GOL_DENSITY")) != NULL && self.densitySlider != nil) {
        val = atof(env);
        if (val < 0.0) val = 0.0;
        if (val > 1.0) val = 1.0;
        self.densitySlider.doubleValue = val;
    }
    if ((env = getenv("GOL_ZOOM")) != NULL) { val = atof(env); if (val > 0.0) self.cellPx = val; }
    if ((env = getenv("GOL_RUN")) != NULL) self.running = atoi(env) != 0;
    if (mode != nil && mode.length > 0) {
        NSString *m = [mode lowercaseString];
        if ([m isEqualToString:@"age"]) self.displayMode = DISPLAY_AGE;
        else if ([m isEqualToString:@"trails"]) self.displayMode = DISPLAY_TRAILS;
        else if ([m isEqualToString:@"heatmap"]) self.displayMode = DISPLAY_HEATMAP;
    }
    if (preset != nil && preset.length > 0) {
        [self applyPreset:[preset UTF8String]];
    } else {
        [self randomize];
    }
}

- (void)applicationDidFinishLaunching:(NSNotification *)note {
    CFTimeInterval now;
    (void)note;
    now = CFAbsoluteTimeGetCurrent();
    self.frameIndex = 0;
    self.displayPlane = 0;
    self.gridW = INITIAL_GRID_W;
    self.gridH = INITIAL_GRID_H;
    self.gen = 0;
    self.dirty = NO;
    self.running = NO;
    self.fpsWindowStart = now;
    self.simLastTime = now;
    self.fpsFrames = 0;
    self.genAccum = 0;
    self.rules = gol_default_rules();
    self.cellPx = CELL_PX;
    self.viewOffsetX = 0.0;
    self.viewOffsetY = 0.0;
    self.tool = 0;
    self.brushRadius = 1;
    self.panning = NO;
    self.lastPanX = 0.0;
    self.lastPanY = 0.0;
    self.displayMode = DISPLAY_AGE;
    if (![self setupMetal]) {
        NSLog(@"Metal setup failed");
        [NSApp terminate:nil];
        return;
    }
    [self setupUI];
    [self applyLaunchConfig];
}

- (BOOL)setupMetal {
    NSString *exeDir;
    NSString *libPath;
    NSURL *libURL;
    NSError *err;
    NSScreen *screen;
    CGFloat screenW;
    CGFloat screenH;
    NSUInteger need;
    MTLRenderPipelineDescriptor *rp;
    MTLRenderPipelineDescriptor *sp;
    id<MTLFunction> stepFunc;
    id<MTLFunction> trailStepFunc;
    id<MTLFunction> trailClearFunc;

    self.device = MTLCreateSystemDefaultDevice();
    if (self.device == nil) {
        return NO;
    }
    self.queue = [self.device newCommandQueue];
    if (self.queue == nil) {
        return NO;
    }

    exeDir = [[NSBundle mainBundle] executablePath];
    exeDir = [exeDir stringByDeletingLastPathComponent];
    libPath = [exeDir stringByAppendingPathComponent:@"shaders.metallib"];
    libURL = [NSURL fileURLWithPath:libPath];
    err = nil;
    self.library = [self.device newLibraryWithURL:libURL error:&err];
    if (self.library == nil) {
        NSLog(@"failed to load metallib: %@", err);
        return NO;
    }

    self.mtkView = [[GOLView alloc] initWithFrame:NSMakeRect(0, BAR_H, VIEW_W, VIEW_H)
                                           device:self.device];
    self.mtkView.wantsLayer = YES;
    self.mtkView.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    self.mtkView.framebufferOnly = YES;
    self.mtkView.autoResizeDrawable = YES;
    self.mtkView.preferredFramesPerSecond = 120;
    self.mtkView.paused = NO;
    self.mtkView.owner = self;

    screen = ScreenForWindow(self.mtkView.window);
    screenW = screen.frame.size.width;
    screenH = screen.frame.size.height;
    self.maxGridW = FloorInt(screenW / MIN_CELL_PX) + 16;
    self.maxGridH = FloorInt(GOL_MAX(0.0, screenH - (CGFloat)BAR_H) / MIN_CELL_PX) + 16;
    self.maxGridW = GOL_MAX(self.maxGridW, INITIAL_GRID_W);
    self.maxGridH = GOL_MAX(self.maxGridH, INITIAL_GRID_H);
    self.maxGridW = GOL_MIN(self.maxGridW, MAX_TEXTURE_SIZE / (int)RENDER_SCALE);
    self.maxGridH = GOL_MIN(self.maxGridH, MAX_TEXTURE_SIZE / (int)RENDER_SCALE);
    need = (NSUInteger)self.maxGridW * (NSUInteger)self.maxGridH;
    if (need > MAX_PLANE_CELLS) {
        double shrink = sqrt((double)MAX_PLANE_CELLS / (double)need);
        self.maxGridW = GOL_MAX(8, FloorInt((double)self.maxGridW * shrink));
        self.maxGridH = GOL_MAX(8, FloorInt((double)self.maxGridH * shrink));
        need = (NSUInteger)self.maxGridW * (NSUInteger)self.maxGridH;
    }
    self.planeCells = ((need + 7) / 8) * 8;
    self.planeBytes = self.planeCells * sizeof(uint16_t);
    self.resizeTmp = (uint16_t *)malloc(self.planeBytes);
    if (self.resizeTmp == nil) {
        return NO;
    }

    self.gridBuf = [self.device newBufferWithLength:(NSUInteger)PLANE_COUNT * self.planeBytes
                                            options:MTLResourceStorageModeShared];
    if (self.gridBuf == nil) {
        return NO;
    }
    memset([self.gridBuf contents], 0, (size_t)PLANE_COUNT * self.planeBytes);

    self.uniformsBuf = [self.device newBufferWithLength:sizeof(Uniforms) options:MTLResourceStorageModeShared];
    if (self.uniformsBuf == nil) {
        return NO;
    }

    self.cb0 = nil;
    self.cb1 = nil;
    self.cb2 = nil;
    self.lastCB = nil;

    rp = [MTLRenderPipelineDescriptor new];
    rp.vertexFunction = [self.library newFunctionWithName:@"vs_main"];
    rp.fragmentFunction = [self.library newFunctionWithName:@"fs_main"];
    rp.colorAttachments[0].pixelFormat = self.mtkView.colorPixelFormat;
    err = nil;
    self.renderPipeline = [self.device newRenderPipelineStateWithDescriptor:rp error:&err];
    if (self.renderPipeline == nil) {
        NSLog(@"failed to create render pipeline: %@", err);
        return NO;
    }

    sp = [MTLRenderPipelineDescriptor new];
    sp.vertexFunction = [self.library newFunctionWithName:@"vs_main"];
    sp.fragmentFunction = [self.library newFunctionWithName:@"fs_scale"];
    sp.colorAttachments[0].pixelFormat = self.mtkView.colorPixelFormat;
    err = nil;
    self.scalePipeline = [self.device newRenderPipelineStateWithDescriptor:sp error:&err];
    if (self.scalePipeline == nil) {
        NSLog(@"failed to create scale pipeline: %@", err);
        return NO;
    }

    stepFunc = [self.library newFunctionWithName:@"gol_step"];
    err = nil;
    self.stepPipeline = [self.device newComputePipelineStateWithFunction:stepFunc error:&err];
    if (self.stepPipeline == nil) {
        NSLog(@"compute pipeline unavailable; using CPU fallback");
        self.stepPipeline = nil;
    }

    trailStepFunc = [self.library newFunctionWithName:@"trail_step"];
    err = nil;
    if (trailStepFunc != nil) {
        self.trailStepPipeline = [self.device newComputePipelineStateWithFunction:trailStepFunc error:&err];
    } else {
        self.trailStepPipeline = nil;
    }

    trailClearFunc = [self.library newFunctionWithName:@"trail_clear"];
    err = nil;
    if (trailClearFunc != nil) {
        self.trailClearPipeline = [self.device newComputePipelineStateWithFunction:trailClearFunc error:&err];
    } else {
        self.trailClearPipeline = nil;
    }

    [self rebuildCellTexture];
    if (self.cellTex == nil) {
        return NO;
    }

    self.mtkView.delegate = self;
    return YES;
}

- (void)updateRuleUI {
    GOLRules r;
    uint8_t b;
    uint8_t s;
    int bMin;
    int bMax;
    int sMin;
    int sMax;
    int i;
    GOLRangeSlider *bv;
    GOLRangeSlider *sv;
    NSMutableString *label;

    r = self.rules;
    b = r.birth;
    s = r.survival;

    bMin = 8;
    bMax = 0;
    for (i = 0; i < 9; i++) {
        if ((b >> i) & 1u) {
            bMin = GOL_MIN(bMin, i);
            bMax = GOL_MAX(bMax, i);
        }
    }
    if (bMin > bMax) {
        bMin = 3;
        bMax = 3;
    }

    sMin = 8;
    sMax = 0;
    for (i = 0; i < 9; i++) {
        if ((s >> i) & 1u) {
            sMin = GOL_MIN(sMin, i);
            sMax = GOL_MAX(sMax, i);
        }
    }
    if (sMin > sMax) {
        sMin = 2;
        sMax = 3;
    }

    bv = self.birthRangeView;
    sv = self.survRangeView;
    dispatch_async(dispatch_get_main_queue(), ^{
        if (bv != nil) {
            [bv setRange:bMin max:bMax];
        }
        if (sv != nil) {
            [sv setRange:sMin max:sMax];
        }
    });

    label = [NSMutableString string];
    [label appendString:@"B"];
    for (i = 0; i < 9; i++) {
        if ((b >> i) & 1u) {
            [label appendFormat:@"%d", i];
        }
    }
    [label appendString:@"/S"];
    for (i = 0; i < 9; i++) {
        if ((s >> i) & 1u) {
            [label appendFormat:@"%d", i];
        }
    }
    if (self.ruleLabel != nil) {
        self.ruleLabel.stringValue = label;
    }
}

- (void)applyPreset:(const char *)name {
    uint16_t *cells;
    uint16_t *base;
    uint32_t p;

    [self waitAll];
    self.frameIndex = 0;
    self.displayPlane = 0;
    self.gen = 0;
    cells = [self currentCells];
    if (cells == nil) {
        return;
    }
    gol_apply_preset(cells, self.gridW, self.gridH, name);
    base = (uint16_t *)[self.gridBuf contents];
    for (p = 1; p < (uint32_t)PLANE_COUNT; p++) {
        memset(base + (size_t)p * (size_t)self.planeCells, 0, self.planeBytes);
    }
    if (self.genLabel != nil) {
        self.genLabel.stringValue = @"Gen 0";
    }
    [self markDirty];
}

- (void)rulePresetClicked:(id)sender {
    NSButton *btn = (NSButton *)sender;
    NSString *preset = [btn title];
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

- (void)ruleSliderChanged:(id)sender {
    GOLRules r;
    int bMin;
    int bMax;
    int sMin;
    int sMax;
    int i;
    int t;

    (void)sender;

    bMin = self.birthRangeView ? [self.birthRangeView minValue] : 3;
    bMax = self.birthRangeView ? [self.birthRangeView maxValue] : 3;
    sMin = self.survRangeView ? [self.survRangeView minValue] : 2;
    sMax = self.survRangeView ? [self.survRangeView maxValue] : 3;

    if (bMin > bMax) {
        t = bMin;
        bMin = bMax;
        bMax = t;
    }
    if (sMin > sMax) {
        t = sMin;
        sMin = sMax;
        sMax = t;
    }

    r = self.rules;
    r.birth = 0;
    for (i = bMin; i <= bMax; i++) {
        r.birth = (uint8_t)(r.birth | (1u << i));
    }
    r.survival = 0;
    for (i = sMin; i <= sMax; i++) {
        r.survival = (uint8_t)(r.survival | (1u << i));
    }
    self.rules = r;

    [self updateRuleUI];
}

- (void)randomize {
    double density;
    uint16_t *cells;
    uint16_t *base;
    uint32_t p;
    int pct;

    density = self.densitySlider ? self.densitySlider.doubleValue : 0.2;
    if (density < 0.0) {
        density = 0.0;
    }
    if (density > 1.0) {
        density = 1.0;
    }
    [self waitAll];
    self.frameIndex = 0;
    self.displayPlane = 0;
    self.gen = 0;
    self.genAccum = 0;
    [self clearTrailNow];
    cells = [self currentCells];
    if (cells == nil) {
        return;
    }
    gol_randomize(cells, self.planeCells, self.gridW, self.gridH, density);
    base = (uint16_t *)[self.gridBuf contents];
    for (p = 1; p < (uint32_t)PLANE_COUNT; p++) {
        memset(base + (size_t)p * (size_t)self.planeCells, 0, self.planeBytes);
    }
    pct = LroundInt(density * 100.0);
    if (self.densityPct != nil) {
        self.densityPct.stringValue = [NSString stringWithFormat:@"%d%%", pct];
    }
    if (self.genLabel != nil) {
        self.genLabel.stringValue = @"Gen 0";
    }
    [self markDirty];
}

- (void)goPause:(id)sender {
    (void)sender;
    if (self.goButton == nil || self.mtkView == nil) {
        return;
    }
    self.running = !self.running;
    self.goButton.title = self.running ? @"Pause" : @"Go";
    if (self.running) {
        self.mtkView.paused = NO;
    } else if (!self.dirty) {
        self.mtkView.paused = YES;
    }
}

- (void)clear:(id)sender {
    (void)sender;
    [self clear];
}

- (void)clear {
    uint16_t *cells;
    uint16_t *base;
    uint32_t p;

    [self waitAll];
    self.frameIndex = 0;
    self.displayPlane = 0;
    self.gen = 0;
    self.genAccum = 0;
    [self clearTrailNow];
    cells = [self currentCells];
    if (cells == nil) {
        return;
    }
    memset(cells, 0, self.planeBytes);
    base = (uint16_t *)[self.gridBuf contents];
    for (p = 1; p < (uint32_t)PLANE_COUNT; p++) {
        memset(base + (size_t)p * (size_t)self.planeCells, 0, self.planeBytes);
    }
    if (self.genLabel != nil) {
        self.genLabel.stringValue = @"Gen 0";
    }
    [self markDirty];
}

- (void)clearTrailNow {
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    NSUInteger tgW;
    NSUInteger tgH;
    NSUInteger gx;
    NSUInteger gy;

    if (self.trailTex == nil || self.trailClearPipeline == nil ||
        self.queue == nil) {
        return;
    }
    [self waitLast];
    cb = [self.queue commandBuffer];
    if (cb == nil) {
        return;
    }
    enc = [cb computeCommandEncoder];
    if (enc == nil) {
        return;
    }
    [enc setComputePipelineState:self.trailClearPipeline];
    [enc setTexture:self.trailTex atIndex:0];
    tgW = 16;
    tgH = 16;
    gx = (self.trailTex.width + tgW - 1) / tgW;
    gy = (self.trailTex.height + tgH - 1) / tgH;
    [enc dispatchThreadgroups:MakeSize((int)gx, (int)gy, 1)
      threadsPerThreadgroup:MakeSize((int)tgW, (int)tgH, 1)];
    [enc endEncoding];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
        if (completedBuffer.error != nil) {
            NSLog(@"trail clear error: %@", completedBuffer.error);
        }
    }];
    [cb commit];
    [cb waitUntilCompleted];
}

- (void)fit:(id)sender {
    (void)sender;
    [self fitView];
}

- (void)fitView {
    NSRect bounds;
    double scaleW;
    double scaleH;
    double s;

    if (self.mtkView == nil || self.gridW < 1 || self.gridH < 1) {
        return;
    }
    bounds = [self.mtkView bounds];
    if (bounds.size.width < 1.0 || bounds.size.height < 1.0) {
        return;
    }
    scaleW = bounds.size.width / (double)self.gridW;
    scaleH = bounds.size.height / (double)self.gridH;
    s = GOL_MIN(scaleW, scaleH);
    s = ClampDouble(s, 0.01, (double)MAX_CELL_PX);
    self.cellPx = (CGFloat)s;
    self.viewOffsetX = (CGFloat)((bounds.size.width - (double)self.gridW * s) / 2.0);
    self.viewOffsetY = (CGFloat)((bounds.size.height - (double)self.gridH * s) / 2.0);
    [self updateZoomLabel];
    [self requestGridResize];
}

- (void)updateZoomLabel {
    int pct;
    if (self.zoomLabel == nil) {
        return;
    }
    pct = LroundInt((self.cellPx / CELL_PX) * 100.0);
    self.zoomLabel.stringValue = [NSString stringWithFormat:@"%d%%", pct];
}

- (void)updateHintLabel {
    NSString *left;
    NSString *right;
    NSString *text;
    if (self.hintLabel == nil) {
        return;
    }
    left = (self.tool == 0) ? @"add" : @"erase";
    right = (self.tool == 0) ? @"erase" : @"add";
    text = [NSString stringWithFormat:@"L:%@ R:%@ M:pan Scroll:zoom", left, right];
    self.hintLabel.stringValue = text;
}

- (void)toolChanged:(id)sender {
    (void)sender;
    if (self.toolControl != nil && self.toolControl.selectedSegment >= 0) {
        self.tool = (int)self.toolControl.selectedSegment;
    } else {
        self.tool = 0;
    }
    [self updateHintLabel];
}

- (void)displayChanged:(id)sender {
    int index;
    (void)sender;
    if (self.displayPopup == nil) {
        return;
    }
    index = (int)self.displayPopup.indexOfSelectedItem;
    if (index < 0 || index > (int)DISPLAY_HEATMAP) {
        index = 0;
    }
    self.displayMode = (uint32_t)index;
    if (self.displayMode == DISPLAY_TRAILS) {
        [self clearTrailNow];
    }
    [self markDirty];
}

- (NSPoint)topPointForEvent:(NSEvent *)e {
    NSPoint pt;
    NSPoint result;
    CGFloat viewH;
    pt = [self.mtkView convertPoint:[e locationInWindow] fromView:nil];
    viewH = self.mtkView.bounds.size.height;
    if ([self.mtkView isFlipped]) {
        result = pt;
    } else {
        result = NSMakePoint(pt.x, viewH - pt.y);
    }
    return result;
}

- (BOOL)gridCellAtPoint:(NSPoint)pt col:(int *)outCol row:(int *)outRow {
    double colD;
    double rowD;
    int col;
    int row;

    if (self.cellPx < 0.01) {
        return NO;
    }
    colD = (pt.x - self.viewOffsetX) / self.cellPx;
    rowD = (pt.y - self.viewOffsetY) / self.cellPx;
    col = FloorInt(colD);
    row = FloorInt(rowD);
    if (col < 0 || col >= self.gridW || row < 0 || row >= self.gridH) {
        return NO;
    }
    if (outCol != NULL) {
        *outCol = col;
    }
    if (outRow != NULL) {
        *outRow = row;
    }
    return YES;
}

- (void)beginPanAtEvent:(NSEvent *)e {
    NSPoint pt;
    pt = [self topPointForEvent:e];
    self.panning = YES;
    self.lastPanX = pt.x;
    self.lastPanY = pt.y;
}

- (void)panWithEvent:(NSEvent *)e {
    NSPoint pt;
    if (!self.panning) {
        return;
    }
    pt = [self topPointForEvent:e];
    self.viewOffsetX += pt.x - self.lastPanX;
    self.viewOffsetY += pt.y - self.lastPanY;
    self.lastPanX = pt.x;
    self.lastPanY = pt.y;
    [self markDirty];
}

- (void)endPan {
    self.panning = NO;
}

- (void)zoomAtEvent:(NSEvent *)e {
    NSPoint pt;
    double gx;
    double gy;
    double dy;
    double factor;
    double newCellPx;

    if (self.mtkView == nil || self.cellPx < 0.01) {
        return;
    }
    pt = [self topPointForEvent:e];
    gx = (pt.x - self.viewOffsetX) / self.cellPx;
    gy = (pt.y - self.viewOffsetY) / self.cellPx;
    dy = e.scrollingDeltaY * (e.hasPreciseScrollingDeltas ? 0.02 : 0.1);
    factor = exp2(dy);
    factor = ClampDouble(factor, 0.5, 2.0);
    newCellPx = (double)self.cellPx * factor;
    newCellPx = ClampDouble(newCellPx, (double)MIN_CELL_PX, (double)MAX_CELL_PX);
    self.cellPx = (CGFloat)newCellPx;
    self.viewOffsetX = pt.x - (CGFloat)gx * self.cellPx;
    self.viewOffsetY = pt.y - (CGFloat)gy * self.cellPx;
    [self updateZoomLabel];
    [self requestGridResize];
}

- (void)hoverAtEvent:(NSEvent *)e {
    NSPoint pt;
    pt = [self topPointForEvent:e];
    [self updateHoverAtPoint:pt];
}

- (void)hoverExited {
    if (self.hoverLabel != nil) {
        self.hoverLabel.stringValue = @"--";
    }
}

- (void)updateHoverAtPoint:(NSPoint)pt {
    int col;
    int row;
    uint16_t *cells;
    uint16_t v;
    size_t idx;
    NSString *text;

    col = 0;
    row = 0;
    if (self.hoverLabel == nil) {
        return;
    }
    if (![self gridCellAtPoint:pt col:&col row:&row]) {
        self.hoverLabel.stringValue = @"--";
        return;
    }
    cells = [self planePointer:self.displayPlane];
    if (cells == nil) {
        self.hoverLabel.stringValue = @"--";
        return;
    }
    idx = (size_t)row * (size_t)self.gridW + (size_t)col;
    v = cells[idx];
    if (GolAlive(v)) {
        text = [NSString stringWithFormat:@"%d,%d alive age %d", col, row, (int)GolAge(v)];
    } else {
        text = [NSString stringWithFormat:@"%d,%d dead", col, row];
    }
    self.hoverLabel.stringValue = text;
}

- (BOOL)handleKey:(NSEvent *)event {
    NSEventModifierFlags flags;

    flags = event.modifierFlags;
    if ((flags & (NSEventModifierFlagCommand | NSEventModifierFlagControl)) != 0) {
        return NO;
    }
    if ([self.window firstResponder] != self.mtkView) {
        return NO;
    }
    switch (event.keyCode) {
        case 49:
            [self goPause:nil];
            return YES;
        case 15:
            [self clear];
            return YES;
        case 6:
            [self randomize];
            return YES;
        case 3:
        case 29:
            [self fitView];
            return YES;
        default:
            return NO;
    }
}

- (void)sliderChanged:(id)sender {
    int val;
    if (sender == self.brushSlider) {
        self.brushRadius = FloorInt(self.brushSlider.doubleValue);
        if (self.brushLabel != nil) {
            self.brushLabel.stringValue = [NSString stringWithFormat:@"%d", self.brushRadius];
        }
        return;
    }
    if (sender == self.speedSlider) {
        val = FloorInt(self.speedSlider.doubleValue);
        if (self.speedLabel != nil) {
            self.speedLabel.stringValue = [NSString stringWithFormat:@"%d gen/s", val];
        }
        return;
    }
    [self randomize];
}

- (void)paintAtEvent:(NSEvent *)e add:(BOOL)add {
    NSPoint pt;
    int col;
    int row;
    int dx;
    int dy;
    int cx;
    int cy;
    uint16_t *cells;

    if (self.mtkView == nil) {
        return;
    }
    [self waitAll];
    pt = [self topPointForEvent:e];
    if (![self gridCellAtPoint:pt col:&col row:&row]) {
        return;
    }
    cells = [self currentCells];
    if (cells == nil) {
        return;
    }
    for (dy = -self.brushRadius; dy <= self.brushRadius; dy++) {
        for (dx = -self.brushRadius; dx <= self.brushRadius; dx++) {
            cx = col + dx;
            cy = row + dy;
            if (cx >= 0 && cx < self.gridW && cy >= 0 && cy < self.gridH) {
                gol_set_plane(cells, self.gridW, self.gridH, cx, cy, add);
            }
        }
    }
    [self markDirty];
    [self updateHoverAtPoint:pt];
}

- (void)waitLast {
    if (self.lastCB != nil) {
        [self.lastCB waitUntilCompleted];
        if (self.lastCB.error != nil) {
            NSLog(@"command buffer error: %@", self.lastCB.error);
        }
        self.lastCB = nil;
    }
}

- (void)waitAll {
    id<MTLCommandBuffer> inflight;

    [self waitLast];
    inflight = self.cb0;
    if (inflight != nil) {
        [inflight waitUntilCompleted];
    }
    inflight = self.cb1;
    if (inflight != nil) {
        [inflight waitUntilCompleted];
    }
    inflight = self.cb2;
    if (inflight != nil) {
        [inflight waitUntilCompleted];
    }
    [self clearCBRing];
}

- (uint16_t *)planePointer:(uint32_t)plane {
    uint16_t *base;
    if (self.gridBuf == nil) {
        return nil;
    }
    base = (uint16_t *)[self.gridBuf contents];
    return base + (size_t)plane * (size_t)self.planeCells;
}

- (uint16_t *)currentCells {
    return [self planePointer:(uint32_t)(self.frameIndex % (uint32_t)PLANE_COUNT)];
}

- (void)updateGridForPixelSize:(CGSize)pixelSize {
    NSRect bounds;
    double newWD;
    double newHD;
    int newW;
    int newH;
    double shrink;
    int oldW;
    int oldH;
    int srcX;
    int srcY;
    uint16_t *base;
    uint16_t *oldPlane;
    uint32_t p;

    (void)pixelSize;
    self.needsGridResize = NO;
    if (self.gridBuf == nil || self.mtkView == nil || self.resizeTmp == nil ||
        self.cellPx < 0.01) {
        return;
    }
    bounds = [self.mtkView bounds];
    if (bounds.size.width < 1.0 || bounds.size.height < 1.0) {
        return;
    }

    newWD = bounds.size.width / (double)self.cellPx;
    newHD = bounds.size.height / (double)self.cellPx;
    newW = CeilIntStable(newWD);
    newH = CeilIntStable(newHD);
    newW = GOL_MAX(8, GOL_MIN(newW, self.maxGridW));
    newH = GOL_MAX(8, GOL_MIN(newH, self.maxGridH));

    if ((NSUInteger)newW * (NSUInteger)newH > self.planeCells) {
        shrink = sqrt((double)self.planeCells / ((double)newW * (double)newH));
        newW = GOL_MAX(8, FloorInt((double)newW * shrink));
        newH = GOL_MAX(8, FloorInt((double)newH * shrink));
    }

    if (newW == self.gridW && newH == self.gridH) {
        return;
    }

    [self waitAll];

    oldW = self.gridW;
    oldH = self.gridH;
    srcX = FloorInt(-self.viewOffsetX / (double)self.cellPx);
    srcY = FloorInt(-self.viewOffsetY / (double)self.cellPx);
    base = (uint16_t *)[self.gridBuf contents];
    oldPlane = base + (size_t)self.displayPlane * (size_t)self.planeCells;

    gol_copy_region(oldPlane, oldW, oldH, self.resizeTmp, newW, newH,
                    self.planeCells, srcX, srcY);
    self.gridW = newW;
    self.gridH = newH;
    self.viewOffsetX = 0.0;
    self.viewOffsetY = 0.0;

    memcpy(base, self.resizeTmp, (size_t)self.planeCells * sizeof(uint16_t));
    for (p = 1; p < (uint32_t)PLANE_COUNT; p++) {
        memset(base + (size_t)p * (size_t)self.planeCells, 0, self.planeBytes);
    }

    self.frameIndex = 0;
    self.displayPlane = 0;
    [self rebuildCellTexture];
    [self clearTrailNow];
    [self markDirty];
}

- (void)requestGridResize {
    self.needsGridResize = YES;
    [self markDirty];
}

- (void)rebuildCellTexture {
    NSUInteger w;
    NSUInteger h;
    MTLTextureDescriptor *d;
    MTLTextureDescriptor *td;

    if (self.device == nil || self.mtkView == nil) {
        return;
    }
    w = (NSUInteger)GOL_MAX(1, self.gridW) * (NSUInteger)RENDER_SCALE;
    h = (NSUInteger)GOL_MAX(1, self.gridH) * (NSUInteger)RENDER_SCALE;
    w = GOL_MIN(w, (NSUInteger)MAX_TEXTURE_SIZE);
    h = GOL_MIN(h, (NSUInteger)MAX_TEXTURE_SIZE);
    d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:self.mtkView.colorPixelFormat
                                                            width:w
                                                           height:h
                                                       mipmapped:NO];
    d.usage = (MTLTextureUsage)(MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
    d.storageMode = MTLStorageModePrivate;
    self.cellTex = [self.device newTextureWithDescriptor:d];

    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                           width:w
                                                          height:h
                                                      mipmapped:NO];
    td.usage = (MTLTextureUsage)(MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    td.storageMode = MTLStorageModePrivate;
    self.trailTex = [self.device newTextureWithDescriptor:td];
}

- (void)markDirty {
    self.dirty = YES;
    if (self.mtkView != nil) {
        self.mtkView.paused = NO;
    }
}

- (void)setupUI {
    NSRect content;
    NSScreen *screen;
    NSView *contentView;
    NSView *bar;
    CGFloat x;
    CGFloat y;
    NSButton *clearButton;
    NSButton *fitButton;
    NSPopUpButton *presetsPopup;
    __weak App *weakSelf;

    content = NSMakeRect(0, 0, VIEW_W, VIEW_H + BAR_H);
    self.window = [[NSWindow alloc] initWithContentRect:content
                                              styleMask:(NSWindowStyleMask)(NSWindowStyleMaskTitled |
                                                                            NSWindowStyleMaskClosable |
                                                                            NSWindowStyleMaskMiniaturizable |
                                                                            NSWindowStyleMaskResizable)
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
    self.window.releasedWhenClosed = NO;
    [self.window setTitle:@"Game of Life"];
    [self.window setContentSize:content.size];
    [self.window setBackgroundColor:[NSColor colorWithSRGBRed:0.04 green:0.05 blue:0.08 alpha:1.0]];
    [self.window setDelegate:self];
    self.window.acceptsMouseMovedEvents = YES;
    [self.window center];

    screen = ScreenForWindow(self.window);
    self.window.minSize = NSMakeSize(500.0, 350.0 + (CGFloat)BAR_H);
    self.window.maxSize = NSMakeSize(screen.frame.size.width, screen.frame.size.height);

    contentView = [self.window contentView];
    self.mtkView.frame = NSMakeRect(0, BAR_H, VIEW_W, VIEW_H);
    self.mtkView.autoresizingMask = (NSAutoresizingMaskOptions)(NSViewWidthSizable | NSViewHeightSizable);
    [contentView addSubview:self.mtkView];

    bar = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, VIEW_W, BAR_H)];
    bar.wantsLayer = YES;
    bar.autoresizingMask = (NSAutoresizingMaskOptions)NSViewWidthSizable;
    bar.layer.backgroundColor = [NSColor colorWithSRGBRed:0.07 green:0.08 blue:0.11 alpha:1.0].CGColor;
    [contentView addSubview:bar];

    x = 12;
    y = 8;

    self.fpsLabel = MakeLabel(@"FPS --", NSMakeRect(x, y, 80, 18));
    [bar addSubview:self.fpsLabel];
    x += 90;

    [bar addSubview:MakeLabelSmall(@"B:", NSMakeRect(x, y - 2, 16, 22))];
    x += 20;
    self.birthRangeView = [[GOLRangeSlider alloc] initWithFrame:NSMakeRect(x, y - 4, 180, 28)
                                                       minValue:0
                                                            max:8
                                                     defaultMin:3
                                                     defaultMax:3];
    [bar addSubview:self.birthRangeView];
    x += 188;

    [bar addSubview:MakeLabelSmall(@"S:", NSMakeRect(x, y - 2, 16, 22))];
    x += 20;
    self.survRangeView = [[GOLRangeSlider alloc] initWithFrame:NSMakeRect(x, y - 4, 180, 28)
                                                      minValue:0
                                                           max:8
                                                    defaultMin:2
                                                    defaultMax:3];
    [bar addSubview:self.survRangeView];
    x += 188;

    x += 8;
    self.ruleLabel = MakeLabel(@"B3/S23", NSMakeRect(x, y - 2, 120, 22));
    [bar addSubview:self.ruleLabel];
    x += 110;

    self.popLabel = MakeLabelSmall(@"Pop: 0", NSMakeRect(x, y - 2, 90, 22));
    [bar addSubview:self.popLabel];
    x += 100;

    self.maxAgeLabel = MakeLabelSmall(@"Age: 0", NSMakeRect(x, y - 2, 70, 22));
    [bar addSubview:self.maxAgeLabel];
    x += 80;

    y = 38;
    x = 12;

    [bar addSubview:MakeLabelSmall(@"Density", NSMakeRect(x, y, 55, 18))];
    x += 60;

    self.densitySlider = [[NSSlider alloc] initWithFrame:NSMakeRect(x, y - 3, 100, 20)];
    self.densitySlider.minValue = 0.0;
    self.densitySlider.maxValue = 1.0;
    self.densitySlider.doubleValue = 0.2;
    self.densitySlider.allowsTickMarkValuesOnly = NO;
    self.densitySlider.numberOfTickMarks = 0;
    self.densitySlider.continuous = YES;
    self.densitySlider.target = self;
    self.densitySlider.action = @selector(sliderChanged:);
    [bar addSubview:self.densitySlider];
    x += 110;

    self.densityPct = MakeLabelSmall(@"20%", NSMakeRect(x, y, 35, 18));
    [bar addSubview:self.densityPct];
    x += 45;

    [bar addSubview:MakeLabelSmall(@"Speed", NSMakeRect(x, y, 45, 18))];
    x += 50;

    self.speedSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(x, y - 3, 120, 20)];
    self.speedSlider.minValue = 1.0;
    self.speedSlider.maxValue = 120.0;
    self.speedSlider.doubleValue = 30.0;
    self.speedSlider.allowsTickMarkValuesOnly = NO;
    self.speedSlider.numberOfTickMarks = 6;
    self.speedSlider.continuous = YES;
    self.speedSlider.target = self;
    self.speedSlider.action = @selector(sliderChanged:);
    [bar addSubview:self.speedSlider];
    x += 130;

    self.speedLabel = MakeLabelSmall(@"30 gen/s", NSMakeRect(x, y, 65, 18));
    [bar addSubview:self.speedLabel];
    x += 75;

    [bar addSubview:MakeLabelSmall(@"Preset", NSMakeRect(x, y, 45, 18))];
    x += 50;

    presetsPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(x, y - 3, 120, 24)
                                                pullsDown:NO];
    [presetsPopup addItemWithTitle:@"Glider"];
    [presetsPopup addItemWithTitle:@"Blinker"];
    [presetsPopup addItemWithTitle:@"Block"];
    [presetsPopup addItemWithTitle:@"Beacon"];
    [presetsPopup addItemWithTitle:@"Toad"];
    [presetsPopup addItemWithTitle:@"Pentadecathlon"];
    [presetsPopup addItemWithTitle:@"LWSS"];
    [presetsPopup addItemWithTitle:@"R-Pentomino"];
    [presetsPopup addItemWithTitle:@"Heptomino"];
    [presetsPopup setTarget:self];
    [presetsPopup setAction:@selector(rulePresetClicked:)];
    [bar addSubview:presetsPopup];
    x += 130;

    self.goButton = [[NSButton alloc] initWithFrame:NSMakeRect(x, y - 3, 65, 24)];
    self.goButton.title = @"Go";
    self.goButton.bezelStyle = NSBezelStyleRounded;
    self.goButton.target = self;
    self.goButton.action = @selector(goPause:);
    [bar addSubview:self.goButton];
    x += 75;

    clearButton = [[NSButton alloc] initWithFrame:NSMakeRect(x, y - 3, 65, 24)];
    clearButton.title = @"Clear";
    clearButton.bezelStyle = NSBezelStyleRounded;
    clearButton.target = self;
    clearButton.action = @selector(clear:);
    [bar addSubview:clearButton];
    x += 75;

    self.genLabel = MakeLabelSmall(@"Gen 0", NSMakeRect(x, y, 90, 18));
    [bar addSubview:self.genLabel];
    x += 100;

    self.hintLabel = MakeLabelSmall(@"L:add  R:erase", NSMakeRect(x, y, 220, 18));
    [bar addSubview:self.hintLabel];

    y = 68;
    x = 12;

    [bar addSubview:MakeLabelSmall(@"Display", NSMakeRect(x, y, 50, 18))];
    x += 55;

    self.displayPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(x, y - 3, 90, 24)
                                                     pullsDown:NO];
    [self.displayPopup addItemWithTitle:@"Age"];
    [self.displayPopup addItemWithTitle:@"Trails"];
    [self.displayPopup addItemWithTitle:@"Heatmap"];
    [self.displayPopup selectItemAtIndex:0];
    self.displayPopup.target = self;
    self.displayPopup.action = @selector(displayChanged:);
    [bar addSubview:self.displayPopup];
    x += 100;

    [bar addSubview:MakeLabelSmall(@"Tool", NSMakeRect(x, y, 35, 18))];
    x += 40;

    self.toolControl = [[NSSegmentedControl alloc] initWithFrame:NSMakeRect(x, y - 3, 90, 24)];
    self.toolControl.segmentCount = 2;
    [self.toolControl setLabel:@"Add" forSegment:0];
    [self.toolControl setLabel:@"Erase" forSegment:1];
    self.toolControl.trackingMode = NSSegmentSwitchTrackingSelectOne;
    self.toolControl.selectedSegment = 0;
    self.toolControl.target = self;
    self.toolControl.action = @selector(toolChanged:);
    [bar addSubview:self.toolControl];
    x += 100;

    [bar addSubview:MakeLabelSmall(@"Brush", NSMakeRect(x, y, 40, 18))];
    x += 45;

    self.brushSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(x, y - 3, 90, 20)];
    self.brushSlider.minValue = 0.0;
    self.brushSlider.maxValue = 10.0;
    self.brushSlider.doubleValue = 1.0;
    self.brushSlider.allowsTickMarkValuesOnly = NO;
    self.brushSlider.numberOfTickMarks = 0;
    self.brushSlider.continuous = YES;
    self.brushSlider.target = self;
    self.brushSlider.action = @selector(sliderChanged:);
    [bar addSubview:self.brushSlider];
    x += 100;

    self.brushLabel = MakeLabelSmall(@"1", NSMakeRect(x, y, 30, 18));
    [bar addSubview:self.brushLabel];
    x += 40;

    [bar addSubview:MakeLabelSmall(@"Zoom", NSMakeRect(x, y, 35, 18))];
    x += 40;

    self.zoomLabel = MakeLabelSmall(@"100%", NSMakeRect(x, y, 55, 18));
    [bar addSubview:self.zoomLabel];
    x += 65;

    fitButton = [[NSButton alloc] initWithFrame:NSMakeRect(x, y - 3, 50, 24)];
    fitButton.title = @"Fit";
    fitButton.bezelStyle = NSBezelStyleRounded;
    fitButton.target = self;
    fitButton.action = @selector(fit:);
    [bar addSubview:fitButton];
    x += 60;

    self.hoverLabel = MakeLabel(@"--", NSMakeRect(x, y, 180, 18));
    [bar addSubview:self.hoverLabel];

    weakSelf = self;
    self.birthRangeView.rangeChanged = ^{
        App *strongSelf = weakSelf;
        if (strongSelf != nil) {
            [strongSelf ruleSliderChanged:strongSelf];
        }
    };
    self.survRangeView.rangeChanged = ^{
        App *strongSelf = weakSelf;
        if (strongSelf != nil) {
            [strongSelf ruleSliderChanged:strongSelf];
        }
    };

    self.keyMonitor = [NSEvent addLocalMonitorForEventsMatchingMask:NSEventMaskKeyDown
                                                           handler:^NSEvent *(NSEvent *event) {
        App *strongSelf = weakSelf;
        if (strongSelf != nil && [strongSelf handleKey:event]) {
            return nil;
        }
        return event;
    }];

    [self updateRuleUI];
    [self updateZoomLabel];
    [self updateHintLabel];
    [self fitView];
    [self.window makeKeyAndOrderFront:nil];
    [self.window makeFirstResponder:self.mtkView];
}

- (void)drawInMTKView:(MTKView *)view {
    (void)view;
    [self tick];
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    (void)view;
    (void)size;
    [self requestGridResize];
}

- (id<MTLCommandBuffer>)cbAt:(uint32_t)plane {
    uint32_t index = plane % (uint32_t)PLANE_COUNT;
    switch (index) {
        case 0:
            return self.cb0;
        case 1:
            return self.cb1;
        default:
            return self.cb2;
    }
}

- (void)setCB:(id<MTLCommandBuffer>)cb at:(uint32_t)plane {
    uint32_t index = plane % (uint32_t)PLANE_COUNT;
    switch (index) {
        case 0:
            self.cb0 = cb;
            break;
        case 1:
            self.cb1 = cb;
            break;
        default:
            self.cb2 = cb;
            break;
    }
}

- (void)clearCBRing {
    self.cb0 = nil;
    self.cb1 = nil;
    self.cb2 = nil;
}

- (void)tick {
    id<CAMetalDrawable> drawable;
    id<MTLCommandBuffer> cb;
    uint64_t f;
    uint32_t slot;
    uint32_t curPlane;
    uint32_t renderPlane;
    uint32_t writePlane;
    BOOL willStep;
    CFTimeInterval now;
    CFTimeInterval frameDt;
    GOLRules r;
    Uniforms *u;
    CGFloat boundsW;
    CGFloat boundsH;
    double genPerSec;
    double genInterval;
    int gensToAdvance;
    id<MTLCommandBuffer> oldest;
    BOOL oldestDone;
    NSUInteger curByte;
    NSUInteger writeByte;
    NSUInteger tgW;
    NSUInteger tgH;
    NSUInteger gx;
    NSUInteger gy;
    id<MTLComputeCommandEncoder> enc;
    uint16_t *cur;
    uint16_t *write;
    MTLRenderPassDescriptor *crp;
    id<MTLRenderCommandEncoder> cenc;
    MTLViewport cvp;
    MTLRenderPassDescriptor *rp;
    id<MTLRenderCommandEncoder> rend;
    MTLViewport vp;
    BOOL needCell;
    uint16_t *cells;
    int alive;
    int maxAge;
    CFTimeInterval fpsDt;
    int fps;

    if (self.mtkView == nil || self.queue == nil || self.renderPipeline == nil ||
        self.scalePipeline == nil || self.uniformsBuf == nil || self.gridBuf == nil) {
        return;
    }

    if (self.needsGridResize) {
        [self updateGridForPixelSize:CGSizeZero];
    }

    if (!self.running && !self.dirty) {
        self.mtkView.paused = YES;
        return;
    }

    drawable = self.mtkView.currentDrawable;
    if (drawable == nil) {
        return;
    }
    cb = [self.queue commandBuffer];
    if (cb == nil) {
        return;
    }

    f = self.frameIndex;
    slot = (uint32_t)(f % (uint32_t)PLANE_COUNT);
    curPlane = slot;
    renderPlane = self.displayPlane;
    writePlane = 0;
    willStep = NO;

    now = CFAbsoluteTimeGetCurrent();

    r = self.rules;
    u = (Uniforms *)[self.uniformsBuf contents];
    u->gridW = (uint32_t)self.gridW;
    u->gridH = (uint32_t)self.gridH;
    u->pad = 0;
    u->birth = r.birth;
    u->survival = r.survival;
    u->pad2 = 0;
    u->pad3 = 0;
    boundsW = self.mtkView.bounds.size.width;
    boundsH = self.mtkView.bounds.size.height;
    if (boundsW < 1.0) {
        boundsW = 1.0;
    }
    if (boundsH < 1.0) {
        boundsH = 1.0;
    }
    if (self.cellPx < 0.01) {
        self.cellPx = CELL_PX;
    }
    u->viewScaleX = (float)(1.0 / self.cellPx);
    u->viewScaleY = (float)(1.0 / self.cellPx);
    u->viewOffsetX = (float)self.viewOffsetX;
    u->viewOffsetY = (float)self.viewOffsetY;
    u->viewWidth = (float)boundsW;
    u->viewHeight = (float)boundsH;
    u->displayMode = self.displayMode;
    u->pad4 = 0;

    if (self.running) {
        genPerSec = self.speedSlider ? self.speedSlider.doubleValue : 30.0;
        if (genPerSec < 1.0) {
            genPerSec = 1.0;
        }
        genInterval = 1.0 / genPerSec;
        frameDt = now - self.simLastTime;
        self.genAccum += (frameDt > 0 && self.simLastTime > 0) ? frameDt : genInterval;
        self.simLastTime = now;

        gensToAdvance = FloorInt(self.genAccum / genInterval);
        gensToAdvance = GOL_MIN(gensToAdvance, 5);
        self.genAccum -= (double)gensToAdvance * genInterval;

        if (gensToAdvance > 0) {
            gensToAdvance = 1;
        }

        if (gensToAdvance > 0 && self.stepPipeline != nil) {
            oldest = [self cbAt:slot];
            oldestDone = (oldest == nil) ||
                          oldest.status == MTLCommandBufferStatusCompleted ||
                          oldest.status == MTLCommandBufferStatusError;
            if (oldestDone) {
                writePlane = (curPlane + 1) % (uint32_t)PLANE_COUNT;
                curByte = (NSUInteger)curPlane * self.planeBytes;
                writeByte = (NSUInteger)writePlane * self.planeBytes;
                tgW = 16;
                tgH = 16;
                gx = ((NSUInteger)self.gridW + tgW - 1) / tgW;
                gy = ((NSUInteger)self.gridH + tgH - 1) / tgH;
                enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:self.stepPipeline];
                [enc setBuffer:self.gridBuf offset:curByte atIndex:0];
                [enc setBuffer:self.gridBuf offset:writeByte atIndex:1];
                [enc setBuffer:self.uniformsBuf offset:0 atIndex:2];
                [enc dispatchThreadgroups:MakeSize((int)gx, (int)gy, 1)
                  threadsPerThreadgroup:MakeSize((int)tgW, (int)tgH, 1)];
                [enc endEncoding];
                renderPlane = writePlane;
                self.displayPlane = writePlane;
                willStep = YES;
                [self setCB:cb at:writePlane];
            }
        } else if (gensToAdvance > 0) {
            writePlane = (curPlane + 1) % (uint32_t)PLANE_COUNT;
            cur = [self planePointer:curPlane];
            write = [self planePointer:writePlane];
            if (cur != nil && write != nil) {
                gol_step_cpu(cur, write, self.gridW, self.gridH, r);
                renderPlane = writePlane;
                self.displayPlane = writePlane;
                willStep = YES;
            }
        }
    }

    u->curOffset = (uint32_t)renderPlane * (uint32_t)self.planeCells;

    needCell = willStep || self.dirty;
    if (needCell && self.cellTex != nil && self.cellTex.width > 0 && self.cellTex.height > 0) {
        crp = [MTLRenderPassDescriptor renderPassDescriptor];
        crp.colorAttachments[0].texture = self.cellTex;
        crp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        crp.colorAttachments[0].storeAction = MTLStoreActionStore;
        cenc = [cb renderCommandEncoderWithDescriptor:crp];
        [cenc setRenderPipelineState:self.renderPipeline];
        cvp.originX = 0.0;
        cvp.originY = 0.0;
        cvp.width = (double)self.cellTex.width;
        cvp.height = (double)self.cellTex.height;
        cvp.znear = 0.0;
        cvp.zfar = 1.0;
        [cenc setViewport:cvp];
        [cenc setFragmentBuffer:self.gridBuf offset:0 atIndex:0];
        [cenc setFragmentBuffer:self.uniformsBuf offset:0 atIndex:1];
        [cenc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        [cenc endEncoding];
    }

    if (self.displayMode == DISPLAY_TRAILS && self.trailStepPipeline != nil &&
        self.trailTex != nil && self.cellTex != nil &&
        self.trailTex.width > 0 && self.trailTex.height > 0) {
        tgW = 16;
        tgH = 16;
        gx = (self.trailTex.width + tgW - 1) / tgW;
        gy = (self.trailTex.height + tgH - 1) / tgH;
        enc = [cb computeCommandEncoder];
        if (enc != nil) {
            [enc setComputePipelineState:self.trailStepPipeline];
            [enc setTexture:self.cellTex atIndex:0];
            [enc setTexture:self.trailTex atIndex:1];
            [enc setTexture:self.trailTex atIndex:2];
            [enc dispatchThreadgroups:MakeSize((int)gx, (int)gy, 1)
              threadsPerThreadgroup:MakeSize((int)tgW, (int)tgH, 1)];
            [enc endEncoding];
        }
    }

    rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = drawable.texture;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rend = [cb renderCommandEncoderWithDescriptor:rp];
    [rend setRenderPipelineState:self.scalePipeline];
    vp.originX = 0.0;
    vp.originY = 0.0;
    vp.width = (double)drawable.texture.width;
    vp.height = (double)drawable.texture.height;
    vp.znear = 0.0;
    vp.zfar = 1.0;
    [rend setViewport:vp];
    [rend setFragmentBuffer:self.uniformsBuf offset:0 atIndex:0];
    if (self.cellTex != nil) {
        [rend setFragmentTexture:self.cellTex atIndex:0];
    }
    if (self.trailTex != nil) {
        [rend setFragmentTexture:self.trailTex atIndex:1];
    }
    [rend drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [rend endEncoding];

    [cb presentDrawable:drawable];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
        if (completedBuffer.error != nil) {
            NSLog(@"command buffer error: %@", completedBuffer.error);
        }
    }];
    [cb commit];

    if (willStep) {
        self.frameIndex = f + 1u;
        self.gen = self.gen + 1u;
    }
    if (needCell && self.cellTex != nil) {
        self.dirty = NO;
    }
    self.lastCB = cb;

    if (self.genLabel != nil) {
        self.genLabel.stringValue = [NSString stringWithFormat:@"Gen %u", self.gen];
    }

    cells = [self currentCells];
    alive = 0;
    maxAge = 0;
    if (cells != nil) {
        gol_count_alive(cells, self.gridW, self.gridH, &alive, &maxAge);
    }
    if (self.popLabel != nil) {
        self.popLabel.stringValue = [NSString stringWithFormat:@"Pop: %d", alive];
    }
    if (self.maxAgeLabel != nil) {
        self.maxAgeLabel.stringValue = [NSString stringWithFormat:@"MaxAge: %d", maxAge];
    }

    self.fpsFrames = self.fpsFrames + 1u;
    if (now - self.fpsWindowStart >= 1.0) {
        fpsDt = now - self.fpsWindowStart;
        if (fpsDt > 0.0) {
            fps = LroundInt((double)self.fpsFrames / fpsDt);
            if (self.fpsLabel != nil) {
                self.fpsLabel.stringValue = [NSString stringWithFormat:@"FPS %d", fps];
            }
        }
        self.fpsWindowStart = now;
        self.fpsFrames = 0;
    }

    if (!self.running && !self.dirty) {
        self.mtkView.paused = YES;
    }
}

- (void)applicationWillTerminate:(NSNotification *)note {
    (void)note;
    if (self.keyMonitor != nil) {
        [NSEvent removeMonitor:self.keyMonitor];
        self.keyMonitor = nil;
    }
    free(self.resizeTmp);
    self.resizeTmp = nil;
}

- (BOOL)windowShouldClose:(NSWindow *)sender {
    (void)sender;
    [NSApp terminate:nil];
    return NO;
}

@end

int main(int argc, const char *argv[]) {
    App *app;
    (void)argc;
    (void)argv;
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        app = [[App alloc] init];
        [NSApp setDelegate:app];
        [NSApp activateIgnoringOtherApps:YES];
        [NSApp run];
    }
    return 0;
}
