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
static const CGFloat MIN_CELL_PX = 0.1;
static const CGFloat MAX_CELL_PX = 128.0;
static const int VIEW_W = (int)(INITIAL_GRID_W * CELL_PX);
static const int VIEW_H = (int)(INITIAL_GRID_H * CELL_PX);
static const int BAR_H = 120;
static const int RENDER_SCALE = 1;
static const int PLANE_COUNT = 3;
// Total simulation memory budget: the shared grid buffer and the cell + trail
// textures together. The maximum grid size is derived from this in setupMetal,
// so a deep zoom-out grows the grid until it would exceed the budget.
static const NSUInteger MAX_TOTAL_MEMORY = 1024u * 1024u * 1024u; // 1 GiB
// Peak per-cell footprint in bytes when the grid is at its maximum size:
//   gridBuf  PLANE_COUNT * sizeof(uint16_t)  (3 planes, shared)
//   cellTex  4 * RENDER_SCALE^2              (BGRA8Unorm)
//   trailTex 2 * RENDER_SCALE^2              (R16Unorm intensity)
static const NSUInteger BYTES_PER_CELL =
    PLANE_COUNT * (NSUInteger)sizeof(uint16_t) +
    4u * (NSUInteger)(RENDER_SCALE * RENDER_SCALE) +
    2u * (NSUInteger)(RENDER_SCALE * RENDER_SCALE);
static const int MAX_TEXTURE_SIZE = 16384;
static const uint32_t DISPLAY_AGE = 0u;
static const uint32_t DISPLAY_TRAILS = 1u;
static const uint32_t DISPLAY_HEATMAP = 2u;

// Color palettes for the Age/Heatmap display modes (index into the shader ramp).
static const uint32_t PALETTE_VIRIDIS = 0u;
static const uint32_t PALETTE_INFERNO = 1u;
static const uint32_t PALETTE_PLASMA = 2u;
static const uint32_t PALETTE_TURBO = 3u;

#define GOL_MIN(A, B) ((A) < (B) ? (A) : (B))
#define GOL_MAX(A, B) ((A) > (B) ? (A) : (B))

typedef struct {
    uint32_t gridW;
    uint32_t gridH;
    uint32_t curOffset;
    uint32_t pad;
    uint16_t birth;
    uint16_t survival;
    float viewScaleX;
    float viewScaleY;
    float viewOffsetX;
    float viewOffsetY;
    float viewWidth;
    float viewHeight;
    uint32_t displayMode;
    uint32_t palette;
    float glow;
} Uniforms;

// Per-plane statistics written by the gol_step kernel (buffer(3)): alive count
// and max age of the plane just written. One 16-byte slot per plane.
typedef struct {
    uint32_t alive;
    uint32_t maxAge;
    uint32_t pad0;
    uint32_t pad1;
} GOLStats;

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

@interface GOLRuleToggleView : NSView
@property (nonatomic, assign) uint16_t birth;
@property (nonatomic, assign) uint16_t survival;
@property (nonatomic, copy) void (^toggleChanged)(void);
- (void)setBirth:(uint16_t)b survival:(uint16_t)s;
@end

@implementation GOLRuleToggleView

@synthesize birth = _birth;
@synthesize survival = _survival;
@synthesize toggleChanged = _toggleChanged;

- (instancetype)initWithFrame:(NSRect)frame {
    if ((self = [super initWithFrame:frame])) {
        self.birth = (1u << 3);                  // B3
        self.survival = (1u << 2) | (1u << 3);  // S23
        self.toggleChanged = nil;
        self.wantsLayer = YES;
        self.layer.backgroundColor = [NSColor clearColor].CGColor;
        self.toolTip = @"Click a cell to toggle that neighbor count";
    }
    return self;
}

- (void)setBirth:(uint16_t)b survival:(uint16_t)s {
    self.birth = b & 0x1FFu;
    self.survival = s & 0x1FFu;
    [self setNeedsDisplay:YES];
}

- (NSRect)cellRect:(int)row col:(int)col {
    CGFloat labelW = 14.0;
    CGFloat cellW = 15.0;
    CGFloat gap = 1.0;
    CGFloat rowH = 16.0;
    NSRect bounds = [self bounds];
    CGFloat x = labelW + (CGFloat)col * (cellW + gap);
    CGFloat y = bounds.size.height - rowH * (CGFloat)(row + 1);
    return NSMakeRect(x, y, cellW, rowH);
}

- (void)drawRect:(NSRect)dirtyRect {
    NSColor *on;
    NSColor *off;
    NSDictionary *attrsOn;
    NSDictionary *attrsOff;
    NSDictionary *lblAttrs;
    NSString *lbl;
    NSSize lblSize;
    int row;
    int col;

    (void)dirtyRect;
    [super drawRect:dirtyRect];

    on = [NSColor colorWithSRGBRed:0.25 green:0.70 blue:0.40 alpha:1.0];
    off = [NSColor colorWithSRGBRed:0.16 green:0.19 blue:0.24 alpha:1.0];
    attrsOn = @{ NSFontAttributeName: [NSFont systemFontOfSize:9 weight:NSFontWeightSemibold],
                 NSForegroundColorAttributeName: [NSColor colorWithSRGBRed:0.05 green:0.10 blue:0.05 alpha:1.0] };
    attrsOff = @{ NSFontAttributeName: [NSFont systemFontOfSize:9],
                  NSForegroundColorAttributeName: [NSColor colorWithSRGBRed:0.60 green:0.65 blue:0.70 alpha:1.0] };
    lblAttrs = @{ NSFontAttributeName: [NSFont systemFontOfSize:10 weight:NSFontWeightBold],
                  NSForegroundColorAttributeName: [NSColor whiteColor] };

    for (row = 0; row < 2; row++) {
        NSRect cell0;
        lbl = (row == 0) ? @"B" : @"S";
        cell0 = [self cellRect:row col:0];
        lblSize = [lbl sizeWithAttributes:lblAttrs];
        [lbl drawAtPoint:NSMakePoint(2.0, cell0.origin.y + (cell0.size.height - lblSize.height) * 0.5)
          withAttributes:lblAttrs];
        for (col = 0; col < 9; col++) {
            uint16_t mask;
            BOOL set;
            NSRect r;
            NSBezierPath *p;
            NSString *num;
            NSDictionary *a;
            NSSize sz;

            mask = (row == 0) ? self.birth : self.survival;
            set = ((mask >> col) & 1u) != 0u;
            r = [self cellRect:row col:col];
            p = [NSBezierPath bezierPathWithRoundedRect:r xRadius:3.0 yRadius:3.0];
            [(set ? on : off) setFill];
            [p fill];
            num = [NSString stringWithFormat:@"%d", col];
            a = set ? attrsOn : attrsOff;
            sz = [num sizeWithAttributes:a];
            [num drawAtPoint:NSMakePoint(r.origin.x + (r.size.width - sz.width) * 0.5,
                                         r.origin.y + (r.size.height - sz.height) * 0.5)
               withAttributes:a];
        }
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
    int row;
    int col;

    [self becomeFirstResponder];
    pt = [self convertPoint:[e locationInWindow] fromView:nil];
    for (row = 0; row < 2; row++) {
        for (col = 0; col < 9; col++) {
            NSRect r = [self cellRect:row col:col];
            if (NSPointInRect(pt, NSInsetRect(r, -2.0, -2.0))) {
                if (row == 0) {
                    self.birth = (uint16_t)(self.birth ^ (1u << col));
                } else {
                    self.survival = (uint16_t)(self.survival ^ (1u << col));
                }
                [self setNeedsDisplay:YES];
                if (self.toggleChanged != nil) {
                    self.toggleChanged();
                }
                return;
            }
        }
    }
}

@end

// A small live chart of the recent population history.
static const int kSparkCapacity = 120;

@interface GOLSparklineView : NSView
@property (nonatomic, assign) int *values; // ring buffer of population samples
@property (nonatomic, assign) int head;    // index of the next slot to write
@property (nonatomic, assign) int count;   // number of valid samples (<= kSparkCapacity)
- (void)pushValue:(int)v;
- (void)resetBuffer;
@end

@implementation GOLSparklineView

@synthesize values = _values;
@synthesize head = _head;
@synthesize count = _count;

- (instancetype)initWithFrame:(NSRect)frame {
    if ((self = [super initWithFrame:frame])) {
        self.wantsLayer = YES;
        self.layer.backgroundColor = [NSColor colorWithSRGBRed:0.05 green:0.06 blue:0.09 alpha:1.0].CGColor;
        self.values = (int *)calloc((size_t)kSparkCapacity, sizeof(int));
        self.head = 0;
        self.count = 0;
    }
    return self;
}

- (void)dealloc {
    free(self.values);
}

- (void)pushValue:(int)v {
    if (self.values == NULL) {
        return;
    }
    self.values[self.head] = v;
    self.head = (self.head + 1) % kSparkCapacity;
    if (self.count < kSparkCapacity) {
        self.count++;
    }
    [self setNeedsDisplay:YES];
}

- (void)resetBuffer {
    int i;
    if (self.values == NULL) {
        return;
    }
    for (i = 0; i < kSparkCapacity; i++) {
        self.values[i] = 0;
    }
    self.head = 0;
    self.count = 0;
    [self setNeedsDisplay:YES];
}

- (void)drawRect:(NSRect)dirtyRect {
    NSRect b;
    int maxV;
    int i;
    NSBezierPath *area;
    NSBezierPath *line;
    NSColor *fillC;
    NSColor *lineC;

    (void)dirtyRect;
    [super drawRect:dirtyRect];
    b = [self bounds];
    if (b.size.width < 2.0 || b.size.height < 2.0 || self.count < 1) {
        return;
    }

    maxV = 1;
    for (i = 0; i < self.count; i++) {
        int idx = (self.head - self.count + i + kSparkCapacity) % kSparkCapacity;
        if (self.values[idx] > maxV) {
            maxV = self.values[idx];
        }
    }
    lineC = [NSColor colorWithSRGBRed:0.45 green:0.85 blue:1.00 alpha:1.0];

    if (self.count == 1) {
        // A single sample (e.g. right after loading a pattern while the sim is
        // paused): render it as a dot at the right edge so the chart never looks
        // empty. It grows into a trace once generations start advancing.
        int idx = (self.head - 1 + kSparkCapacity) % kSparkCapacity;
        CGFloat py = b.size.height - 1.0 - (b.size.height - 2.0) * ((CGFloat)self.values[idx] / (CGFloat)maxV);
        NSBezierPath *dot = [NSBezierPath bezierPathWithOvalInRect:
            NSMakeRect(b.size.width - 4.0, py - 2.0, 4.0, 4.0)];
        [lineC setFill];
        [dot fill];
        return;
    }

    area = [NSBezierPath bezierPath];
    line = [NSBezierPath bezierPath];
    [area moveToPoint:NSMakePoint(0.0, b.size.height)];
    for (i = 0; i < self.count; i++) {
        int idx = (self.head - self.count + i + kSparkCapacity) % kSparkCapacity;
        CGFloat px;
        CGFloat py;
        // Plot the most recent samples across the full width so the trace
        // scrolls left as new generations arrive.
        px = (b.size.width - 1.0) * ((CGFloat)i / (CGFloat)(self.count - 1));
        py = b.size.height - 1.0 - (b.size.height - 2.0) * ((CGFloat)self.values[idx] / (CGFloat)maxV);
        [area lineToPoint:NSMakePoint(px, py)];
        if (i == 0) {
            [line moveToPoint:NSMakePoint(px, py)];
        } else {
            [line lineToPoint:NSMakePoint(px, py)];
        }
    }
    [area lineToPoint:NSMakePoint(b.size.width, b.size.height)];
    [area closePath];

    fillC = [NSColor colorWithSRGBRed:0.20 green:0.65 blue:0.85 alpha:0.30];
    [fillC setFill];
    [area fill];
    line.lineWidth = 1.5;
    [lineC setStroke];
    [line stroke];
}

@end

// Famous rule presets. bit n set if the rule acts on exactly n neighbors (0..8).
typedef struct {
    const char *name;
    uint16_t birth;
    uint16_t survival;
    uint32_t pad; // explicit padding: keeps 8-byte alignment without implicit gaps
} GOLFamousRule;

static const GOLFamousRule kFamousRules[] = {
    { "Life",           (1u << 3), (1u << 2) | (1u << 3), 0 },
    { "HighLife",       (1u << 3) | (1u << 6), (1u << 2) | (1u << 3), 0 },
    { "Day & Night",    (1u << 3) | (1u << 6), (1u << 1) | (1u << 3) | (1u << 5) | (1u << 6), 0 },
    { "Seeds",          (1u << 2), 0, 0 },
    { "Maze",           (1u << 3), (1u << 1) | (1u << 3) | (1u << 5) | (1u << 6) | (1u << 7) | (1u << 8), 0 },
    { "Life w/o Death", (1u << 3), (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) | (1u << 4) | (1u << 5) | (1u << 6) | (1u << 7) | (1u << 8), 0 },
    { "Replicator",     (1u << 1) | (1u << 3) | (1u << 5) | (1u << 7), (1u << 1) | (1u << 3) | (1u << 5) | (1u << 7), 0 },
    { "Diamoeba",       (1u << 3) | (1u << 5) | (1u << 6) | (1u << 7) | (1u << 8), (1u << 5) | (1u << 6) | (1u << 7) | (1u << 8), 0 },
};

static const int kFamousRuleCount = (int)(sizeof(kFamousRules) / sizeof(kFamousRules[0]));

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
@property (nonatomic, strong) id<MTLBuffer> statsBuf;
@property (nonatomic, strong) id<MTLCommandBuffer> cb0;
@property (nonatomic, strong) id<MTLCommandBuffer> cb1;
@property (nonatomic, strong) id<MTLCommandBuffer> cb2;
@property (nonatomic, strong) id<MTLCommandBuffer> lastCB;
@property (nonatomic, assign) uint64_t frameIndex;
@property (nonatomic, assign) uint32_t displayPlane;
@property (nonatomic, assign) uint32_t completedPlane;
@property (nonatomic, assign) int gridW;
@property (nonatomic, assign) int gridH;
@property (nonatomic, assign) int maxGridW;
@property (nonatomic, assign) int maxGridH;
@property (nonatomic, assign) NSUInteger planeCells;
@property (nonatomic, assign) NSUInteger planeBytes;
@property (nonatomic, assign) BOOL running;
@property (nonatomic, assign) BOOL dirty;
@property (nonatomic, assign) BOOL needsGridResize;
@property (nonatomic, assign) uint32_t gen;
@property (nonatomic, assign) uint32_t popAlive;
@property (nonatomic, assign) uint32_t popMaxAge;
@property (nonatomic, assign) uint32_t lastStatsGen;
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
@property (nonatomic, strong) GOLSparklineView *popSpark;
@property (nonatomic, strong) NSSlider *speedSlider;
@property (nonatomic, strong) NSTextField *speedLabel;
@property (nonatomic, strong) NSPopUpButton *rulePopup;
@property (nonatomic, strong) GOLRuleToggleView *ruleToggleView;
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
@property (nonatomic, assign) uint32_t palette;
@property (nonatomic, assign) BOOL glowOn;
@property (nonatomic, strong) NSPopUpButton *displayPopup;
@property (nonatomic, strong) NSPopUpButton *palettePopup;
@property (nonatomic, strong) NSButton *glowButton;
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
- (void)randomize:(id)sender;
- (void)goPause:(id)sender;
- (void)clear:(id)sender;
- (void)clear;
- (void)fit:(id)sender;
- (void)fitView;
- (void)updateZoomLabel;
- (void)updateHintLabel;
- (void)toolChanged:(id)sender;
- (void)displayChanged:(id)sender;
- (void)paletteChanged:(id)sender;
- (void)glowToggle:(id)sender;
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
- (void)rulePopupChanged:(id)sender;
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
@synthesize statsBuf = _statsBuf;
@synthesize cb0 = _cb0;
@synthesize cb1 = _cb1;
@synthesize cb2 = _cb2;
@synthesize lastCB = _lastCB;
@synthesize frameIndex = _frameIndex;
@synthesize displayPlane = _displayPlane;
@synthesize completedPlane = _completedPlane;
@synthesize gridW = _gridW;
@synthesize gridH = _gridH;
@synthesize maxGridW = _maxGridW;
@synthesize maxGridH = _maxGridH;
@synthesize planeCells = _planeCells;
@synthesize planeBytes = _planeBytes;
@synthesize running = _running;
@synthesize dirty = _dirty;
@synthesize needsGridResize = _needsGridResize;
@synthesize gen = _gen;
@synthesize popAlive = _popAlive;
@synthesize popMaxAge = _popMaxAge;
@synthesize lastStatsGen = _lastStatsGen;
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
@synthesize popSpark = _popSpark;
@synthesize speedSlider = _speedSlider;
@synthesize speedLabel = _speedLabel;
@synthesize rulePopup = _rulePopup;
@synthesize ruleToggleView = _ruleToggleView;
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
@synthesize palette = _palette;
@synthesize glowOn = _glowOn;
@synthesize displayPopup = _displayPopup;
@synthesize palettePopup = _palettePopup;
@synthesize glowButton = _glowButton;
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
    NSString *palette = nil;
    double density = 0.0;
    double zoom = 0.0;
    int run = 0;
    int densitySet = 0;
    int zoomSet = 0;
    int runSet = 0;
    const char *env;
    double val;
    NSUInteger i;
    args = [[NSProcessInfo processInfo] arguments];
    for (i = 1; i < args.count; i++) {
        NSString *a = args[i];
        if (i + 1 < args.count) {
            NSString *n = args[i + 1];
            if ([a caseInsensitiveCompare:@"--mode"] == NSOrderedSame) { mode = n; i++; }
            else if ([a caseInsensitiveCompare:@"--preset"] == NSOrderedSame) { preset = n; i++; }
            else if ([a caseInsensitiveCompare:@"--palette"] == NSOrderedSame) { palette = n; i++; }
            else if ([a caseInsensitiveCompare:@"--density"] == NSOrderedSame) { density = atof(n.UTF8String); densitySet = 1; i++; }
            else if ([a caseInsensitiveCompare:@"--zoom"] == NSOrderedSame) { zoom = atof(n.UTF8String); zoomSet = 1; i++; }
            else if ([a caseInsensitiveCompare:@"--run"] == NSOrderedSame) { run = atoi(n.UTF8String); runSet = 1; i++; }
        }
    }
    if (mode == nil && (env = getenv("GOL_MODE")) != NULL) mode = [NSString stringWithUTF8String:env];
    if (preset == nil && (env = getenv("GOL_PRESET")) != NULL) preset = [NSString stringWithUTF8String:env];
    if (palette == nil && (env = getenv("GOL_PALETTE")) != NULL) palette = [NSString stringWithUTF8String:env];
    if (!densitySet && (env = getenv("GOL_DENSITY")) != NULL) { density = atof(env); densitySet = 1; }
    if (!zoomSet && (env = getenv("GOL_ZOOM")) != NULL) { zoom = atof(env); zoomSet = 1; }
    if (!runSet && (env = getenv("GOL_RUN")) != NULL) { run = atoi(env); runSet = 1; }
    if (densitySet && self.densitySlider != nil) {
        val = density;
        if (val < 0.0) val = 0.0;
        if (val > 1.0) val = 1.0;
        self.densitySlider.doubleValue = val;
    }
    if (zoomSet && zoom > 0.0) { self.cellPx = zoom; }
    if (runSet) { self.running = run != 0; }
    if (mode != nil && mode.length > 0) {
        NSString *m = [mode lowercaseString];
        if ([m isEqualToString:@"age"]) self.displayMode = DISPLAY_AGE;
        else if ([m isEqualToString:@"trails"]) self.displayMode = DISPLAY_TRAILS;
        else if ([m isEqualToString:@"heatmap"]) self.displayMode = DISPLAY_HEATMAP;
    }
    if (palette != nil && palette.length > 0) {
        NSString *p = [palette lowercaseString];
        if ([p isEqualToString:@"inferno"]) self.palette = PALETTE_INFERNO;
        else if ([p isEqualToString:@"plasma"]) self.palette = PALETTE_PLASMA;
        else if ([p isEqualToString:@"turbo"]) self.palette = PALETTE_TURBO;
        else self.palette = PALETTE_VIRIDIS;
    }
    if (preset != nil && preset.length > 0) {
        [self applyPreset:[preset UTF8String]];
    } else {
        [self randomize];
    }
}

- (void)applicationDidFinishLaunching:(NSNotification *)note {
    CFTimeInterval now;
    NSMenu *mainMenu;
    NSMenuItem *appItem;
    NSMenu *appMenu;
    (void)note;
    now = CFAbsoluteTimeGetCurrent();
    self.frameIndex = 0;
    self.displayPlane = 0;
    self.completedPlane = 0;
    self.gridW = INITIAL_GRID_W;
    self.gridH = INITIAL_GRID_H;
    self.gen = 0;
    self.lastStatsGen = 0;
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
    self.palette = PALETTE_VIRIDIS;
    self.glowOn = YES;
    if (![self setupMetal]) {
        NSLog(@"Metal setup failed");
        [NSApp terminate:nil];
        return;
    }
    [self setupUI];
    [self applyLaunchConfig];

    mainMenu = [NSMenu new];
    appItem = [mainMenu addItemWithTitle:@"" action:nil keyEquivalent:@""];
    appMenu = [NSMenu new];
    [appMenu addItemWithTitle:@"Quit GOL" action:@selector(terminate:) keyEquivalent:@"q"];
    appItem.submenu = appMenu;
    NSApp.mainMenu = mainMenu;
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
    NSUInteger budgetCells;
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
    // Cap the maximum grid so the peak footprint (grid buffer + CPU scratch +
    // cell/trail textures) stays under MAX_TOTAL_MEMORY. This is what actually
    // bounds how far out you can zoom.
    budgetCells = MAX_TOTAL_MEMORY / BYTES_PER_CELL;
    if (budgetCells < 64) budgetCells = 64;
    need = (NSUInteger)self.maxGridW * (NSUInteger)self.maxGridH;
    if (need > budgetCells) {
        double shrink = sqrt((double)budgetCells / (double)need);
        self.maxGridW = GOL_MAX(8, FloorInt((double)self.maxGridW * shrink));
        self.maxGridH = GOL_MAX(8, FloorInt((double)self.maxGridH * shrink));
        need = (NSUInteger)self.maxGridW * (NSUInteger)self.maxGridH;
    }
    self.planeCells = ((need + 7) / 8) * 8;
    self.planeBytes = self.planeCells * sizeof(uint16_t);

    self.gridBuf = [self.device newBufferWithLength:(NSUInteger)PLANE_COUNT * self.planeBytes
                                            options:MTLResourceStorageModeShared];
    if (self.gridBuf == nil) {
        return NO;
    }
    memset([self.gridBuf contents], 0, (size_t)PLANE_COUNT * self.planeBytes);
    NSLog(@"GOL: grid up to %lux%lu (%.1fM cells), peak ~%.0f MB of 1024 MB budget",
          (unsigned long)self.maxGridW, (unsigned long)self.maxGridH,
          (double)self.planeCells / 1e6,
          (double)(self.planeCells * BYTES_PER_CELL) / 1048576.0);

    self.uniformsBuf = [self.device newBufferWithLength:sizeof(Uniforms) options:MTLResourceStorageModeShared];
    if (self.uniformsBuf == nil) {
        return NO;
    }
    self.statsBuf = [self.device newBufferWithLength:(NSUInteger)PLANE_COUNT * sizeof(GOLStats)
                                            options:MTLResourceStorageModeShared];
    if (self.statsBuf == nil) {
        return NO;
    }
    memset([self.statsBuf contents], 0, (size_t)PLANE_COUNT * sizeof(GOLStats));

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
    uint16_t b;
    uint16_t s;
    int i;
    int match;
    int sel;
    NSMutableString *label;

    r = self.rules;
    b = r.birth & 0x1FFu;
    s = r.survival & 0x1FFu;

    if (self.ruleToggleView != nil) {
        [self.ruleToggleView setBirth:b survival:s];
    }

    // Select the matching famous rule in the dropdown, else "Custom".
    match = -1;
    for (i = 0; i < kFamousRuleCount; i++) {
        if (kFamousRules[i].birth == b && kFamousRules[i].survival == s) {
            match = i;
            break;
        }
    }
    sel = (match >= 0) ? match : kFamousRuleCount;
    if (self.rulePopup != nil && (int)[self.rulePopup indexOfSelectedItem] != sel) {
        [self.rulePopup selectItemAtIndex:sel];
    }

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
    int alive;
    int maxAge;

    [self waitAll];
    self.frameIndex = 0;
    self.displayPlane = 0;
    self.completedPlane = 0;
    self.gen = 0;
    self.lastStatsGen = 0;
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
    if (self.popSpark != nil) {
        [self.popSpark resetBuffer];
    }
    alive = 0;
    maxAge = 0;
    gol_count_alive(cells, self.gridW, self.gridH, &alive, &maxAge);
    [self setPopulation:(uint32_t)alive maxAge:(uint32_t)maxAge forGeneration:self.gen];
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
    } else if ([preset isEqualToString:@"Pulsar"]) {
        [self applyPreset:"pulsar"];
    } else if ([preset isEqualToString:@"Acorn"]) {
        [self applyPreset:"acorn"];
    }
}

- (void)rulePopupChanged:(id)sender {
    GOLRules r;
    int idx;

    (void)sender;
    if (self.rulePopup == nil) {
        return;
    }
    idx = (int)[self.rulePopup indexOfSelectedItem];
    if (idx >= 0 && idx < kFamousRuleCount) {
        r = self.rules;
        r.birth = kFamousRules[idx].birth;
        r.survival = kFamousRules[idx].survival;
        self.rules = r;
    }
    [self updateRuleUI];
}

- (void)randomize {
    double density;
    uint16_t *cells;
    uint16_t *base;
    uint32_t p;
    int pct;
    int alive;
    int maxAge;

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
    self.completedPlane = 0;
    self.gen = 0;
    self.lastStatsGen = 0;
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
    if (self.popSpark != nil) {
        [self.popSpark resetBuffer];
    }
    alive = 0;
    maxAge = 0;
    gol_count_alive(cells, self.gridW, self.gridH, &alive, &maxAge);
    [self setPopulation:(uint32_t)alive maxAge:(uint32_t)maxAge forGeneration:self.gen];
    [self markDirty];
}

- (void)randomize:(id)sender {
    (void)sender;
    [self randomize];
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
    int alive;
    int maxAge;

    [self waitAll];
    self.frameIndex = 0;
    self.displayPlane = 0;
    self.completedPlane = 0;
    self.gen = 0;
    self.lastStatsGen = 0;
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
    if (self.popSpark != nil) {
        [self.popSpark resetBuffer];
    }
    alive = 0;
    maxAge = 0;
    gol_count_alive(cells, self.gridW, self.gridH, &alive, &maxAge);
    [self setPopulation:(uint32_t)alive maxAge:(uint32_t)maxAge forGeneration:self.gen];
    [self markDirty];
}

// Applies a population/max-age result. Results arrive async from the GPU step
// readback, so anything older than the last applied generation is dropped.
- (void)setPopulation:(uint32_t)alive maxAge:(uint32_t)maxAge forGeneration:(uint32_t)gen {
    if (gen < self.lastStatsGen) {
        return;
    }
    self.lastStatsGen = gen;
    self.popAlive = alive;
    self.popMaxAge = maxAge;
    if (self.popLabel != nil) {
        self.popLabel.stringValue = [NSString stringWithFormat:@"Pop: %u", alive];
    }
    if (self.maxAgeLabel != nil) {
        self.maxAgeLabel.stringValue = [NSString stringWithFormat:@"MaxAge: %u", maxAge];
    }
    if (self.popSpark != nil) {
        [self.popSpark pushValue:(int)alive];
    }
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
    uint16_t *cells;
    int y;
    int x;
    int dMinCol;
    int dMaxCol;
    int dMinRow;
    int dMaxRow;
    int rMinCol;
    int rMaxCol;
    int rMinRow;
    int rMaxRow;
    int hasPattern;
    int compact;
    double pw;
    double ph;
    double bw;
    double bh;
    double margin;
    double fitW;
    double fitH;
    double scaleW;
    double scaleH;
    double s;
    double centerCol;
    double centerRow;

    if (self.mtkView == nil || self.gridW < 1 || self.gridH < 1) {
        return;
    }
    bounds = [self.mtkView bounds];
    if (bounds.size.width < 1.0 || bounds.size.height < 1.0) {
        return;
    }
    [self waitAll];

    // Bounding box of the live cells in the currently displayed plane.
    dMinCol = self.gridW;
    dMaxCol = -1;
    dMinRow = self.gridH;
    dMaxRow = -1;
    cells = [self currentCells];
    if (cells != nil) {
        for (y = 0; y < self.gridH; y++) {
            for (x = 0; x < self.gridW; x++) {
                if (GolAlive(cells[(size_t)y * (size_t)self.gridW + (size_t)x])) {
                    if (x < dMinCol) dMinCol = x;
                    if (x > dMaxCol) dMaxCol = x;
                    if (y < dMinRow) dMinRow = y;
                    if (y > dMaxRow) dMaxRow = y;
                }
            }
        }
    }

    hasPattern = (dMaxCol >= dMinCol && dMaxRow >= dMinRow);
    pw = hasPattern ? (double)(dMaxCol - dMinCol + 1) : 0.0;
    ph = hasPattern ? (double)(dMaxRow - dMinRow + 1) : 0.0;
    // Only treat it as a "pattern" if it is clearly smaller than the whole
    // grid; otherwise (a full soup) we fit the entire grid as before.
    compact = hasPattern && pw <= (double)self.gridW * 0.9 && ph <= (double)self.gridH * 0.9;

    if (compact) {
        rMinCol = dMinCol;
        rMaxCol = dMaxCol;
        rMinRow = dMinRow;
        rMaxRow = dMaxRow;
        bw = pw;
        bh = ph;
    } else {
        rMinCol = 0;
        rMaxCol = self.gridW - 1;
        rMinRow = 0;
        rMaxRow = self.gridH - 1;
        bw = (double)self.gridW;
        bh = (double)self.gridH;
    }

    if (compact) {
        // Add breathing room so the pattern is not edge-to-edge.
        margin = fmax(4.0, fmin(bw, bh) * 0.2);
        fitW = bw + 2.0 * margin;
        fitH = bh + 2.0 * margin;
    } else {
        fitW = bw;
        fitH = bh;
    }

    scaleW = bounds.size.width / fitW;
    scaleH = bounds.size.height / fitH;
    s = GOL_MIN(scaleW, scaleH);
    s = ClampDouble(s, (double)MIN_CELL_PX, (double)MAX_CELL_PX);
    self.cellPx = (CGFloat)s;

    centerCol = (double)(rMinCol + rMaxCol) * 0.5;
    centerRow = (double)(rMinRow + rMaxRow) * 0.5;
    self.viewOffsetX = (CGFloat)(bounds.size.width * 0.5 - centerCol * s);
    self.viewOffsetY = (CGFloat)(bounds.size.height * 0.5 - centerRow * s);
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

- (void)paletteChanged:(id)sender {
    int index;
    (void)sender;
    if (self.palettePopup == nil) {
        return;
    }
    index = (int)self.palettePopup.indexOfSelectedItem;
    if (index < 0 || index > (int)PALETTE_TURBO) {
        index = 0;
    }
    self.palette = (uint32_t)index;
    [self markDirty];
}

- (void)glowToggle:(id)sender {
    (void)sender;
    self.glowOn = !self.glowOn;
    if (self.glowButton != nil) {
        self.glowButton.state = self.glowOn ? NSControlStateValueOn : NSControlStateValueOff;
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
    cells = [self planePointer:self.completedPlane];
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
    NSString *ch;

    flags = event.modifierFlags;
    if ((flags & (NSEventModifierFlagCommand | NSEventModifierFlagControl)) != 0) {
        return NO;
    }
    if ([self.window firstResponder] != self.mtkView) {
        return NO;
    }
    ch = [event.charactersIgnoringModifiers lowercaseString];
    if (ch == nil || ch.length == 0) {
        return NO;
    }
    if ([ch isEqualToString:@" "]) {
        [self goPause:nil];
        return YES;
    }
    if ([ch isEqualToString:@"r"]) {
        [self clear];
        return YES;
    }
    if ([ch isEqualToString:@"z"]) {
        [self randomize];
        return YES;
    }
    if ([ch isEqualToString:@"g"]) {
        [self glowToggle:nil];
        return YES;
    }
    if ([ch isEqualToString:@"f"] || [ch isEqualToString:@"0"]) {
        [self fitView];
        return YES;
    }
    return NO;
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
    int alive;
    int maxAge;

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
    alive = 0;
    maxAge = 0;
    gol_count_alive(cells, self.gridW, self.gridH, &alive, &maxAge);
    [self setPopulation:(uint32_t)alive maxAge:(uint32_t)maxAge forGeneration:self.gen];
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
    uint16_t *scratchPlane;
    uint32_t p;
    uint16_t *cells;
    int alive;
    int maxAge;

    (void)pixelSize;
    self.needsGridResize = NO;
    if (self.gridBuf == nil || self.mtkView == nil ||
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
    scratchPlane = base + ((size_t)(self.displayPlane + 1) % (size_t)PLANE_COUNT) *
                    (size_t)self.planeCells;

    gol_copy_region(oldPlane, oldW, oldH, scratchPlane, newW, newH,
                    self.planeCells, srcX, srcY);
    self.gridW = newW;
    self.gridH = newH;
    self.viewOffsetX = 0.0;
    self.viewOffsetY = 0.0;

    if ((self.displayPlane + 1) % (uint32_t)PLANE_COUNT != 0) {
        memcpy(base, scratchPlane, (size_t)self.planeCells * sizeof(uint16_t));
    }
    for (p = 1; p < (uint32_t)PLANE_COUNT; p++) {
        memset(base + (size_t)p * (size_t)self.planeCells, 0, self.planeBytes);
    }

    self.frameIndex = 0;
    self.displayPlane = 0;
    self.completedPlane = 0;
    [self rebuildCellTexture];
    [self clearTrailNow];
    cells = [self currentCells];
    if (cells != nil) {
        alive = 0;
        maxAge = 0;
        gol_count_alive(cells, self.gridW, self.gridH, &alive, &maxAge);
        [self setPopulation:(uint32_t)alive maxAge:(uint32_t)maxAge forGeneration:self.gen];
    }
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

    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Unorm
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
    NSButton *randomButton;
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

    [bar addSubview:MakeLabelSmall(@"Rule", NSMakeRect(x, y, 35, 18))];
    x += 40;
    self.rulePopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(x, y - 2, 115, 24) pullsDown:NO];
    {
        int i;
        for (i = 0; i < kFamousRuleCount; i++) {
            NSString *title = [NSString stringWithUTF8String:kFamousRules[i].name];
            if (title != nil) {
                [self.rulePopup addItemWithTitle:title];
            }
        }
        [self.rulePopup addItemWithTitle:@"Custom"];
        [self.rulePopup selectItemAtIndex:0]; // Life
        self.rulePopup.target = self;
        self.rulePopup.action = @selector(rulePopupChanged:);
    }
    [bar addSubview:self.rulePopup];
    x += 120;

    self.ruleToggleView = [[GOLRuleToggleView alloc] initWithFrame:NSMakeRect(x, y - 6, 165, 34)];
    [bar addSubview:self.ruleToggleView];
    x += 170;

    self.ruleLabel = MakeLabel(@"B3/S23", NSMakeRect(x, y - 2, 120, 22));
    [bar addSubview:self.ruleLabel];
    x += 110;

    self.popLabel = MakeLabelSmall(@"Pop: 0", NSMakeRect(x, y - 2, 90, 22));
    [bar addSubview:self.popLabel];
    x += 100;

    self.maxAgeLabel = MakeLabelSmall(@"Age: 0", NSMakeRect(x, y - 2, 70, 22));
    [bar addSubview:self.maxAgeLabel];
    x += 80;

    [bar addSubview:MakeLabelSmall(@"Trend", NSMakeRect(x, y, 40, 18))];
    x += 45;

    self.popSpark = [[GOLSparklineView alloc] initWithFrame:NSMakeRect(x, y - 3, 120, 24)];
    self.popSpark.toolTip = @"Population over recent generations";
    [bar addSubview:self.popSpark];
    x += 130;

    self.glowButton = [[NSButton alloc] initWithFrame:NSMakeRect(x, y - 3, 54, 24)];
    self.glowButton.title = @"Glow";
    [self.glowButton setButtonType:NSButtonTypeSwitch];
    self.glowButton.state = self.glowOn ? NSControlStateValueOn : NSControlStateValueOff;
    self.glowButton.target = self;
    self.glowButton.action = @selector(glowToggle:);
    [bar addSubview:self.glowButton];

    y = 38;
    x = 12;

    [bar addSubview:MakeLabelSmall(@"Density", NSMakeRect(x, y, 50, 18))];
    x += 55;

    self.densitySlider = [[NSSlider alloc] initWithFrame:NSMakeRect(x, y - 3, 90, 20)];
    self.densitySlider.minValue = 0.0;
    self.densitySlider.maxValue = 1.0;
    self.densitySlider.doubleValue = 0.2;
    self.densitySlider.allowsTickMarkValuesOnly = NO;
    self.densitySlider.numberOfTickMarks = 0;
    self.densitySlider.continuous = YES;
    self.densitySlider.target = self;
    self.densitySlider.action = @selector(sliderChanged:);
    [bar addSubview:self.densitySlider];
    x += 95;

    self.densityPct = MakeLabelSmall(@"20%", NSMakeRect(x, y, 34, 18));
    [bar addSubview:self.densityPct];
    x += 39;

    [bar addSubview:MakeLabelSmall(@"Speed", NSMakeRect(x, y, 42, 18))];
    x += 47;

    self.speedSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(x, y - 3, 105, 20)];
    self.speedSlider.minValue = 1.0;
    self.speedSlider.maxValue = 120.0;
    self.speedSlider.doubleValue = 30.0;
    self.speedSlider.allowsTickMarkValuesOnly = NO;
    self.speedSlider.numberOfTickMarks = 6;
    self.speedSlider.continuous = YES;
    self.speedSlider.target = self;
    self.speedSlider.action = @selector(sliderChanged:);
    [bar addSubview:self.speedSlider];
    x += 110;

    self.speedLabel = MakeLabelSmall(@"30 gen/s", NSMakeRect(x, y, 60, 18));
    [bar addSubview:self.speedLabel];
    x += 65;

    [bar addSubview:MakeLabelSmall(@"Preset", NSMakeRect(x, y, 40, 18))];
    x += 45;

    presetsPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(x, y - 3, 110, 24)
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
    [presetsPopup addItemWithTitle:@"Pulsar"];
    [presetsPopup addItemWithTitle:@"Acorn"];
    [presetsPopup setTarget:self];
    [presetsPopup setAction:@selector(rulePresetClicked:)];
    [bar addSubview:presetsPopup];
    x += 115;

    self.goButton = [[NSButton alloc] initWithFrame:NSMakeRect(x, y - 3, 52, 24)];
    self.goButton.title = @"Go";
    self.goButton.bezelStyle = NSBezelStyleRounded;
    self.goButton.target = self;
    self.goButton.action = @selector(goPause:);
    [bar addSubview:self.goButton];
    x += 57;

    randomButton = [[NSButton alloc] initWithFrame:NSMakeRect(x, y - 3, 70, 24)];
    randomButton.title = @"Random";
    randomButton.bezelStyle = NSBezelStyleRounded;
    randomButton.target = self;
    randomButton.action = @selector(randomize:);
    [bar addSubview:randomButton];
    x += 75;

    clearButton = [[NSButton alloc] initWithFrame:NSMakeRect(x, y - 3, 58, 24)];
    clearButton.title = @"Clear";
    clearButton.bezelStyle = NSBezelStyleRounded;
    clearButton.target = self;
    clearButton.action = @selector(clear:);
    [bar addSubview:clearButton];
    x += 63;

    self.genLabel = MakeLabelSmall(@"Gen 0", NSMakeRect(x, y, 60, 18));
    [bar addSubview:self.genLabel];
    x += 70;

    self.hintLabel = MakeLabelSmall(@"L:add  R:erase", NSMakeRect(x, y, 105, 18));
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

    [bar addSubview:MakeLabelSmall(@"Palette", NSMakeRect(x, y, 50, 18))];
    x += 55;

    self.palettePopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(x, y - 3, 90, 24)
                                                      pullsDown:NO];
    [self.palettePopup addItemWithTitle:@"Viridis"];
    [self.palettePopup addItemWithTitle:@"Inferno"];
    [self.palettePopup addItemWithTitle:@"Plasma"];
    [self.palettePopup addItemWithTitle:@"Turbo"];
    [self.palettePopup selectItemAtIndex:(int)self.palette];
    self.palettePopup.target = self;
    self.palettePopup.action = @selector(paletteChanged:);
    [bar addSubview:self.palettePopup];
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

    self.hoverLabel = MakeLabel(@"--", NSMakeRect(x, y, 145, 18));
    [bar addSubview:self.hoverLabel];

    weakSelf = self;
    self.ruleToggleView.toggleChanged = ^{
        App *strongSelf = weakSelf;
        if (strongSelf != nil) {
            GOLRules r = strongSelf.rules;
            r.birth = strongSelf.ruleToggleView.birth;
            r.survival = strongSelf.ruleToggleView.survival;
            strongSelf.rules = r;
            [strongSelf updateRuleUI];
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
    GOLStats *s;
    uint32_t stepGen;
    __weak App *weakSelf;
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
    u->palette = self.palette;
    u->glow = self.glowOn ? 1.0f : 0.0f;

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
                // The step also writes the stats slot of the write plane, so
                // that plane's last command buffer must be done too.
                writePlane = (curPlane + 1) % (uint32_t)PLANE_COUNT;
                oldest = [self cbAt:writePlane];
                oldestDone = (oldest == nil) ||
                              oldest.status == MTLCommandBufferStatusCompleted ||
                              oldest.status == MTLCommandBufferStatusError;
            }
            if (oldestDone) {
                s = (GOLStats *)[self.statsBuf contents];
                memset(&s[writePlane], 0, sizeof(GOLStats));
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
                [enc setBuffer:self.statsBuf offset:writePlane * sizeof(GOLStats) atIndex:3];
                [enc dispatchThreadgroups:MakeSize((int)gx, (int)gy, 1)
                  threadsPerThreadgroup:MakeSize((int)tgW, (int)tgH, 1)];
                [enc endEncoding];
                renderPlane = writePlane;
                self.displayPlane = writePlane;
                self.completedPlane = curPlane;
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
                self.completedPlane = writePlane;
                willStep = YES;
                cells = [self planePointer:writePlane];
                alive = 0;
                maxAge = 0;
                if (cells != nil) {
                    gol_count_alive(cells, self.gridW, self.gridH, &alive, &maxAge);
                }
                [self setPopulation:(uint32_t)alive maxAge:(uint32_t)maxAge
                   forGeneration:self.gen + 1u];
            }
        }
    }

    // Skip idle frames: no step this frame and nothing painted since the last
    // encode, so there is nothing new to draw. Return before touching the
    // drawable or any render pass. The accumulator above keeps ticking, so the
    // next step still fires on time; lastCB and the FPS counter below are only
    // reached when a frame is actually presented.
    if (!willStep && !self.dirty) {
        return;
    }

    drawable = self.mtkView.currentDrawable;
    if (drawable == nil) {
        return;
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

    if (willStep && self.displayMode == DISPLAY_TRAILS && self.trailStepPipeline != nil &&
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
    if (willStep && self.stepPipeline != nil) {
        // The step kernel wrote the stats slot of writePlane; read it back once
        // the buffer completes and apply it on the main queue.
        stepGen = self.gen + 1u;
        weakSelf = self;
        [cb addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            App *strongSelf;
            GOLStats *stats;
            uint32_t a;
            uint32_t m;
            if (completedBuffer.error != nil) {
                NSLog(@"command buffer error: %@", completedBuffer.error);
                return;
            }
            strongSelf = weakSelf;
            if (strongSelf == nil) {
                return;
            }
            stats = (GOLStats *)[strongSelf.statsBuf contents];
            a = stats[writePlane].alive;
            m = stats[writePlane].maxAge;
            dispatch_async(dispatch_get_main_queue(), ^{
                [strongSelf setPopulation:a maxAge:m forGeneration:stepGen];
            });
        }];
    } else {
        [cb addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            if (completedBuffer.error != nil) {
                NSLog(@"command buffer error: %@", completedBuffer.error);
            }
        }];
    }
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
