#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/QuartzCore.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
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
static const CGFloat SIDE_W = 190.0;
static const int RENDER_SCALE = 1;
static const int PLANE_COUNT = 3;
// Total simulation memory budget: the shared grid buffer and the cell + trail
// textures together. The maximum grid size is derived from this in setupMetal,
// so a deep zoom-out grows the grid until it would exceed the budget.
static const NSUInteger MAX_TOTAL_MEMORY = 1024u * 1024u * 1024u; // 1 GiB
// Peak per-cell footprint in bytes when the grid is at its maximum size:
//   gridBuf  PLANE_COUNT * sizeof(uint16_t)  (3 planes, shared)
//   cellTex  4 * RENDER_SCALE^2 * 4/3        (BGRA8Unorm + full mip chain,
//                                             rounded up to 6 bytes/cell)
//   trailTex 2 * RENDER_SCALE^2              (R16Unorm intensity)
static const NSUInteger BYTES_PER_CELL =
    PLANE_COUNT * (NSUInteger)sizeof(uint16_t) +
    6u * (NSUInteger)(RENDER_SCALE * RENDER_SCALE) +
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
    float cursorX;
    float cursorY;
    float brushR;
    float pad2;
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
    l.textColor = [NSColor labelColor];
    return l;
}

static NSTextField *MakeLabelSmall(NSString *s, NSRect f) {
    NSTextField *l = [NSTextField labelWithString:s];
    l.frame = f;
    l.font = [NSFont systemFontOfSize:11];
    l.textColor = [NSColor secondaryLabelColor];
    return l;
}

static NSTextField *MakeValueLabel(NSString *s, NSRect f) {
    NSTextField *l = [NSTextField labelWithString:s];
    l.frame = f;
    l.font = [NSFont monospacedDigitSystemFontOfSize:11 weight:NSFontWeightRegular];
    l.textColor = [NSColor secondaryLabelColor];
    return l;
}

// Adds a menu item. The modifier mask only applies when key is non-empty.
static NSMenuItem *MakeMenuItem(NSMenu *menu, NSString *title, SEL action,
                                NSString *key, NSUInteger mask) {
    NSMenuItem *item = [menu addItemWithTitle:title action:action keyEquivalent:key];
    if (key.length > 0) {
        item.keyEquivalentModifierMask = (NSEventModifierFlags)mask;
    }
    return item;
}

// Marks v for Auto Layout and appends it as an arranged subview of stack.
static void StackAdd(NSStackView *stack, NSView *v) {
    v.translatesAutoresizingMaskIntoConstraints = NO;
    [stack addArrangedSubview:v];
}

// A horizontal, center-Y-aligned stack of the given views (spacing 4).
// Pass nil for b/c to build a shorter group.
static NSStackView *HGroup(NSView *a, NSView *b, NSView *c) {
    NSStackView *s = [[NSStackView alloc] init];
    s.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    s.spacing = 4.0;
    s.alignment = NSLayoutAttributeCenterY;
    if (a != nil) {
        StackAdd(s, a);
    }
    if (b != nil) {
        StackAdd(s, b);
    }
    if (c != nil) {
        StackAdd(s, c);
    }
    return s;
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
    off = [NSColor quaternaryLabelColor];
    attrsOn = @{ NSFontAttributeName: [NSFont systemFontOfSize:9 weight:NSFontWeightSemibold],
                  NSForegroundColorAttributeName: [NSColor labelColor] };
    attrsOff = @{ NSFontAttributeName: [NSFont systemFontOfSize:9],
                   NSForegroundColorAttributeName: [NSColor secondaryLabelColor] };
    lblAttrs = @{ NSFontAttributeName: [NSFont systemFontOfSize:10 weight:NSFontWeightBold],
                   NSForegroundColorAttributeName: [NSColor labelColor] };

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
        self.layer.backgroundColor = [NSColor colorWithWhite:0 alpha:0.25].CGColor;
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
    lineC = [NSColor controlAccentColor];

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

    fillC = [[NSColor controlAccentColor] colorWithAlphaComponent:0.3];
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

// One queued paint sample (a single mouse event). Records are applied together
// at the top of the next tick so a fast drag does not stall the pipeline per event.
typedef struct {
    int col;
    int row;
    int radius;
    int add;
} PaintOp;

enum { kPaintQueueCap = 256 };

// App is a singleton, so the queue lives at file scope rather than as ivars.
static PaintOp paintQueue[kPaintQueueCap];
static int paintCount;

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
@property (nonatomic, assign) BOOL stepOnce;
@property (nonatomic, assign) BOOL dirty;
@property (nonatomic, assign) BOOL needsGridResize;
@property (nonatomic, assign) uint32_t gen;
@property (nonatomic, assign) uint32_t popAlive;
@property (nonatomic, assign) uint32_t popMaxAge;
@property (nonatomic, assign) uint32_t lastStatsGen;
@property (nonatomic, assign) BOOL statsDirty;
@property (nonatomic, assign) int fpsValue;
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
@property (nonatomic, strong) NSTimer *statsTimer;
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
@property (nonatomic, assign) double cursorGridX;
@property (nonatomic, assign) double cursorGridY;
@property (nonatomic, assign) BOOL cursorInside;
@property (nonatomic, assign) BOOL panning;
@property (nonatomic, assign) CGFloat lastPanX;
@property (nonatomic, assign) CGFloat lastPanY;
@property (nonatomic, assign) uint32_t displayMode;
@property (nonatomic, assign) uint32_t palette;
@property (nonatomic, assign) BOOL glowOn;
@property (nonatomic, strong) NSPopUpButton *displayPopup;
@property (nonatomic, strong) NSPopUpButton *palettePopup;
@property (nonatomic, strong) NSArray<NSMenuItem *> *displayItems;
@property (nonatomic, strong) NSArray<NSMenuItem *> *paletteItems;
@property (nonatomic, strong) NSButton *glowButton;
@property (nonatomic, strong) NSSegmentedControl *toolControl;
@property (nonatomic, strong) NSSlider *brushSlider;
@property (nonatomic, strong) NSTextField *brushLabel;
@property (nonatomic, strong) NSTextField *zoomLabel;
@property (nonatomic, strong) NSTextField *hoverLabel;
@property (nonatomic, copy) NSString *screenshotPath;
@property (nonatomic, assign) int screenshotGen;
@property (nonatomic, assign) int screenshotSize;
@property (nonatomic, copy) NSString *launchPreset;

- (BOOL)setupMetal;
- (void)setupUI;
- (void)buildMenus;
- (void)randomize;
- (void)randomize:(id)sender;
- (void)goPause:(id)sender;
- (void)stepOnce:(id)sender;
- (void)clear:(id)sender;
- (void)clear;
- (void)fit:(id)sender;
- (void)fitView;
- (void)updateZoomLabel;
- (void)toolChanged:(id)sender;
- (void)displayChanged:(id)sender;
- (void)setDisplayModeIndex:(NSInteger)index;
- (void)displayMenuClicked:(id)sender;
- (void)paletteChanged:(id)sender;
- (void)setPaletteIndex:(NSInteger)index;
- (void)paletteMenuClicked:(id)sender;
- (void)glowToggle:(id)sender;
- (void)clearTrailNow;
- (void)beginPanAtEvent:(NSEvent *)e;
- (void)panWithEvent:(NSEvent *)e;
- (void)endPan;
 - (void)zoomAtEvent:(NSEvent *)e;
 - (void)zoomByFactor:(double)factor atPoint:(NSPoint)pt;
 - (void)zoomBy:(id)sender;
 - (void)scrollPanWithEvent:(NSEvent *)e;
- (void)hoverAtEvent:(NSEvent *)e;
- (void)hoverExited;
- (void)updateHoverAtPoint:(NSPoint)pt;
- (NSPoint)topPointForEvent:(NSEvent *)e;
- (BOOL)gridCellAtPoint:(NSPoint)pt col:(int *)outCol row:(int *)outRow;
- (BOOL)validateMenuItem:(NSMenuItem *)item;
- (void)sliderChanged:(id)sender;
- (void)paintAtEvent:(NSEvent *)e add:(BOOL)add;
- (void)applyPaintQueue;
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
- (void)runScreenshot;
- (void)advanceGenerations:(int)n;
- (void)renderCellPlane:(uint32_t)plane cb:(id<MTLCommandBuffer>)cb;
- (void)renderFrameToTexture:(id<MTLTexture>)target cb:(id<MTLCommandBuffer>)cb;
- (BOOL)writePNGFromTexture:(id<MTLTexture>)tex path:(NSString *)path;
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
    if (owner == nil) {
        return;
    }
    if (([e modifierFlags] & NSEventModifierFlagCommand) != 0) {
        [owner zoomAtEvent:e];
    } else {
        [owner scrollPanWithEvent:e];
    }
}

- (void)magnifyWithEvent:(NSEvent *)e {
    App *owner = self.owner;
    NSPoint pt;
    if (owner == nil) {
        return;
    }
    pt = [owner topPointForEvent:e];
    [owner zoomByFactor:1.0 + e.magnification atPoint:pt];
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

- (void)updateTrackingAreas {
    NSTrackingArea *old;
    NSTrackingArea *area;
    NSTrackingAreaOptions options;

    for (old in self.trackingAreas) {
        [self removeTrackingArea:old];
    }
    options = (NSTrackingAreaOptions)(NSTrackingMouseEnteredAndExited | NSTrackingMouseMoved |
                                     NSTrackingActiveInKeyWindow | NSTrackingInVisibleRect);
    area = [[NSTrackingArea alloc] initWithRect:self.bounds
                                       options:options
                                         owner:self
                                       userInfo:nil];
    [self addTrackingArea:area];
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
@synthesize stepOnce = _stepOnce;
@synthesize dirty = _dirty;
@synthesize needsGridResize = _needsGridResize;
@synthesize gen = _gen;
@synthesize popAlive = _popAlive;
@synthesize popMaxAge = _popMaxAge;
@synthesize lastStatsGen = _lastStatsGen;
@synthesize statsDirty = _statsDirty;
@synthesize fpsValue = _fpsValue;
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
@synthesize statsTimer = _statsTimer;
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
@synthesize cursorGridX = _cursorGridX;
@synthesize cursorGridY = _cursorGridY;
@synthesize cursorInside = _cursorInside;
@synthesize panning = _panning;
@synthesize lastPanX = _lastPanX;
@synthesize lastPanY = _lastPanY;
@synthesize displayMode = _displayMode;
@synthesize palette = _palette;
@synthesize glowOn = _glowOn;
@synthesize displayPopup = _displayPopup;
@synthesize palettePopup = _palettePopup;
@synthesize displayItems = _displayItems;
@synthesize paletteItems = _paletteItems;
@synthesize glowButton = _glowButton;
@synthesize toolControl = _toolControl;
@synthesize brushSlider = _brushSlider;
@synthesize brushLabel = _brushLabel;
@synthesize zoomLabel = _zoomLabel;
@synthesize hoverLabel = _hoverLabel;
@synthesize screenshotPath = _screenshotPath;
@synthesize screenshotGen = _screenshotGen;
@synthesize screenshotSize = _screenshotSize;
@synthesize launchPreset = _launchPreset;

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
    NSString *screenshot = nil;
    int gen = 300;
    int size = 1024;
    int genSet = 0;
    int sizeSet = 0;
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
            else if ([a caseInsensitiveCompare:@"--screenshot"] == NSOrderedSame) { screenshot = n; i++; }
            else if ([a caseInsensitiveCompare:@"--gen"] == NSOrderedSame) { gen = atoi(n.UTF8String); genSet = 1; i++; }
            else if ([a caseInsensitiveCompare:@"--size"] == NSOrderedSame) { size = atoi(n.UTF8String); sizeSet = 1; i++; }
        }
    }
    if (mode == nil && (env = getenv("GOL_MODE")) != NULL) mode = [NSString stringWithUTF8String:env];
    if (preset == nil && (env = getenv("GOL_PRESET")) != NULL) preset = [NSString stringWithUTF8String:env];
    if (palette == nil && (env = getenv("GOL_PALETTE")) != NULL) palette = [NSString stringWithUTF8String:env];
    if (!densitySet && (env = getenv("GOL_DENSITY")) != NULL) { density = atof(env); densitySet = 1; }
    if (!zoomSet && (env = getenv("GOL_ZOOM")) != NULL) { zoom = atof(env); zoomSet = 1; }
    if (!runSet && (env = getenv("GOL_RUN")) != NULL) { run = atoi(env); runSet = 1; }
    if (screenshot == nil && (env = getenv("GOL_SCREENSHOT")) != NULL) screenshot = [NSString stringWithUTF8String:env];
    if (!genSet && (env = getenv("GOL_GEN")) != NULL) { gen = atoi(env); genSet = 1; }
    if (!sizeSet && (env = getenv("GOL_SIZE")) != NULL) { size = atoi(env); sizeSet = 1; }
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
        if ([m isEqualToString:@"age"]) [self setDisplayModeIndex:(NSInteger)DISPLAY_AGE];
        else if ([m isEqualToString:@"trails"]) [self setDisplayModeIndex:(NSInteger)DISPLAY_TRAILS];
        else if ([m isEqualToString:@"heatmap"]) [self setDisplayModeIndex:(NSInteger)DISPLAY_HEATMAP];
    }
    if (palette != nil && palette.length > 0) {
        NSString *p = [palette lowercaseString];
        if ([p isEqualToString:@"inferno"]) [self setPaletteIndex:(NSInteger)PALETTE_INFERNO];
        else if ([p isEqualToString:@"plasma"]) [self setPaletteIndex:(NSInteger)PALETTE_PLASMA];
        else if ([p isEqualToString:@"turbo"]) [self setPaletteIndex:(NSInteger)PALETTE_TURBO];
        else [self setPaletteIndex:(NSInteger)PALETTE_VIRIDIS];
    }
    self.launchPreset = preset;
    if (screenshot != nil && screenshot.length > 0) {
        self.screenshotPath = screenshot;
        self.screenshotGen = (gen > 0) ? gen : 300;
        self.screenshotSize = (size > 0) ? size : 1024;
    }
    if (preset != nil && preset.length > 0) {
        [self applyPreset:[preset UTF8String]];
    } else {
        [self randomize];
    }
}

- (void)applicationDidFinishLaunching:(NSNotification *)note {
    CFTimeInterval now;
    NSRunLoop *runLoop;
    (void)note;
    now = CFAbsoluteTimeGetCurrent();
    self.frameIndex = 0;
    self.displayPlane = 0;
    self.completedPlane = 0;
    self.gridW = INITIAL_GRID_W;
    self.gridH = INITIAL_GRID_H;
    self.gen = 0;
    self.lastStatsGen = 0;
    self.statsDirty = NO;
    self.fpsValue = -1;
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
    [self buildMenus];
    [self setupUI];
    [self applyLaunchConfig];

    if (self.screenshotPath != nil) {
        [self runScreenshot];
        return;
    }

    self.statsTimer = [NSTimer timerWithTimeInterval:1.0 / 15.0
                                              target:self
                                            selector:@selector(flushStats)
                                            userInfo:nil
                                             repeats:YES];
    runLoop = [NSRunLoop mainRunLoop];
    [runLoop addTimer:self.statsTimer forMode:NSRunLoopCommonModes];
}

- (void)buildMenus {
    NSMenu *mainMenu;
    NSMenuItem *item;
    NSMenu *appMenu;
    NSMenu *simMenu;
    NSMenu *viewMenu;
    NSMenu *paletteMenu;
    NSMenu *windowMenu;
    NSMutableArray<NSMenuItem *> *displayItems;
    NSMutableArray<NSMenuItem *> *paletteItems;

    mainMenu = [[NSMenu alloc] init];

    // App menu.
    item = [mainMenu addItemWithTitle:@"" action:nil keyEquivalent:@""];
    appMenu = [[NSMenu alloc] init];
    item.submenu = appMenu;
    MakeMenuItem(appMenu, @"About GOL", @selector(orderFrontStandardAboutPanel:), @"", 0);
    [appMenu addItem:[NSMenuItem separatorItem]];
    MakeMenuItem(appMenu, @"Hide GOL", @selector(hide:), @"h", NSEventModifierFlagCommand);
    MakeMenuItem(appMenu, @"Hide Others", @selector(hideOtherApplications:), @"h",
                 NSEventModifierFlagCommand | NSEventModifierFlagOption);
    MakeMenuItem(appMenu, @"Show All", @selector(unhideAllApplications:), @"", 0);
    [appMenu addItem:[NSMenuItem separatorItem]];
    MakeMenuItem(appMenu, @"Quit GOL", @selector(terminate:), @"q", NSEventModifierFlagCommand);

    // Simulation menu.
    item = [mainMenu addItemWithTitle:@"Simulation" action:nil keyEquivalent:@""];
    simMenu = [[NSMenu alloc] init];
    item.submenu = simMenu;
    item = MakeMenuItem(simMenu, @"Go", @selector(goPause:), @" ", 0);
    item.target = self;
    item = MakeMenuItem(simMenu, @"Step", @selector(stepOnce:), @".", 0);
    item.target = self;
    [simMenu addItem:[NSMenuItem separatorItem]];
    item = MakeMenuItem(simMenu, @"Random", @selector(randomize:), @"z", 0);
    item.target = self;
    item = MakeMenuItem(simMenu, @"Clear", @selector(clear:), @"r", 0);
    item.target = self;
    [simMenu addItem:[NSMenuItem separatorItem]];
    item = MakeMenuItem(simMenu, @"Fit", @selector(fit:), @"f", 0);
    item.target = self;
    item = MakeMenuItem(simMenu, @"Fit", @selector(fit:), @"0", 0);
    item.target = self;
    item.hidden = YES;

    // View menu.
    item = [mainMenu addItemWithTitle:@"View" action:nil keyEquivalent:@""];
    viewMenu = [[NSMenu alloc] init];
    item.submenu = viewMenu;
    displayItems = [NSMutableArray array];
    item = MakeMenuItem(viewMenu, @"Age", @selector(displayMenuClicked:), @"1", 0);
    item.target = self;
    item.tag = (int)DISPLAY_AGE;
    [displayItems addObject:item];
    item = MakeMenuItem(viewMenu, @"Trails", @selector(displayMenuClicked:), @"2", 0);
    item.target = self;
    item.tag = (int)DISPLAY_TRAILS;
    [displayItems addObject:item];
    item = MakeMenuItem(viewMenu, @"Heatmap", @selector(displayMenuClicked:), @"3", 0);
    item.target = self;
    item.tag = (int)DISPLAY_HEATMAP;
    [displayItems addObject:item];
    self.displayItems = displayItems;

    item = [viewMenu addItemWithTitle:@"Palette" action:nil keyEquivalent:@""];
    paletteMenu = [[NSMenu alloc] init];
    item.submenu = paletteMenu;
    paletteItems = [NSMutableArray array];
    item = MakeMenuItem(paletteMenu, @"Viridis", @selector(paletteMenuClicked:), @"", 0);
    item.target = self;
    item.tag = (int)PALETTE_VIRIDIS;
    [paletteItems addObject:item];
    item = MakeMenuItem(paletteMenu, @"Inferno", @selector(paletteMenuClicked:), @"", 0);
    item.target = self;
    item.tag = (int)PALETTE_INFERNO;
    [paletteItems addObject:item];
    item = MakeMenuItem(paletteMenu, @"Plasma", @selector(paletteMenuClicked:), @"", 0);
    item.target = self;
    item.tag = (int)PALETTE_PLASMA;
    [paletteItems addObject:item];
    item = MakeMenuItem(paletteMenu, @"Turbo", @selector(paletteMenuClicked:), @"", 0);
    item.target = self;
    item.tag = (int)PALETTE_TURBO;
    [paletteItems addObject:item];
    self.paletteItems = paletteItems;

    item = MakeMenuItem(viewMenu, @"Glow", @selector(glowToggle:), @"g", 0);
    item.target = self;
    [viewMenu addItem:[NSMenuItem separatorItem]];
    item = MakeMenuItem(viewMenu, @"Zoom In", @selector(zoomBy:), @"+", NSEventModifierFlagCommand);
    item.target = self;
    item.tag = 1;
    item = MakeMenuItem(viewMenu, @"Zoom Out", @selector(zoomBy:), @"-", NSEventModifierFlagCommand);
    item.target = self;
    item.tag = 2;

    // Window menu.
    item = [mainMenu addItemWithTitle:@"Window" action:nil keyEquivalent:@""];
    windowMenu = [[NSMenu alloc] init];
    item.submenu = windowMenu;
    MakeMenuItem(windowMenu, @"Close", @selector(performClose:), @"w", NSEventModifierFlagCommand);
    MakeMenuItem(windowMenu, @"Minimize", @selector(performMiniaturize:), @"m", NSEventModifierFlagCommand);
    NSApp.windowsMenu = windowMenu;

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

    // Prefer the bundle's default library (works when packaged as GOL.app);
    // fall back to the executable directory for an unbundled ./bin/gol.
    self.library = [self.device newDefaultLibrary];
    if (self.library == nil) {
        exeDir = [[NSBundle mainBundle] executablePath];
        exeDir = [exeDir stringByDeletingLastPathComponent];
        libPath = [exeDir stringByAppendingPathComponent:@"default.metallib"];
        libURL = [NSURL fileURLWithPath:libPath];
        err = nil;
        self.library = [self.device newLibraryWithURL:libURL error:&err];
    }
    if (self.library == nil) {
        NSLog(@"failed to load metallib: %@", err);
        return NO;
    }

    self.mtkView = [[GOLView alloc] initWithFrame:NSMakeRect(0, 0, VIEW_W, VIEW_H)
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
    self.maxGridH = FloorInt(screenH / MIN_CELL_PX) + 16;
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
    self.goButton.image = [NSImage imageWithSystemSymbolName:(self.running ? @"pause.fill" : @"play.fill")
                                          accessibilityDescription:(self.running ? @"Pause" : @"Go")];
    self.goButton.toolTip = self.running ? @"Pause" : @"Go";
    if (self.running) {
        self.mtkView.paused = NO;
    } else if (!self.dirty) {
        self.mtkView.paused = YES;
    }
}

- (void)stepOnce:(id)sender {
    (void)sender;
    self.stepOnce = YES;
    [self markDirty];
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
    if (self.popSpark != nil) {
        [self.popSpark pushValue:(int)alive];
    }
    self.statsDirty = YES;
    if (!self.running) {
        [self flushStats];
    }
}

- (void)flushStats {
    if (!self.statsDirty) {
        return;
    }
    if (self.genLabel != nil) {
        self.genLabel.stringValue = [NSString stringWithFormat:@"Gen %u", self.gen];
    }
    if (self.popLabel != nil) {
        self.popLabel.stringValue = [NSString stringWithFormat:@"Pop %u", self.popAlive];
    }
    if (self.maxAgeLabel != nil) {
        self.maxAgeLabel.stringValue = [NSString stringWithFormat:@"Age %u", self.popMaxAge];
    }
    if (self.fpsLabel != nil) {
        self.fpsLabel.stringValue = (self.fpsValue < 0) ? @"-- fps"
            : [NSString stringWithFormat:@"%d fps", self.fpsValue];
    }
    if (self.popSpark != nil) {
        [self.popSpark setNeedsDisplay:YES];
    }
    self.statsDirty = NO;
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

- (void)toolChanged:(id)sender {
    (void)sender;
    if (self.toolControl != nil && self.toolControl.selectedSegment >= 0) {
        self.tool = (int)self.toolControl.selectedSegment;
    } else {
        self.tool = 0;
    }
}

- (void)displayChanged:(id)sender {
    (void)sender;
    if (self.displayPopup == nil) {
        return;
    }
    [self setDisplayModeIndex:self.displayPopup.indexOfSelectedItem];
}

- (void)setDisplayModeIndex:(NSInteger)index {
    NSUInteger i;
    if (index < 0 || index > (int)DISPLAY_HEATMAP) {
        return;
    }
    self.displayMode = (uint32_t)index;
    if (self.displayMode == DISPLAY_TRAILS) {
        [self clearTrailNow];
    }
    if (self.displayPopup != nil) {
        [self.displayPopup selectItemAtIndex:index];
    }
    for (i = 0; i < self.displayItems.count; i++) {
        self.displayItems[i].state = (i == (NSUInteger)index) ? NSControlStateValueOn : NSControlStateValueOff;
    }
    [self markDirty];
}

- (void)displayMenuClicked:(id)sender {
    [self setDisplayModeIndex:((NSMenuItem *)sender).tag];
}

- (void)paletteChanged:(id)sender {
    (void)sender;
    if (self.palettePopup == nil) {
        return;
    }
    [self setPaletteIndex:self.palettePopup.indexOfSelectedItem];
}

- (void)setPaletteIndex:(NSInteger)index {
    NSUInteger i;
    if (index < 0 || index > (int)PALETTE_TURBO) {
        return;
    }
    self.palette = (uint32_t)index;
    if (self.palettePopup != nil) {
        [self.palettePopup selectItemAtIndex:index];
    }
    for (i = 0; i < self.paletteItems.count; i++) {
        self.paletteItems[i].state = (i == (NSUInteger)index) ? NSControlStateValueOn : NSControlStateValueOff;
    }
    [self markDirty];
}

- (void)paletteMenuClicked:(id)sender {
    [self setPaletteIndex:((NSMenuItem *)sender).tag];
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
    double dy;
    double factor;

    if (self.mtkView == nil || self.cellPx < 0.01) {
        return;
    }
    pt = [self topPointForEvent:e];
    dy = e.scrollingDeltaY * (e.hasPreciseScrollingDeltas ? 0.02 : 0.1);
    factor = exp2(dy);
    factor = ClampDouble(factor, 0.5, 2.0);
    [self zoomByFactor:factor atPoint:pt];
}

- (void)zoomByFactor:(double)factor atPoint:(NSPoint)pt {
    double gx;
    double gy;
    double newCellPx;

    if (self.mtkView == nil || self.cellPx < 0.01) {
        return;
    }
    gx = (pt.x - self.viewOffsetX) / self.cellPx;
    gy = (pt.y - self.viewOffsetY) / self.cellPx;
    newCellPx = (double)self.cellPx * factor;
    newCellPx = ClampDouble(newCellPx, (double)MIN_CELL_PX, (double)MAX_CELL_PX);
    self.cellPx = (CGFloat)newCellPx;
    self.viewOffsetX = pt.x - (CGFloat)gx * self.cellPx;
    self.viewOffsetY = pt.y - (CGFloat)gy * self.cellPx;
    [self updateZoomLabel];
    [self requestGridResize];
}

- (void)scrollPanWithEvent:(NSEvent *)e {
    double k;
    // Precise (trackpad) deltas are in points; line-based wheel deltas are
    // scaled up so a wheel mouse pans at a usable speed. The view is not
    // flipped, so Y is negated to match topPointForEvent: coordinates.
    k = e.hasPreciseScrollingDeltas ? 1.0 : 8.0;
    self.viewOffsetX += (CGFloat)(e.scrollingDeltaX * k);
    self.viewOffsetY -= (CGFloat)(e.scrollingDeltaY * k);
    [self markDirty];
}

- (void)zoomBy:(id)sender {
    NSRect bounds;
    double factor;
    double gx;
    double gy;
    double newCellPx;

    if (self.mtkView == nil || self.cellPx < 0.01) {
        return;
    }
    bounds = [self.mtkView bounds];
    if (bounds.size.width < 1.0 || bounds.size.height < 1.0) {
        return;
    }
    factor = (((NSMenuItem *)sender).tag == 1) ? 2.0 : 0.5;
    gx = (bounds.size.width / 2.0 - self.viewOffsetX) / self.cellPx;
    gy = (bounds.size.height / 2.0 - self.viewOffsetY) / self.cellPx;
    newCellPx = (double)self.cellPx * factor;
    newCellPx = ClampDouble(newCellPx, (double)MIN_CELL_PX, (double)MAX_CELL_PX);
    self.cellPx = (CGFloat)newCellPx;
    self.viewOffsetX = bounds.size.width / 2.0 - (CGFloat)gx * self.cellPx;
    self.viewOffsetY = bounds.size.height / 2.0 - (CGFloat)gy * self.cellPx;
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
    if (self.cursorInside) {
        self.cursorInside = NO;
        [self markDirty];
    }
}

- (void)updateHoverAtPoint:(NSPoint)pt {
    int col;
    int row;
    uint16_t *cells;
    uint16_t v;
    size_t idx;
    NSString *text;
    double gx;
    double gy;

    col = 0;
    row = 0;
    if (self.hoverLabel == nil) {
        return;
    }
    if (self.cellPx >= 0.01) {
        gx = (pt.x - self.viewOffsetX) / self.cellPx;
        gy = (pt.y - self.viewOffsetY) / self.cellPx;
        if (!self.cursorInside ||
            FloorInt(gx) != FloorInt(self.cursorGridX) ||
            FloorInt(gy) != FloorInt(self.cursorGridY)) {
            self.cursorGridX = gx;
            self.cursorGridY = gy;
            self.cursorInside = YES;
            [self markDirty];
        } else {
            self.cursorGridX = gx;
            self.cursorGridY = gy;
        }
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

- (BOOL)validateMenuItem:(NSMenuItem *)item {
    if (item.action == @selector(goPause:)) {
        item.title = self.running ? @"Pause" : @"Go";
    } else if (item.action == @selector(displayMenuClicked:)) {
        item.state = (item.tag == (NSInteger)self.displayMode) ? NSControlStateValueOn : NSControlStateValueOff;
    } else if (item.action == @selector(paletteMenuClicked:)) {
        item.state = (item.tag == (NSInteger)self.palette) ? NSControlStateValueOn : NSControlStateValueOff;
    }
    return YES;
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

    if (self.mtkView == nil) {
        return;
    }
    pt = [self topPointForEvent:e];
    if (![self gridCellAtPoint:pt col:&col row:&row]) {
        return;
    }
    // Queue the stroke instead of editing the plane now. A fast drag fires many
    // events per frame, and each used to force a waitAll that stalled the pipeline;
    // the records are applied together at the top of the next tick.
    if (paintCount >= kPaintQueueCap) {
        memmove(&paintQueue[0], &paintQueue[1], (size_t)(kPaintQueueCap - 1) * sizeof(PaintOp));
        paintCount = kPaintQueueCap - 1;
    }
    paintQueue[paintCount].col = col;
    paintQueue[paintCount].row = row;
    paintQueue[paintCount].radius = self.brushRadius;
    paintQueue[paintCount].add = add;
    paintCount++;
    [self markDirty];
    [self updateHoverAtPoint:pt];
}

- (void)applyPaintQueue {
    uint16_t *cells;
    int alive;
    int maxAge;
    int i;

    if (paintCount == 0) {
        return;
    }
    [self waitAll];
    cells = [self currentCells];
    if (cells == nil) {
        paintCount = 0;
        return;
    }
    for (i = 0; i < paintCount; i++) {
        int dx;
        int dy;
        int cx;
        int cy;
        for (dy = -paintQueue[i].radius; dy <= paintQueue[i].radius; dy++) {
            for (dx = -paintQueue[i].radius; dx <= paintQueue[i].radius; dx++) {
                cx = paintQueue[i].col + dx;
                cy = paintQueue[i].row + dy;
                if (cx >= 0 && cx < self.gridW && cy >= 0 && cy < self.gridH) {
                    gol_set_plane(cells, self.gridW, self.gridH, cx, cy, paintQueue[i].add);
                }
            }
        }
    }
    // While running the step kernel recomputes the population via its stats slot,
    // so only fall back to a CPU count on the paused path (avoids a full-grid pass).
    if (!self.running) {
        alive = 0;
        maxAge = 0;
        gol_count_alive(cells, self.gridW, self.gridH, &alive, &maxAge);
        [self setPopulation:(uint32_t)alive maxAge:(uint32_t)maxAge forGeneration:self.gen];
    }
    paintCount = 0;
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
    paintCount = 0;
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
                                                       mipmapped:YES];
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
    NSStackView *bar;
    NSStackView *simRow;
    NSStackView *viewRow;
    NSStackView *statusRow;
    NSButton *clearButton;
    NSButton *randomButton;
    NSButton *stepButton;
    NSButton *fitButton;
    NSPopUpButton *presetsPopup;
    NSTextField *presetName;
    NSTextField *speedName;
    NSTextField *densityName;
    NSTextField *ruleName;
    NSTextField *displayName;
    NSTextField *paletteName;
    NSTextField *toolName;
    NSTextField *brushName;
    NSTextField *zoomName;
    NSTextField *trendName;
    NSBox *separator;
    NSView *spacer;
    NSView *sidePanel;
    __weak App *weakSelf;

    content = NSMakeRect(0, 0, VIEW_W, VIEW_H + BAR_H);
    self.window = [[NSWindow alloc] initWithContentRect:content
                                              styleMask:(NSWindowStyleMask)(NSWindowStyleMaskTitled |
                                                                            NSWindowStyleMaskClosable |
                                                                            NSWindowStyleMaskMiniaturizable |
                                                                            NSWindowStyleMaskResizable)
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
    self.window.appearance = [NSAppearance appearanceNamed:NSAppearanceNameDarkAqua];
    self.window.releasedWhenClosed = NO;
    [self.window setTitle:@"Game of Life"];
    [self.window setContentSize:content.size];
    [self.window setBackgroundColor:[NSColor colorWithSRGBRed:0.04 green:0.05 blue:0.08 alpha:1.0]];
    [self.window setDelegate:self];
    self.window.acceptsMouseMovedEvents = YES;
    [self.window center];

    screen = ScreenForWindow(self.window);
    self.window.maxSize = NSMakeSize(screen.frame.size.width, screen.frame.size.height);

    contentView = [self.window contentView];

    // The Metal view fills the top of the window, down to just above the bar
    // and left of the side panel.
    self.mtkView.translatesAutoresizingMaskIntoConstraints = NO;
    [contentView addSubview:self.mtkView];

    // The bar is a vertical stack of three horizontal rows.
    bar = [[NSStackView alloc] init];
    bar.orientation = NSUserInterfaceLayoutOrientationVertical;
    bar.alignment = NSLayoutAttributeLeading;
    bar.spacing = 6.0;
    bar.translatesAutoresizingMaskIntoConstraints = NO;
    bar.wantsLayer = YES;
    [contentView addSubview:bar];

    // --- Simulation row: transport, then preset and the sliders. ---
    simRow = [[NSStackView alloc] init];
    simRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    simRow.alignment = NSLayoutAttributeCenterY;
    simRow.spacing = 10.0;

    self.goButton = [[NSButton alloc] initWithFrame:NSMakeRect(0, 0, 32, 24)];
    self.goButton.title = @"Go";
    self.goButton.image = [NSImage imageWithSystemSymbolName:@"play.fill" accessibilityDescription:@"Go"];
    self.goButton.imagePosition = NSImageOnly;
    self.goButton.bezelStyle = NSBezelStyleTexturedRounded;
    self.goButton.toolTip = @"Go";
    self.goButton.target = self;
    self.goButton.action = @selector(goPause:);
    StackAdd(simRow, self.goButton);

    stepButton = [[NSButton alloc] initWithFrame:NSMakeRect(0, 0, 32, 24)];
    stepButton.title = @"Step";
    stepButton.image = [NSImage imageWithSystemSymbolName:@"forward.frame.fill" accessibilityDescription:@"Step"];
    stepButton.imagePosition = NSImageOnly;
    stepButton.bezelStyle = NSBezelStyleTexturedRounded;
    stepButton.toolTip = @"Step";
    stepButton.target = self;
    stepButton.action = @selector(stepOnce:);
    StackAdd(simRow, stepButton);

    randomButton = [[NSButton alloc] initWithFrame:NSMakeRect(0, 0, 32, 24)];
    randomButton.title = @"Random";
    randomButton.image = [NSImage imageWithSystemSymbolName:@"dice.fill" accessibilityDescription:@"Random"];
    randomButton.imagePosition = NSImageOnly;
    randomButton.bezelStyle = NSBezelStyleTexturedRounded;
    randomButton.toolTip = @"Random";
    randomButton.target = self;
    randomButton.action = @selector(randomize:);
    StackAdd(simRow, randomButton);

    clearButton = [[NSButton alloc] initWithFrame:NSMakeRect(0, 0, 32, 24)];
    clearButton.title = @"Clear";
    clearButton.image = [NSImage imageWithSystemSymbolName:@"trash" accessibilityDescription:@"Clear"];
    clearButton.imagePosition = NSImageOnly;
    clearButton.bezelStyle = NSBezelStyleTexturedRounded;
    clearButton.toolTip = @"Clear";
    clearButton.target = self;
    clearButton.action = @selector(clear:);
    StackAdd(simRow, clearButton);

    presetsPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(0, 0, 110, 24) pullsDown:NO];
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
    presetName = MakeLabelSmall(@"Preset", NSZeroRect);
    StackAdd(simRow, HGroup(presetName, presetsPopup, nil));

    self.speedSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(0, 0, 105, 20)];
    self.speedSlider.minValue = 1.0;
    self.speedSlider.maxValue = 600.0;
    self.speedSlider.doubleValue = 30.0;
    self.speedSlider.allowsTickMarkValuesOnly = NO;
    self.speedSlider.numberOfTickMarks = 7;
    self.speedSlider.continuous = YES;
    self.speedSlider.target = self;
    self.speedSlider.action = @selector(sliderChanged:);
    self.speedSlider.translatesAutoresizingMaskIntoConstraints = NO;
    [self.speedSlider.widthAnchor constraintEqualToConstant:105.0].active = YES;
    self.speedLabel = MakeValueLabel(@"30 gen/s", NSZeroRect);
    speedName = MakeLabelSmall(@"Speed", NSZeroRect);
    StackAdd(simRow, HGroup(speedName, self.speedSlider, self.speedLabel));

    self.densitySlider = [[NSSlider alloc] initWithFrame:NSMakeRect(0, 0, 90, 20)];
    self.densitySlider.minValue = 0.0;
    self.densitySlider.maxValue = 1.0;
    self.densitySlider.doubleValue = 0.2;
    self.densitySlider.allowsTickMarkValuesOnly = NO;
    self.densitySlider.numberOfTickMarks = 0;
    self.densitySlider.continuous = YES;
    self.densitySlider.target = self;
    self.densitySlider.action = @selector(sliderChanged:);
    self.densitySlider.translatesAutoresizingMaskIntoConstraints = NO;
    [self.densitySlider.widthAnchor constraintEqualToConstant:90.0].active = YES;
    self.densityPct = MakeValueLabel(@"20%", NSZeroRect);
    densityName = MakeLabelSmall(@"Density", NSZeroRect);
    StackAdd(simRow, HGroup(densityName, self.densitySlider, self.densityPct));

    [bar addArrangedSubview:simRow];

    // --- Rule and view row. ---
    viewRow = [[NSStackView alloc] init];
    viewRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    viewRow.alignment = NSLayoutAttributeCenterY;
    viewRow.spacing = 10.0;

    ruleName = MakeLabelSmall(@"Rule", NSZeroRect);
    self.rulePopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(0, 0, 115, 24) pullsDown:NO];
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
    StackAdd(viewRow, HGroup(ruleName, self.rulePopup, nil));

    self.ruleToggleView = [[GOLRuleToggleView alloc] initWithFrame:NSMakeRect(0, 0, 165, 34)];
    self.ruleToggleView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.ruleToggleView.widthAnchor constraintEqualToConstant:165.0].active = YES;
    [self.ruleToggleView.heightAnchor constraintEqualToConstant:34.0].active = YES;
    StackAdd(viewRow, self.ruleToggleView);

    self.ruleLabel = MakeLabel(@"B3/S23", NSZeroRect);
    StackAdd(viewRow, self.ruleLabel);

    separator = [[NSBox alloc] init];
    separator.translatesAutoresizingMaskIntoConstraints = NO;
    separator.boxType = NSBoxSeparator;
    [separator.widthAnchor constraintEqualToConstant:1.0].active = YES;
    StackAdd(viewRow, separator);
    [separator.heightAnchor constraintEqualToAnchor:self.ruleToggleView.heightAnchor].active = YES;

    displayName = MakeLabelSmall(@"Display", NSZeroRect);
    self.displayPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(0, 0, 90, 24) pullsDown:NO];
    [self.displayPopup addItemWithTitle:@"Age"];
    [self.displayPopup addItemWithTitle:@"Trails"];
    [self.displayPopup addItemWithTitle:@"Heatmap"];
    [self.displayPopup selectItemAtIndex:0];
    self.displayPopup.target = self;
    self.displayPopup.action = @selector(displayChanged:);
    StackAdd(viewRow, HGroup(displayName, self.displayPopup, nil));

    paletteName = MakeLabelSmall(@"Palette", NSZeroRect);
    self.palettePopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(0, 0, 90, 24) pullsDown:NO];
    [self.palettePopup addItemWithTitle:@"Viridis"];
    [self.palettePopup addItemWithTitle:@"Inferno"];
    [self.palettePopup addItemWithTitle:@"Plasma"];
    [self.palettePopup addItemWithTitle:@"Turbo"];
    [self.palettePopup selectItemAtIndex:(int)self.palette];
    self.palettePopup.target = self;
    self.palettePopup.action = @selector(paletteChanged:);
    StackAdd(viewRow, HGroup(paletteName, self.palettePopup, nil));

    self.glowButton = [[NSButton alloc] initWithFrame:NSMakeRect(0, 0, 54, 24)];
    self.glowButton.title = @"Glow";
    [self.glowButton setButtonType:NSButtonTypeSwitch];
    self.glowButton.state = self.glowOn ? NSControlStateValueOn : NSControlStateValueOff;
    self.glowButton.target = self;
    self.glowButton.action = @selector(glowToggle:);
    StackAdd(viewRow, self.glowButton);

    toolName = MakeLabelSmall(@"Tool", NSZeroRect);
    self.toolControl = [[NSSegmentedControl alloc] initWithFrame:NSMakeRect(0, 0, 90, 24)];
    self.toolControl.segmentCount = 2;
    [self.toolControl setLabel:@"Add" forSegment:0];
    [self.toolControl setLabel:@"Erase" forSegment:1];
    self.toolControl.trackingMode = NSSegmentSwitchTrackingSelectOne;
    self.toolControl.selectedSegment = 0;
    self.toolControl.target = self;
    self.toolControl.action = @selector(toolChanged:);
    self.toolControl.toolTip = @"Left-click paints with the selected tool; right-click does the opposite. Option-drag or middle-drag pans.";
    StackAdd(viewRow, HGroup(toolName, self.toolControl, nil));

    brushName = MakeLabelSmall(@"Brush", NSZeroRect);
    self.brushSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(0, 0, 90, 20)];
    self.brushSlider.minValue = 0.0;
    self.brushSlider.maxValue = 10.0;
    self.brushSlider.doubleValue = 1.0;
    self.brushSlider.allowsTickMarkValuesOnly = NO;
    self.brushSlider.numberOfTickMarks = 0;
    self.brushSlider.continuous = YES;
    self.brushSlider.target = self;
    self.brushSlider.action = @selector(sliderChanged:);
    self.brushSlider.translatesAutoresizingMaskIntoConstraints = NO;
    [self.brushSlider.widthAnchor constraintEqualToConstant:90.0].active = YES;
    self.brushLabel = MakeValueLabel(@"1", NSZeroRect);
    StackAdd(viewRow, HGroup(brushName, self.brushSlider, self.brushLabel));

    zoomName = MakeLabelSmall(@"Zoom", NSZeroRect);
    self.zoomLabel = MakeValueLabel(@"100%", NSZeroRect);
    StackAdd(viewRow, HGroup(zoomName, self.zoomLabel, nil));

    fitButton = [[NSButton alloc] initWithFrame:NSMakeRect(0, 0, 32, 24)];
    fitButton.title = @"Fit";
    fitButton.image = [NSImage imageWithSystemSymbolName:@"arrow.up.left.and.arrow.down.right"
                              accessibilityDescription:@"Fit"];
    fitButton.imagePosition = NSImageOnly;
    fitButton.bezelStyle = NSBezelStyleTexturedRounded;
    fitButton.toolTip = @"Fit";
    fitButton.target = self;
    fitButton.action = @selector(fit:);
    StackAdd(viewRow, fitButton);

    [bar addArrangedSubview:viewRow];

    // --- Status row: live readouts on the left, trend sparkline on the right. ---
    statusRow = [[NSStackView alloc] init];
    statusRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    statusRow.alignment = NSLayoutAttributeCenterY;
    statusRow.spacing = 10.0;

    self.fpsLabel = MakeValueLabel(@"-- fps", NSZeroRect);
    StackAdd(statusRow, self.fpsLabel);

    self.genLabel = MakeValueLabel(@"Gen 0", NSZeroRect);
    StackAdd(statusRow, self.genLabel);

    self.popLabel = MakeValueLabel(@"Pop 0", NSZeroRect);
    StackAdd(statusRow, self.popLabel);

    self.maxAgeLabel = MakeValueLabel(@"Age 0", NSZeroRect);
    StackAdd(statusRow, self.maxAgeLabel);

    spacer = [[NSView alloc] init];
    [spacer setContentHuggingPriority:NSLayoutPriorityDefaultLow forOrientation:NSLayoutConstraintOrientationHorizontal];
    [spacer setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow forOrientation:NSLayoutConstraintOrientationHorizontal];
    {
        NSLayoutConstraint *w = [spacer.widthAnchor constraintEqualToConstant:0.0];
        w.priority = NSLayoutPriorityDefaultLow;
        w.active = YES;
    }
    [spacer.heightAnchor constraintEqualToConstant:1.0].active = YES;
    StackAdd(statusRow, spacer);

    trendName = MakeLabelSmall(@"Trend", NSZeroRect);
    self.popSpark = [[GOLSparklineView alloc] initWithFrame:NSMakeRect(0, 0, 120, 24)];
    self.popSpark.toolTip = @"Population over recent generations";
    self.popSpark.translatesAutoresizingMaskIntoConstraints = NO;
    [self.popSpark.widthAnchor constraintEqualToConstant:120.0].active = YES;
    [self.popSpark.heightAnchor constraintEqualToConstant:24.0].active = YES;
    StackAdd(statusRow, HGroup(trendName, self.popSpark, nil));

    [bar addArrangedSubview:statusRow];

    // --- Side panel: the hover readout, right of the canvas. Its width is
    // fixed, so the changing text can never move the canvas. ---
    sidePanel = [[NSView alloc] init];
    sidePanel.translatesAutoresizingMaskIntoConstraints = NO;
    sidePanel.wantsLayer = YES;
    sidePanel.layer.cornerRadius = 6.0;
    sidePanel.layer.backgroundColor = [NSColor colorWithSRGBRed:0.07 green:0.08 blue:0.11 alpha:1.0].CGColor;
    [contentView addSubview:sidePanel];

    self.hoverLabel = MakeLabelSmall(@"--", NSZeroRect);
    self.hoverLabel.lineBreakMode = NSLineBreakByTruncatingTail;
    [self.hoverLabel setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow forOrientation:NSLayoutConstraintOrientationHorizontal];
    self.hoverLabel.translatesAutoresizingMaskIntoConstraints = NO;
    [sidePanel addSubview:self.hoverLabel];

    // --- Pin the bar to the bottom; the Metal view fills the space above it
    // and left of the side panel. ---
    [NSLayoutConstraint activateConstraints:@[
        [bar.leadingAnchor constraintEqualToAnchor:contentView.leadingAnchor constant:8.0],
        [bar.trailingAnchor constraintEqualToAnchor:contentView.trailingAnchor constant:-8.0],
        [bar.bottomAnchor constraintEqualToAnchor:contentView.bottomAnchor constant:-8.0],
        [statusRow.widthAnchor constraintEqualToAnchor:bar.widthAnchor],
        [self.mtkView.leadingAnchor constraintEqualToAnchor:contentView.leadingAnchor constant:8.0],
        [self.mtkView.trailingAnchor constraintEqualToAnchor:sidePanel.leadingAnchor constant:-8.0],
        [self.mtkView.topAnchor constraintEqualToAnchor:contentView.topAnchor constant:8.0],
        [self.mtkView.bottomAnchor constraintEqualToAnchor:bar.topAnchor constant:-8.0],
        [sidePanel.widthAnchor constraintEqualToConstant:SIDE_W],
        [sidePanel.trailingAnchor constraintEqualToAnchor:contentView.trailingAnchor constant:-8.0],
        [sidePanel.topAnchor constraintEqualToAnchor:contentView.topAnchor constant:8.0],
        [sidePanel.bottomAnchor constraintEqualToAnchor:bar.topAnchor constant:-8.0],
        [self.hoverLabel.leadingAnchor constraintEqualToAnchor:sidePanel.leadingAnchor constant:8.0],
        [self.hoverLabel.trailingAnchor constraintEqualToAnchor:sidePanel.trailingAnchor constant:-8.0],
        [self.hoverLabel.topAnchor constraintEqualToAnchor:sidePanel.topAnchor constant:8.0]
    ]];

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

    [self updateRuleUI];
    [self updateZoomLabel];
    [self fitView];
    [self.window makeKeyAndOrderFront:nil];
    [self.window makeFirstResponder:self.mtkView];

    // Let the bar size itself, then refuse to shrink below its full width so no
    // control is ever clipped.
    [bar layoutSubtreeIfNeeded];
    self.window.minSize = NSMakeSize([bar fittingSize].width + 16.0, 400.0);
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

- (void)renderCellPlane:(uint32_t)plane cb:(id<MTLCommandBuffer>)cb {
    MTLRenderPassDescriptor *rp;
    id<MTLRenderCommandEncoder> enc;
    MTLViewport vp;
    Uniforms *u;

    if (cb == nil || self.cellTex == nil || self.renderPipeline == nil ||
        self.gridBuf == nil || self.uniformsBuf == nil ||
        self.cellTex.width == 0 || self.cellTex.height == 0) {
        return;
    }
    u = (Uniforms *)[self.uniformsBuf contents];
    u->curOffset = plane * (uint32_t)self.planeCells;
    rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = self.cellTex;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    enc = [cb renderCommandEncoderWithDescriptor:rp];
    if (enc == nil) {
        return;
    }
    [enc setRenderPipelineState:self.renderPipeline];
    vp.originX = 0.0;
    vp.originY = 0.0;
    vp.width = (double)self.cellTex.width;
    vp.height = (double)self.cellTex.height;
    vp.znear = 0.0;
    vp.zfar = 1.0;
    [enc setViewport:vp];
    [enc setFragmentBuffer:self.gridBuf offset:0 atIndex:0];
    [enc setFragmentBuffer:self.uniformsBuf offset:0 atIndex:1];
    [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [enc endEncoding];
}

- (void)renderFrameToTexture:(id<MTLTexture>)target cb:(id<MTLCommandBuffer>)cb {
    MTLRenderPassDescriptor *rp;
    id<MTLRenderCommandEncoder> enc;
    MTLViewport vp;

    if (cb == nil || target == nil || self.scalePipeline == nil ||
        self.uniformsBuf == nil) {
        return;
    }
    rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = target;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    enc = [cb renderCommandEncoderWithDescriptor:rp];
    if (enc == nil) {
        return;
    }
    [enc setRenderPipelineState:self.scalePipeline];
    vp.originX = 0.0;
    vp.originY = 0.0;
    vp.width = (double)target.width;
    vp.height = (double)target.height;
    vp.znear = 0.0;
    vp.zfar = 1.0;
    [enc setViewport:vp];
    [enc setFragmentBuffer:self.uniformsBuf offset:0 atIndex:0];
    if (self.cellTex != nil) {
        [enc setFragmentTexture:self.cellTex atIndex:0];
    }
    if (self.trailTex != nil) {
        [enc setFragmentTexture:self.trailTex atIndex:1];
    }
    [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [enc endEncoding];
}

- (BOOL)writePNGFromTexture:(id<MTLTexture>)tex path:(NSString *)path {
    NSUInteger w;
    NSUInteger h;
    NSUInteger rowBytes;
    NSUInteger totalBytes;
    void *bytes;
    CGDataProviderRef provider;
    CGColorSpaceRef cs;
    CGImageRef img;
    NSURL *url;
    CFURLRef cfurl;
    CGImageDestinationRef dest;
    CFIndex len;
    BOOL ok;

    if (tex == nil || path == nil || tex.width == 0 || tex.height == 0) {
        return NO;
    }
    w = tex.width;
    h = tex.height;
    rowBytes = w * 4;
    totalBytes = rowBytes * h;
    bytes = malloc(totalBytes);
    if (bytes == NULL) {
        return NO;
    }
    [tex getBytes:bytes
       bytesPerRow:rowBytes
        fromRegion:MTLRegionMake2D(0, 0, w, h)
        mipmapLevel:0];

    // The texture is BGRA8Unorm (memory byte order B,G,R,A). Describe it to
    // Core Graphics as little-endian 32-bit with the alpha byte first: a
    // big-endian [A,R,G,B] word reversed by 32Little yields [B,G,R,A], matching
    // the buffer. The render is fully opaque, so premultiplied == straight.
    provider = CGDataProviderCreateWithData(NULL, bytes, totalBytes, NULL);
    if (provider == NULL) {
        free(bytes);
        return NO;
    }
    cs = CGColorSpaceCreateDeviceRGB();
    img = CGImageCreate(w, h, 8, 32, rowBytes, cs,
                        (CGBitmapInfo)((CGBitmapInfo)kCGImageAlphaPremultipliedFirst |
                                       (CGBitmapInfo)kCGBitmapByteOrder32Little),
                        provider, NULL, false, kCGRenderingIntentDefault);
    CGColorSpaceRelease(cs);
    if (img == NULL) {
        CGDataProviderRelease(provider);
        free(bytes);
        return NO;
    }

    url = [NSURL fileURLWithPath:path];
    len = (CFIndex)strlen(url.fileSystemRepresentation);
    cfurl = CFURLCreateFromFileSystemRepresentation(kCFAllocatorDefault,
                                                    (const UInt8 *)url.fileSystemRepresentation,
                                                    len, false);
    if (cfurl == NULL) {
        CGImageRelease(img);
        CGDataProviderRelease(provider);
        free(bytes);
        return NO;
    }
    dest = CGImageDestinationCreateWithURL(cfurl, CFSTR("public.png"), 1, NULL);
    CFRelease(cfurl);
    if (dest == NULL) {
        CGImageRelease(img);
        CGDataProviderRelease(provider);
        free(bytes);
        return NO;
    }
    CGImageDestinationAddImage(dest, img, NULL);
    ok = CGImageDestinationFinalize(dest);
    CFRelease(dest);
    CGImageRelease(img);
    CGDataProviderRelease(provider);
    free(bytes);
    return ok;
}

- (void)advanceGenerations:(int)n {
    GOLRules r;
    Uniforms *u;
    uint32_t curPlane;
    uint32_t readPlane;
    uint32_t wp;
    int i;
    int stepIdx;
    uint16_t *cur;
    uint16_t *write;
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    NSUInteger tgW;
    NSUInteger tgH;
    NSUInteger gx;
    NSUInteger gy;

    if (n <= 0) {
        return;
    }
    r = self.rules;
    curPlane = (uint32_t)(self.frameIndex % (uint32_t)PLANE_COUNT);
    u = (Uniforms *)[self.uniformsBuf contents];
    u->gridW = (uint32_t)self.gridW;
    u->gridH = (uint32_t)self.gridH;
    u->pad = 0;
    u->birth = r.birth;
    u->survival = r.survival;

    if (self.displayMode != DISPLAY_TRAILS) {
        for (i = 1; i <= n; i++) {
            readPlane = (curPlane + (uint32_t)(i - 1)) % (uint32_t)PLANE_COUNT;
            wp = (curPlane + (uint32_t)i) % (uint32_t)PLANE_COUNT;
            cur = [self planePointer:readPlane];
            write = [self planePointer:wp];
            if (cur != nil && write != nil) {
                gol_step_cpu(cur, write, self.gridW, self.gridH, r);
            }
        }
        self.frameIndex = self.frameIndex + (uint64_t)n;
        self.displayPlane = (curPlane + (uint32_t)n) % (uint32_t)PLANE_COUNT;
        self.gen = self.gen + (uint32_t)n;
        return;
    }

    // Trails: the intensity is a visual accumulation, so each generation must
    // step the grid, render it into cellTex, and fold that into trailTex.
    [self clearTrailNow];
    cb = [self.queue commandBuffer];
    if (cb == nil) {
        for (i = 1; i <= n; i++) {
            readPlane = (curPlane + (uint32_t)(i - 1)) % (uint32_t)PLANE_COUNT;
            wp = (curPlane + (uint32_t)i) % (uint32_t)PLANE_COUNT;
            cur = [self planePointer:readPlane];
            write = [self planePointer:wp];
            if (cur != nil && write != nil) {
                gol_step_cpu(cur, write, self.gridW, self.gridH, r);
            }
        }
        self.frameIndex = self.frameIndex + (uint64_t)n;
        self.displayPlane = (curPlane + (uint32_t)n) % (uint32_t)PLANE_COUNT;
        self.gen = self.gen + (uint32_t)n;
        return;
    }

    tgW = 16;
    tgH = 16;
    gx = ((NSUInteger)self.gridW + tgW - 1) / tgW;
    gy = ((NSUInteger)self.gridH + tgH - 1) / tgH;
    for (stepIdx = 1; stepIdx <= n; stepIdx++) {
        readPlane = (curPlane + (uint32_t)(stepIdx - 1)) % (uint32_t)PLANE_COUNT;
        wp = (curPlane + (uint32_t)stepIdx) % (uint32_t)PLANE_COUNT;

        if (self.stepPipeline != nil) {
            enc = [cb computeCommandEncoder];
            if (enc != nil) {
                [enc setComputePipelineState:self.stepPipeline];
                [enc setBuffer:self.gridBuf offset:(NSUInteger)readPlane * self.planeBytes atIndex:0];
                [enc setBuffer:self.gridBuf offset:(NSUInteger)wp * self.planeBytes atIndex:1];
                [enc setBuffer:self.uniformsBuf offset:0 atIndex:2];
                [enc setBuffer:self.statsBuf offset:(NSUInteger)wp * sizeof(GOLStats) atIndex:3];
                [enc dispatchThreadgroups:MakeSize((int)gx, (int)gy, 1)
                  threadsPerThreadgroup:MakeSize((int)tgW, (int)tgH, 1)];
                [enc endEncoding];
            }
        } else {
            cur = [self planePointer:readPlane];
            write = [self planePointer:wp];
            if (cur != nil && write != nil) {
                gol_step_cpu(cur, write, self.gridW, self.gridH, r);
            }
        }

        [self renderCellPlane:wp cb:cb];

        if (self.trailStepPipeline != nil && self.trailTex != nil &&
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
    }
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error != nil) {
        NSLog(@"screenshot: trail advance error: %@", cb.error);
    }
    self.frameIndex = self.frameIndex + (uint64_t)n;
    self.displayPlane = (curPlane + (uint32_t)n) % (uint32_t)PLANE_COUNT;
    self.gen = self.gen + (uint32_t)n;
}

- (void)runScreenshot {
    int S;
    int G;
    GOLRules r;
    Uniforms *u;
    id<MTLCommandBuffer> cb;
    id<MTLBlitCommandEncoder> blit;
    MTLTextureDescriptor *d;
    id<MTLTexture> outTex;
    uint16_t *cells;
    int minCol, maxCol, minRow, maxRow;
    int x, y;
    int bboxW, bboxH, pad;
    double centerCol, centerRow, regionW, regionH, maxDim, fit;

    if (self.device == nil || self.queue == nil ||
        self.renderPipeline == nil || self.scalePipeline == nil ||
        self.uniformsBuf == nil || self.gridBuf == nil) {
        NSLog(@"screenshot: Metal not ready");
        [NSApp terminate:nil];
        return;
    }

    S = self.screenshotSize;
    if (S < 16) {
        S = 16;
    }
    if (S > MAX_TEXTURE_SIZE) {
        S = MAX_TEXTURE_SIZE;
    }

    // Render the grid at a quarter of the output size so each cell spans four
    // pixels; fs_scale then uses its point-read path for clean rounded cells.
    G = S / 4;
    if (G < 8) {
        G = 8;
    }
    if (G > self.maxGridW) {
        G = self.maxGridW;
    }
    if (G > self.maxGridH) {
        G = self.maxGridH;
    }
    while ((long long)G * (long long)G > (long long)self.planeCells && G > 8) {
        G--;
    }

    self.gridW = G;
    self.gridH = G;
    [self rebuildCellTexture];

    if (self.launchPreset != nil && self.launchPreset.length > 0) {
        [self applyPreset:[self.launchPreset UTF8String]];
    } else {
        [self randomize];
    }

    [self advanceGenerations:self.screenshotGen];

    r = self.rules;
    u = (Uniforms *)[self.uniformsBuf contents];
    u->gridW = (uint32_t)self.gridW;
    u->gridH = (uint32_t)self.gridH;
    u->pad = 0;
    u->birth = r.birth;
    u->survival = r.survival;
    u->viewWidth = (float)S;
    u->viewHeight = (float)S;
    u->displayMode = self.displayMode;
    u->palette = self.palette;
    u->glow = self.glowOn ? 1.0f : 0.0f;
    u->cursorX = -1e9f;
    u->cursorY = -1e9f;

    // The pattern usually occupies a small central region of the grid, so map
    // the whole grid and it would render tiny. Instead, auto-fit the view to
    // the live-cell bounding box (with a margin) so the pattern fills the
    // square canvas, centered and aspect-preserving.
    minCol = INT_MAX; maxCol = -1; minRow = INT_MAX; maxRow = -1;
    cells = [self planePointer:self.displayPlane];
    if (cells != nil) {
        for (y = 0; y < (int)self.gridH; y++) {
            for (x = 0; x < (int)self.gridW; x++) {
                if ((cells[(size_t)y * (size_t)self.gridW + (size_t)x] & 1u) != 0u) {
                    if (x < minCol) minCol = x;
                    if (x > maxCol) maxCol = x;
                    if (y < minRow) minRow = y;
                    if (y > maxRow) maxRow = y;
                }
            }
        }
    }
    if (maxCol >= minCol && maxRow >= minRow) {
        bboxW = maxCol - minCol + 1;
        bboxH = maxRow - minRow + 1;
        pad = (int)(0.18 * (double)(bboxW > bboxH ? bboxW : bboxH)) + 4;
        centerCol = ((double)minCol + (double)maxCol + 1.0) * 0.5;
        centerRow = ((double)minRow + (double)maxRow + 1.0) * 0.5;
        regionW = (double)bboxW + 2.0 * (double)pad;
        regionH = (double)bboxH + 2.0 * (double)pad;
        maxDim = regionW > regionH ? regionW : regionH;
        fit = maxDim / (double)S;
        u->viewScaleX = (float)fit;
        u->viewScaleY = (float)fit;
        u->viewOffsetX = (float)((double)S * 0.5 - centerCol / fit);
        u->viewOffsetY = (float)((double)S * 0.5 - centerRow / fit);
    } else {
        u->viewScaleX = (float)self.gridW / (float)S;
        u->viewScaleY = (float)self.gridH / (float)S;
        u->viewOffsetX = 0.0f;
        u->viewOffsetY = 0.0f;
    }

    d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                           width:(NSUInteger)S
                                                          height:(NSUInteger)S
                                                      mipmapped:NO];
    d.usage = (MTLTextureUsage)(MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
    d.storageMode = MTLStorageModeShared;
    outTex = [self.device newTextureWithDescriptor:d];
    if (outTex == nil) {
        NSLog(@"screenshot: failed to create output texture");
        [NSApp terminate:nil];
        return;
    }

    cb = [self.queue commandBuffer];
    if (cb == nil) {
        NSLog(@"screenshot: no command buffer");
        [NSApp terminate:nil];
        return;
    }
    [self renderCellPlane:self.displayPlane cb:cb];
    if (u->viewScaleX > 1.0f || u->viewScaleY > 1.0f) {
        blit = [cb blitCommandEncoder];
        if (blit != nil) {
            [blit generateMipmapsForTexture:self.cellTex];
            [blit endEncoding];
        }
    }
    [self renderFrameToTexture:outTex cb:cb];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error != nil) {
        NSLog(@"screenshot: render error: %@", cb.error);
    }

    if (![self writePNGFromTexture:outTex path:self.screenshotPath]) {
        NSLog(@"screenshot: PNG write to %@ failed", self.screenshotPath);
    } else {
        NSLog(@"screenshot: wrote %dx%d PNG to %@", S, S, self.screenshotPath);
    }
    [NSApp terminate:nil];
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
    int gensToAdvance = 0;
    id<MTLCommandBuffer> oldest;
    BOOL oldestDone;
    uint32_t planeA;
    uint32_t planeB;
    uint32_t readPlane;
    uint32_t wp;
    uint32_t plane;
    int stepIdx;
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
    id<MTLBlitCommandEncoder> blit;
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

    // Apply any queued paint strokes once per frame (no-op when the queue is empty).
    [self applyPaintQueue];

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
    u->cursorX = self.cursorInside ? (float)self.cursorGridX : -1e9f;
    u->cursorY = self.cursorInside ? (float)self.cursorGridY : -1e9f;
    u->brushR = (float)self.brushRadius + 0.5f;

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
    }

    if (self.stepOnce) {
        gensToAdvance = 1;
        self.stepOnce = NO;
    }

    if (gensToAdvance > 0 && self.stepPipeline != nil) {
        // Each dispatch advances the data by one plane (cur -> A -> B ->
        // cur ...), so after N steps the latest generation sits on
        // (curPlane + N) % 3, keeping displayPlane == frameIndex % 3.
        planeA = (curPlane + 1) % (uint32_t)PLANE_COUNT;
        planeB = (curPlane + 2) % (uint32_t)PLANE_COUNT;
        // The buffer touches every plane, so all three ring slots must be
        // done before we may reuse them.
        oldest = [self cbAt:slot];
        oldestDone = (oldest == nil) ||
                      oldest.status == MTLCommandBufferStatusCompleted ||
                      oldest.status == MTLCommandBufferStatusError;
        if (oldestDone) {
            oldest = [self cbAt:planeA];
            oldestDone = (oldest == nil) ||
                          oldest.status == MTLCommandBufferStatusCompleted ||
                          oldest.status == MTLCommandBufferStatusError;
        }
        if (oldestDone) {
            oldest = [self cbAt:planeB];
            oldestDone = (oldest == nil) ||
                          oldest.status == MTLCommandBufferStatusCompleted ||
                          oldest.status == MTLCommandBufferStatusError;
        }
        if (oldestDone) {
            s = (GOLStats *)[self.statsBuf contents];
            for (plane = 0; plane < (uint32_t)PLANE_COUNT; plane++) {
                memset(&s[plane], 0, sizeof(GOLStats));
            }
            tgW = 16;
            tgH = 16;
            gx = ((NSUInteger)self.gridW + tgW - 1) / tgW;
            gy = ((NSUInteger)self.gridH + tgH - 1) / tgH;
            enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:self.stepPipeline];
            for (stepIdx = 1; stepIdx <= gensToAdvance; stepIdx++) {
                readPlane = (curPlane + (uint32_t)(stepIdx - 1)) %
                            (uint32_t)PLANE_COUNT;
                wp = (curPlane + (uint32_t)stepIdx) % (uint32_t)PLANE_COUNT;
                [enc setBuffer:self.gridBuf
                       offset:(NSUInteger)readPlane * self.planeBytes
                      atIndex:0];
                [enc setBuffer:self.gridBuf
                       offset:(NSUInteger)wp * self.planeBytes
                      atIndex:1];
                [enc setBuffer:self.uniformsBuf offset:0 atIndex:2];
                [enc setBuffer:self.statsBuf
                       offset:wp * sizeof(GOLStats)
                      atIndex:3];
                [enc dispatchThreadgroups:MakeSize((int)gx, (int)gy, 1)
                  threadsPerThreadgroup:MakeSize((int)tgW, (int)tgH, 1)];
                if (stepIdx < gensToAdvance) {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            }
            [enc endEncoding];
            writePlane = (curPlane + (uint32_t)gensToAdvance) %
                         (uint32_t)PLANE_COUNT;
            renderPlane = writePlane;
            self.displayPlane = writePlane;
            self.completedPlane = curPlane;
            willStep = YES;
            [self setCB:cb at:writePlane];
        }
    } else if (gensToAdvance > 0) {
        for (stepIdx = 1; stepIdx <= gensToAdvance; stepIdx++) {
            readPlane = (curPlane + (uint32_t)(stepIdx - 1)) %
                        (uint32_t)PLANE_COUNT;
            wp = (curPlane + (uint32_t)stepIdx) % (uint32_t)PLANE_COUNT;
            cur = [self planePointer:readPlane];
            write = [self planePointer:wp];
            if (cur != nil && write != nil) {
                gol_step_cpu(cur, write, self.gridW, self.gridH, r);
            }
        }
        writePlane = (curPlane + (uint32_t)gensToAdvance) %
                     (uint32_t)PLANE_COUNT;
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
           forGeneration:self.gen + (uint32_t)gensToAdvance];
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
        if (u->viewScaleX > 1.0f || u->viewScaleY > 1.0f) {
            blit = [cb blitCommandEncoder];
            if (blit != nil) {
                [blit generateMipmapsForTexture:self.cellTex];
                [blit endEncoding];
            }
        }
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
        // The last dispatch wrote the stats slot of writePlane; read it back
        // once the buffer completes and apply it on the main queue.
        stepGen = self.gen + (uint32_t)gensToAdvance;
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
        self.frameIndex = f + (uint64_t)gensToAdvance;
        self.gen = self.gen + (uint32_t)gensToAdvance;
    }
    if (needCell && self.cellTex != nil) {
        self.dirty = NO;
    }
    self.lastCB = cb;

    self.fpsFrames = self.fpsFrames + 1u;
    if (now - self.fpsWindowStart >= 1.0) {
        fpsDt = now - self.fpsWindowStart;
        if (fpsDt > 0.0) {
            fps = LroundInt((double)self.fpsFrames / fpsDt);
            self.fpsValue = fps;
            self.statsDirty = YES;
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
    if (self.statsTimer != nil) {
        [self.statsTimer invalidate];
        self.statsTimer = nil;
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
