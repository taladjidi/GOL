#import <Metal/Metal.h>

#include "gol_engine.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    __unsafe_unretained id<MTLComputePipelineState> pipeline;
    __unsafe_unretained id<MTLBuffer> gridBuf;
    __unsafe_unretained id<MTLBuffer> uniformsBuf;
    __unsafe_unretained id<MTLBuffer> statsBuf;
} GOLMetalEngineImpl;

static MTLSize MetalMakeSize(NSUInteger w, NSUInteger h) {
    MTLSize s;
    s.width = w;
    s.height = h;
    s.depth = 1;
    return s;
}

static bool MetalValid(GOLEngine *engine, GOLMetalEngineImpl *impl) {
    if (engine == NULL || impl == NULL || engine->grid == NULL) {
        return false;
    }
    if (engine->grid->planeCount <= 0) {
        return false;
    }
    if (impl->pipeline == nil || impl->gridBuf == nil ||
        impl->uniformsBuf == nil || impl->statsBuf == nil) {
        return false;
    }
    return true;
}

static bool metal_step_single(GOLEngine *engine, void *commandBuffer,
                              uint32_t readPlane, uint32_t writePlane) {
    GOLMetalEngineImpl *impl;
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    GOLGrid *grid;
    NSUInteger tgW;
    NSUInteger tgH;
    NSUInteger gx;
    NSUInteger gy;

    if (engine == NULL) {
        return false;
    }
    impl = (GOLMetalEngineImpl *)engine->impl;
    if (!MetalValid(engine, impl)) {
        return false;
    }
    cb = (__bridge id<MTLCommandBuffer>)commandBuffer;
    if (cb == nil) {
        return false;
    }
    grid = engine->grid;
    tgW = 16;
    tgH = 16;
    gx = ((NSUInteger)grid->w + tgW - 1u) / tgW;
    gy = ((NSUInteger)grid->h + tgH - 1u) / tgH;
    if (gx == 0 || gy == 0) {
        return false;
    }

    enc = [cb computeCommandEncoder];
    if (enc == nil) {
        return false;
    }
    [enc setComputePipelineState:impl->pipeline];
    [enc setBuffer:impl->gridBuf
           offset:(NSUInteger)readPlane * (NSUInteger)gol_grid_plane_bytes(grid)
          atIndex:0];
    [enc setBuffer:impl->gridBuf
           offset:(NSUInteger)writePlane * (NSUInteger)gol_grid_plane_bytes(grid)
          atIndex:1];
    [enc setBuffer:impl->uniformsBuf offset:0 atIndex:2];
    [enc setBuffer:impl->statsBuf
           offset:(NSUInteger)writePlane * (NSUInteger)sizeof(GOLStats)
          atIndex:3];
    [enc dispatchThreadgroups:MetalMakeSize(gx, gy)
      threadsPerThreadgroup:MetalMakeSize(tgW, tgH)];
    [enc endEncoding];
    return true;
}

static bool metal_step_range(GOLEngine *engine, void *commandBuffer,
                             uint32_t startPlane, uint32_t count) {
    GOLMetalEngineImpl *impl;
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    GOLGrid *grid;
    GOLStats *stats;
    NSUInteger tgW;
    NSUInteger tgH;
    NSUInteger gx;
    NSUInteger gy;
    uint32_t planes;
    uint32_t i;
    uint32_t p;

    if (engine == NULL) {
        return false;
    }
    impl = (GOLMetalEngineImpl *)engine->impl;
    if (!MetalValid(engine, impl)) {
        return false;
    }
    cb = (__bridge id<MTLCommandBuffer>)commandBuffer;
    if (cb == nil) {
        return false;
    }
    if (count == 0) {
        return true;
    }

    grid = engine->grid;
    planes = (uint32_t)grid->planeCount;
    stats = (GOLStats *)[impl->statsBuf contents];
    if (stats != NULL) {
        for (p = 0; p < planes; p++) {
            memset(&stats[p], 0, sizeof(GOLStats));
        }
    }

    tgW = 16;
    tgH = 16;
    gx = ((NSUInteger)grid->w + tgW - 1u) / tgW;
    gy = ((NSUInteger)grid->h + tgH - 1u) / tgH;
    if (gx == 0 || gy == 0) {
        return false;
    }

    enc = [cb computeCommandEncoder];
    if (enc == nil) {
        return false;
    }
    [enc setComputePipelineState:impl->pipeline];
    for (i = 0; i < count; i++) {
        uint32_t readPlane = (startPlane + i) % planes;
        uint32_t writePlane = (startPlane + i + 1u) % planes;
        [enc setBuffer:impl->gridBuf
               offset:(NSUInteger)readPlane * (NSUInteger)gol_grid_plane_bytes(grid)
              atIndex:0];
        [enc setBuffer:impl->gridBuf
               offset:(NSUInteger)writePlane * (NSUInteger)gol_grid_plane_bytes(grid)
              atIndex:1];
        [enc setBuffer:impl->uniformsBuf offset:0 atIndex:2];
        [enc setBuffer:impl->statsBuf
               offset:(NSUInteger)writePlane * (NSUInteger)sizeof(GOLStats)
              atIndex:3];
        [enc dispatchThreadgroups:MetalMakeSize(gx, gy)
          threadsPerThreadgroup:MetalMakeSize(tgW, tgH)];
        if (i + 1u < count) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
    }
    [enc endEncoding];
    return true;
}

static void metal_count(GOLEngine *engine, uint32_t plane,
                        uint32_t *alive, uint32_t *maxAge) {
    GOLMetalEngineImpl *impl;
    GOLGrid *grid;
    GOLStats *stats;
    uint32_t a;
    uint32_t m;

    a = 0;
    m = 0;
    if (engine != NULL) {
        impl = (GOLMetalEngineImpl *)engine->impl;
        grid = engine->grid;
        if (impl != NULL && impl->statsBuf != nil && grid != NULL &&
            grid->planeCount > 0 && plane < (uint32_t)grid->planeCount) {
            stats = (GOLStats *)[impl->statsBuf contents];
            if (stats != NULL) {
                a = stats[plane].alive;
                m = stats[plane].maxAge;
            }
        }
    }
    if (alive != NULL) {
        *alive = a;
    }
    if (maxAge != NULL) {
        *maxAge = m;
    }
}

static void metal_destroy_impl(GOLEngine *engine) {
    if (engine != NULL && engine->impl != NULL) {
        free(engine->impl);
        engine->impl = NULL;
    }
}

GOLEngine *gol_engine_create_metal(GOLGrid *grid, void *stepPipeline,
                                   void *gridBuf, void *uniformsBuf,
                                   void *statsBuf) {
    GOLEngine *engine = (GOLEngine *)malloc(sizeof(*engine));
    GOLMetalEngineImpl *impl;

    if (engine == NULL) {
        return NULL;
    }
    memset(engine, 0, sizeof(*engine));

    impl = (GOLMetalEngineImpl *)malloc(sizeof(*impl));
    if (impl == NULL) {
        free(engine);
        return NULL;
    }
    memset(impl, 0, sizeof(*impl));
    impl->pipeline = (__bridge id<MTLComputePipelineState>)stepPipeline;
    impl->gridBuf = (__bridge id<MTLBuffer>)gridBuf;
    impl->uniformsBuf = (__bridge id<MTLBuffer>)uniformsBuf;
    impl->statsBuf = (__bridge id<MTLBuffer>)statsBuf;

    engine->grid = grid;
    engine->rules = gol_default_rules();
    engine->impl = impl;
    engine->flags = (impl->pipeline != nil) ? GOL_ENGINE_FLAG_GPU : 0u;
    engine->stepSingle = metal_step_single;
    engine->stepRange = metal_step_range;
    engine->count = metal_count;
    engine->destroyImpl = metal_destroy_impl;
    return engine;
}
