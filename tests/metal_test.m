#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gol.h"
#include "refstep.h"

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

static uint64_t rng_state = 0xfedcba9876543210ULL;

static uint64_t rng_next(void) {
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static bool random_alive(double density) {
    unsigned int sample = (unsigned int)(rng_next() % 1000u);
    unsigned int threshold = (unsigned int)(density * 1000.0 + 0.5);
    return sample < threshold;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        id<MTLDevice> device;
        NSString *libPath;
        NSError *error;
        id<MTLLibrary> library;
        id<MTLFunction> stepFunc;
        id<MTLComputePipelineState> pipeline;
        id<MTLCommandQueue> queue;
        GOLRules rules[] = {
            { (1u << 3), (1u << 2) | (1u << 3) },
            { (1u << 3) | (1u << 6), (1u << 2) | (1u << 3) },
            { (1u << 2), 0u },
            { (1u << 2) | (1u << 3) | (1u << 4), (1u << 1) | (1u << 2) | (1u << 3) },
            { 0u, (1u << 1) }
        };
        int widths[] = { 16, 32, 48 };
        int heights[] = { 16, 32, 31 };
        double densities[] = { 0.1, 0.4, 0.8 };
        size_t rule_count;
        size_t grid_count;
        size_t density_count;
        NSUInteger execThreads;
        NSUInteger tx;
        NSUInteger ty;

        printf("=== Metal Implementation Tests ===\n\n");

        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("Metal device unavailable; skipping Metal tests\n");
            return 0;
        }

        libPath = argc > 1 ? [NSString stringWithUTF8String:argv[1]] : @"bin/shaders.metallib";
        error = nil;
        library = [device newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&error];
        if (!library) {
            fprintf(stderr, "failed to load metallib %s: %s\n",
                    [libPath UTF8String],
                    error.localizedDescription.UTF8String);
            return 1;
        }

        stepFunc = [library newFunctionWithName:@"gol_step"];
        if (!stepFunc) {
            fprintf(stderr, "failed to find gol_step function\n");
            return 1;
        }

        pipeline = [device newComputePipelineStateWithFunction:stepFunc error:&error];
        if (!pipeline) {
            fprintf(stderr, "failed to create compute pipeline: %s\n",
                    error.localizedDescription.UTF8String);
            return 1;
        }

        queue = [device newCommandQueue];
        if (!queue) {
            fprintf(stderr, "failed to create command queue\n");
            return 1;
        }

        rule_count = sizeof(rules) / sizeof(rules[0]);
        grid_count = sizeof(widths) / sizeof(widths[0]);
        density_count = sizeof(densities) / sizeof(densities[0]);

        execThreads = [pipeline threadExecutionWidth];
        tx = GOL_MIN((NSUInteger)16, GOL_MAX((NSUInteger)1, execThreads));
        ty = GOL_MAX((NSUInteger)1, GOL_MIN((NSUInteger)16, execThreads / tx));

        for (size_t ri = 0; ri < rule_count; ri++) {
            for (size_t gi = 0; gi < grid_count; gi++) {
                for (size_t di = 0; di < density_count; di++) {
                    int w;
                    int h;
                    size_t n;
                    NSUInteger bytes;
                    id<MTLBuffer> curBuf;
                    id<MTLBuffer> nextBuf;
                    id<MTLBuffer> uniBuf;
                    uint16_t *cpuCur;
                    uint16_t *cpuNext;
                    RefGrid ref;
                    uint16_t *metalCur;
                    Uniforms u;

                    w = widths[gi];
                    h = heights[gi];
                    n = (size_t)w * (size_t)h;
                    bytes = (NSUInteger)(n * sizeof(uint16_t));

                    curBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                    nextBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                    uniBuf = [device newBufferWithLength:sizeof(Uniforms) options:MTLResourceStorageModeShared];
                    assert(curBuf != nil && nextBuf != nil && uniBuf != nil);

                    cpuCur = (uint16_t *)calloc(n, sizeof(uint16_t));
                    cpuNext = (uint16_t *)calloc(n, sizeof(uint16_t));
                    assert(cpuCur != NULL && cpuNext != NULL);

                    ref_grid_init(&ref, w, h, true);

                    metalCur = (uint16_t *)[curBuf contents];
                    for (size_t i = 0; i < n; i++) {
                        bool alive = random_alive(densities[di]);
                        metalCur[i] = alive ? 1u : 0u;
                        cpuCur[i] = alive ? 1u : 0u;
                        ref.a[i] = alive ? 1u : 0u;
                    }

                    memset(&u, 0, sizeof(u));
                    u.gridW = (uint32_t)w;
                    u.gridH = (uint32_t)h;
                    u.curOffset = 0;
                    u.birth = (uint8_t)rules[ri].birth;
                    u.survival = (uint8_t)rules[ri].survival;
                    memcpy([uniBuf contents], &u, sizeof(u));

                    for (int step = 0; step < 6; step++) {
                        id<MTLCommandBuffer> cb;
                        id<MTLComputeCommandEncoder> enc;
                        NSUInteger gx;
                        NSUInteger gy;
                        uint16_t *metalNext;
                        size_t mismatches;
                        int refAlive;
                        int metalAlive;
                        id<MTLBuffer> tmpBuf;
                        uint16_t *tmpCpu;

                        cb = [queue commandBuffer];
                        assert(cb != nil);
                        enc = [cb computeCommandEncoder];
                        assert(enc != nil);
                        [enc setComputePipelineState:pipeline];
                        [enc setBuffer:curBuf offset:0 atIndex:0];
                        [enc setBuffer:nextBuf offset:0 atIndex:1];
                        [enc setBuffer:uniBuf offset:0 atIndex:2];
                        gx = ((NSUInteger)w + tx - 1) / tx;
                        gy = ((NSUInteger)h + ty - 1) / ty;
                        [enc dispatchThreadgroups:MakeSize((int)gx, (int)gy, 1)
                          threadsPerThreadgroup:MakeSize((int)tx, (int)ty, 1)];
                        [enc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        if (cb.error) {
                            fprintf(stderr, "Metal command buffer error: %s\n",
                                    cb.error.localizedDescription.UTF8String);
                            return 1;
                        }

                        gol_step_cpu(cpuCur, cpuNext, w, h, rules[ri]);
                        ref_step(&ref, rules[ri]);

                        metalNext = (uint16_t *)[nextBuf contents];
                        mismatches = 0;
                        for (size_t i = 0; i < n; i++) {
                            if (metalNext[i] != cpuNext[i]) {
                                if (mismatches < 8) {
                                    printf("  MISMATCH rule=%zu grid=%dx%d density=%.2f step=%d cell=%zu metal=0x%04x cpu=0x%04x\n",
                                           ri, w, h, densities[di], step, i,
                                           metalNext[i], cpuNext[i]);
                                }
                                mismatches++;
                            }
                            refAlive = ref.a[i] ? 1 : 0;
                            metalAlive = GolAlive(metalNext[i]) ? 1 : 0;
                            if (refAlive != metalAlive) {
                                if (mismatches < 8) {
                                    printf("  REF MISMATCH rule=%zu grid=%dx%d density=%.2f step=%d cell=%zu ref=%d metal=%d\n",
                                           ri, w, h, densities[di], step, i, refAlive, metalAlive);
                                }
                                mismatches++;
                            }
                        }
                        if (mismatches > 0) {
                            fprintf(stderr, "Metal test failed with %zu mismatches\n", mismatches);
                            return 1;
                        }

                        tmpBuf = curBuf;
                        curBuf = nextBuf;
                        nextBuf = tmpBuf;
                        tmpCpu = cpuCur;
                        cpuCur = cpuNext;
                        cpuNext = tmpCpu;
                    }

                    free(cpuCur);
                    free(cpuNext);
                    ref_grid_free(&ref);
                }
            }
        }

        printf("All Metal tests passed!\n");
    }
    return 0;
}
