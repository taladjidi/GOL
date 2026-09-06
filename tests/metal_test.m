#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <assert.h>
#include <math.h>
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
        int widths[] = { 16, 32, 48, 7 };
        int heights[] = { 16, 32, 31, 5 };
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
                    id<MTLBuffer> statsBuf;
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
                    statsBuf = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
                    assert(curBuf != nil && nextBuf != nil && uniBuf != nil && statsBuf != nil);

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
                    u.birth = rules[ri].birth;
                    u.survival = rules[ri].survival;
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
                        int cpuAlive;
                        int cpuMaxAge;
                        uint32_t *stats;
                        id<MTLBuffer> tmpBuf;
                        uint16_t *tmpCpu;

                        cb = [queue commandBuffer];
                        assert(cb != nil);
                        enc = [cb computeCommandEncoder];
                        assert(enc != nil);
                        [enc setComputePipelineState:pipeline];
                        memset([statsBuf contents], 0, 16);
                        [enc setBuffer:curBuf offset:0 atIndex:0];
                        [enc setBuffer:nextBuf offset:0 atIndex:1];
                        [enc setBuffer:uniBuf offset:0 atIndex:2];
                        [enc setBuffer:statsBuf offset:0 atIndex:3];
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

                        gol_count_alive(cpuNext, w, h, &cpuAlive, &cpuMaxAge);
                        stats = (uint32_t *)[statsBuf contents];
                        if (stats[0] != (uint32_t)cpuAlive || stats[1] != (uint32_t)cpuMaxAge) {
                            if (mismatches < 8) {
                                printf("  STATS MISMATCH rule=%zu grid=%dx%d density=%.2f step=%d gpu=%u/%u cpu=%d/%d\n",
                                       ri, w, h, densities[di], step, stats[0], stats[1], cpuAlive, cpuMaxAge);
                            }
                            mismatches++;
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

        {
            id<MTLFunction> trailStepFunc;
            id<MTLFunction> trailClearFunc;
            id<MTLComputePipelineState> trailStepPipeline;
            id<MTLComputePipelineState> trailClearPipeline;
            MTLTextureDescriptor *cellDesc;
            MTLTextureDescriptor *trailDesc;
            id<MTLTexture> cellTex;
            id<MTLTexture> trailTex;
            id<MTLBuffer> readBuf;
            NSUInteger tw;
            NSUInteger th;
            NSUInteger bytesPerRow;
            uint8_t *pixels;
            uint8_t *readback;
            int step;

            tw = 4;
            th = 4;
            bytesPerRow = (NSUInteger)(tw * 2);

            trailStepFunc = [library newFunctionWithName:@"trail_step"];
            if (!trailStepFunc) {
                fprintf(stderr, "failed to find trail_step function\n");
                return 1;
            }
            trailClearFunc = [library newFunctionWithName:@"trail_clear"];
            if (!trailClearFunc) {
                fprintf(stderr, "failed to find trail_clear function\n");
                return 1;
            }
            error = nil;
            trailStepPipeline = [device newComputePipelineStateWithFunction:trailStepFunc error:&error];
            if (!trailStepPipeline) {
                fprintf(stderr, "failed to create trail_step pipeline: %s\n",
                        error.localizedDescription.UTF8String);
                return 1;
            }
            error = nil;
            trailClearPipeline = [device newComputePipelineStateWithFunction:trailClearFunc error:&error];
            if (!trailClearPipeline) {
                fprintf(stderr, "failed to create trail_clear pipeline: %s\n",
                        error.localizedDescription.UTF8String);
                return 1;
            }

            cellDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                            width:tw
                                                                           height:th
                                                                        mipmapped:NO];
            cellDesc.usage = MTLTextureUsageShaderRead;
            cellDesc.storageMode = MTLStorageModeShared;
            cellTex = [device newTextureWithDescriptor:cellDesc];

            trailDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Unorm
                                                                             width:tw
                                                                            height:th
                                                                         mipmapped:NO];
            trailDesc.usage = (MTLTextureUsage)(MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
            trailDesc.storageMode = MTLStorageModePrivate;
            trailTex = [device newTextureWithDescriptor:trailDesc];

            readBuf = [device newBufferWithLength:bytesPerRow * th options:MTLResourceStorageModeShared];
            assert(cellTex != nil && trailTex != nil && readBuf != nil);

            pixels = (uint8_t *)calloc((size_t)tw * th, 4);
            assert(pixels != NULL);

            {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:trailClearPipeline];
                [enc setTexture:trailTex atIndex:0];
                [enc dispatchThreadgroups:MakeSize((int)tw, (int)th, 1)
                  threadsPerThreadgroup:MakeSize(1, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.error) {
                    fprintf(stderr, "trail_clear error: %s\n", cb.error.localizedDescription.UTF8String);
                    return 1;
                }
            }

            for (step = 1; step <= 2; step++) {
                id<MTLCommandBuffer> cb;
                id<MTLComputeCommandEncoder> enc;
                id<MTLBlitCommandEncoder> blit;
                uint16_t raw;
                float value;
                float expected;
                float tol;

                memset(pixels, 0, (size_t)tw * th * 4);
                if (step == 1) {
                    pixels[0] = 0;
                    pixels[1] = 0;
                    pixels[2] = 255;
                    pixels[3] = 255;
                }
                [cellTex replaceRegion:MTLRegionMake2D(0, 0, tw, th)
                          mipmapLevel:0
                           withBytes:pixels
                       bytesPerRow:(tw * 4)];

                cb = [queue commandBuffer];
                assert(cb != nil);
                enc = [cb computeCommandEncoder];
                assert(enc != nil);
                [enc setComputePipelineState:trailStepPipeline];
                [enc setTexture:cellTex atIndex:0];
                [enc setTexture:trailTex atIndex:1];
                [enc setTexture:trailTex atIndex:2];
                [enc dispatchThreadgroups:MakeSize((int)tw, (int)th, 1)
                  threadsPerThreadgroup:MakeSize(1, 1, 1)];
                [enc endEncoding];

                blit = [cb blitCommandEncoder];
                assert(blit != nil);
                [blit copyFromTexture:trailTex
                         sourceSlice:0
                        sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                     sourceSize:MTLSizeMake(tw, th, 1)
                         toBuffer:readBuf
                destinationOffset:0
         destinationBytesPerRow:bytesPerRow
       destinationBytesPerImage:bytesPerRow * th];
                [blit endEncoding];

                [cb commit];
                [cb waitUntilCompleted];
                if (cb.error) {
                    fprintf(stderr, "trail_step error: %s\n", cb.error.localizedDescription.UTF8String);
                    return 1;
                }

                readback = (uint8_t *)[readBuf contents];
                raw = (uint16_t)(readback[0] | (readback[1] << 8));
                value = (float)raw / 65535.0f;
                expected = (step == 1) ? 1.0f : 0.94f;
                tol = 2.0f / 65535.0f;
                if (fabsf(value - expected) > tol) {
                    fprintf(stderr, "trail_step mismatch step=%d value=%.6f expected=%.6f\n",
                            step, (double)value, (double)expected);
                    return 1;
                }
            }

            free(pixels);
        }

        printf("All Metal tests passed!\n");
    }
    return 0;
}
