#ifndef GOL_ENGINE_H
#define GOL_ENGINE_H

#include "gol.h"
#include "gol_grid.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t alive;
    uint32_t maxAge;
    uint32_t pad0;
    uint32_t pad1;
} GOLStats;

typedef struct GOLEngine GOLEngine;

#define GOL_ENGINE_FLAG_GPU 1u

struct GOLEngine {
    GOLGrid *grid;
    void *impl;
    GOLRules rules;
    uint32_t flags;
    bool (*stepSingle)(GOLEngine *engine, void *commandBuffer,
                       uint32_t readPlane, uint32_t writePlane);
    bool (*stepRange)(GOLEngine *engine, void *commandBuffer,
                      uint32_t startPlane, uint32_t count);
    void (*count)(GOLEngine *engine, uint32_t plane,
                  uint32_t *alive, uint32_t *maxAge);
    void (*destroyImpl)(GOLEngine *engine);
};

GOLEngine *gol_engine_create_cpu(GOLGrid *grid);
GOLEngine *gol_engine_create_metal(GOLGrid *grid, void *stepPipeline,
                                   void *gridBuf, void *uniformsBuf,
                                   void *statsBuf);
void gol_engine_destroy(GOLEngine *engine);
void gol_engine_set_rules(GOLEngine *engine, GOLRules rules);
bool gol_engine_step_single(GOLEngine *engine, void *commandBuffer,
                            uint32_t readPlane, uint32_t writePlane);
bool gol_engine_step_range(GOLEngine *engine, void *commandBuffer,
                           uint32_t startPlane, uint32_t count);
void gol_engine_count(GOLEngine *engine, uint32_t plane,
                      uint32_t *alive, uint32_t *maxAge);
uint16_t *gol_engine_plane(GOLEngine *engine, uint32_t plane);

#endif
