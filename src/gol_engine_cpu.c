#include "gol_engine.h"

#include <stdlib.h>
#include <string.h>

static void cpu_destroy_impl(GOLEngine *engine) {
    (void)engine;
}

static bool cpu_step_single(GOLEngine *engine, void *commandBuffer,
                            uint32_t readPlane, uint32_t writePlane) {
    GOLGrid *grid;
    uint16_t *cur;
    uint16_t *write;

    (void)commandBuffer;
    if (engine == NULL) {
        return false;
    }
    grid = engine->grid;
    if (grid == NULL || grid->planeCount <= 0) {
        return false;
    }
    cur = gol_grid_plane(grid, readPlane);
    write = gol_grid_plane(grid, writePlane);
    if (cur == NULL || write == NULL) {
        return false;
    }
    gol_step_cpu(cur, write, grid->w, grid->h, engine->rules);
    return true;
}

static bool cpu_step_range(GOLEngine *engine, void *commandBuffer,
                           uint32_t startPlane, uint32_t count) {
    GOLGrid *grid;
    uint32_t planes;
    uint32_t i;

    (void)commandBuffer;
    if (engine == NULL) {
        return false;
    }
    grid = engine->grid;
    if (grid == NULL || grid->planeCount <= 0) {
        return false;
    }
    planes = (uint32_t)grid->planeCount;
    for (i = 0; i < count; i++) {
        uint32_t readPlane = (startPlane + i) % planes;
        uint32_t writePlane = (startPlane + i + 1u) % planes;
        if (!cpu_step_single(engine, NULL, readPlane, writePlane)) {
            return false;
        }
    }
    return true;
}

static void cpu_count(GOLEngine *engine, uint32_t plane,
                      uint32_t *alive, uint32_t *maxAge) {
    GOLGrid *grid;
    uint16_t *cells;
    int a;
    int m;

    if (engine == NULL) {
        return;
    }
    grid = engine->grid;
    if (grid == NULL) {
        return;
    }
    cells = gol_grid_plane(grid, plane);
    a = 0;
    m = 0;
    if (cells != NULL) {
        gol_count_alive(cells, grid->w, grid->h, &a, &m);
    }
    if (alive != NULL) {
        *alive = (uint32_t)a;
    }
    if (maxAge != NULL) {
        *maxAge = (uint32_t)m;
    }
}

GOLEngine *gol_engine_create_cpu(GOLGrid *grid) {
    GOLEngine *engine = (GOLEngine *)malloc(sizeof(*engine));
    if (engine == NULL) {
        return NULL;
    }
    memset(engine, 0, sizeof(*engine));
    engine->grid = grid;
    engine->rules = gol_default_rules();
    engine->impl = NULL;
    engine->flags = 0;
    engine->stepSingle = cpu_step_single;
    engine->stepRange = cpu_step_range;
    engine->count = cpu_count;
    engine->destroyImpl = cpu_destroy_impl;
    return engine;
}
