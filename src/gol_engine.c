#include "gol_engine.h"

#include <stdlib.h>

void gol_engine_set_rules(GOLEngine *engine, GOLRules rules) {
    if (engine != NULL) {
        engine->rules = rules;
    }
}

bool gol_engine_step_single(GOLEngine *engine, void *commandBuffer,
                            uint32_t readPlane, uint32_t writePlane) {
    if (engine == NULL || engine->stepSingle == NULL) {
        return false;
    }
    return engine->stepSingle(engine, commandBuffer, readPlane, writePlane);
}

bool gol_engine_step_range(GOLEngine *engine, void *commandBuffer,
                           uint32_t startPlane, uint32_t count) {
    if (engine == NULL || engine->stepRange == NULL) {
        return false;
    }
    return engine->stepRange(engine, commandBuffer, startPlane, count);
}

void gol_engine_count(GOLEngine *engine, uint32_t plane,
                      uint32_t *alive, uint32_t *maxAge) {
    if (alive != NULL) {
        *alive = 0;
    }
    if (maxAge != NULL) {
        *maxAge = 0;
    }
    if (engine == NULL || engine->count == NULL) {
        return;
    }
    engine->count(engine, plane, alive, maxAge);
}

uint16_t *gol_engine_plane(GOLEngine *engine, uint32_t plane) {
    if (engine == NULL) {
        return NULL;
    }
    return gol_grid_plane(engine->grid, plane);
}

void gol_engine_destroy(GOLEngine *engine) {
    if (engine == NULL) {
        return;
    }
    if (engine->destroyImpl != NULL) {
        engine->destroyImpl(engine);
    }
    free(engine);
}
