#include "gol_grid.h"

#include <stdlib.h>
#include <string.h>

GOLGrid *gol_grid_alloc(void) {
    GOLGrid *grid = (GOLGrid *)malloc(sizeof(*grid));
    if (grid != NULL) {
        memset(grid, 0, sizeof(*grid));
    }
    return grid;
}

void gol_grid_free(GOLGrid *grid) {
    if (grid == NULL) {
        return;
    }
    if ((grid->flags & GOL_GRID_FLAG_OWNS_CELLS) != 0u && grid->cells != NULL) {
        free(grid->cells);
    }
    free(grid);
}

void gol_grid_wrap(GOLGrid *grid, int w, int h, size_t planeCells,
                   int planeCount, uint16_t *cells, void *gpuHandle) {
    if (grid == NULL) {
        return;
    }
    grid->w = w;
    grid->h = h;
    grid->planeCells = planeCells;
    grid->planeCount = planeCount;
    grid->cells = cells;
    grid->gpuHandle = gpuHandle;
    grid->flags = 0;
}

void gol_grid_set_size(GOLGrid *grid, int w, int h) {
    if (grid == NULL) {
        return;
    }
    grid->w = w;
    grid->h = h;
}

uint16_t *gol_grid_plane(const GOLGrid *grid, uint32_t plane) {
    if (grid == NULL || grid->cells == NULL || grid->planeCount <= 0 ||
        plane >= (uint32_t)grid->planeCount) {
        return NULL;
    }
    return grid->cells + (size_t)plane * grid->planeCells;
}

size_t gol_grid_plane_bytes(const GOLGrid *grid) {
    if (grid == NULL) {
        return 0;
    }
    return grid->planeCells * sizeof(uint16_t);
}
