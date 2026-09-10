#ifndef GOL_GRID_H
#define GOL_GRID_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define GOL_GRID_FLAG_OWNS_CELLS 1u

typedef struct GOLGrid {
    size_t planeCells;
    uint16_t *cells;
    void *gpuHandle;
    int w;
    int h;
    int planeCount;
    uint32_t flags;
} GOLGrid;

GOLGrid *gol_grid_alloc(void);
void gol_grid_free(GOLGrid *grid);
void gol_grid_wrap(GOLGrid *grid, int w, int h, size_t planeCells,
                   int planeCount, uint16_t *cells, void *gpuHandle);
void gol_grid_set_size(GOLGrid *grid, int w, int h);
uint16_t *gol_grid_plane(const GOLGrid *grid, uint32_t plane);
size_t gol_grid_plane_bytes(const GOLGrid *grid);

#endif
