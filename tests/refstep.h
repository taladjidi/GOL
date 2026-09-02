#ifndef GOL_REFSTEP_H
#define GOL_REFSTEP_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "gol.h"

typedef struct {
    int w;
    int h;
    bool torus;
    uint8_t pad[7];
    uint8_t *a;
    uint8_t *b;
} RefGrid;

static inline void ref_grid_init(RefGrid *g, int w, int h, bool torus) {
    size_t n = (size_t)w * (size_t)h;
    g->w = w;
    g->h = h;
    g->torus = torus;
    g->a = (uint8_t *)calloc(n, 1);
    g->b = (uint8_t *)calloc(n, 1);
}

static inline void ref_grid_free(RefGrid *g) {
    free(g->a);
    free(g->b);
    g->a = NULL;
    g->b = NULL;
    g->w = 0;
    g->h = 0;
    g->torus = false;
}

static inline void ref_clear(RefGrid *g) {
    size_t n = (size_t)g->w * (size_t)g->h;
    memset(g->a, 0, n);
    memset(g->b, 0, n);
}

static inline void ref_set(RefGrid *g, int x, int y, bool alive) {
    if (x < 0 || x >= g->w || y < 0 || y >= g->h) return;
    g->a[(size_t)y * (size_t)g->w + (size_t)x] = alive ? 1u : 0u;
}

static inline int ref_count(const RefGrid *g) {
    int count = 0;
    size_t n = (size_t)g->w * (size_t)g->h;
    for (size_t i = 0; i < n; i++) {
        if (g->a[i]) count++;
    }
    return count;
}

static inline void ref_copy(RefGrid *dst, const RefGrid *src) {
    if (dst->w != src->w || dst->h != src->h) return;
    memcpy(dst->a, src->a, (size_t)dst->w * (size_t)dst->h);
}

static inline bool ref_equal(const RefGrid *a, const RefGrid *b) {
    if (a->w != b->w || a->h != b->h) return false;
    return memcmp(a->a, b->a, (size_t)a->w * (size_t)a->h) == 0;
}

static inline void ref_step(RefGrid *g, GOLRules rules) {
    uint8_t *tmp;
    for (int y = 0; y < g->h; y++) {
        for (int x = 0; x < g->w; x++) {
            int neighbors = 0;
            int nx;
            int ny;
            size_t i;
            bool alive;
            bool next;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    nx = x + dx;
                    ny = y + dy;
                    if (g->torus) {
                        nx = (nx + g->w) % g->w;
                        ny = (ny + g->h) % g->h;
                    } else if (nx < 0 || nx >= g->w || ny < 0 || ny >= g->h) {
                        continue;
                    }
                    neighbors += g->a[(size_t)ny * (size_t)g->w + (size_t)nx];
                }
            }
            i = (size_t)y * (size_t)g->w + (size_t)x;
            alive = g->a[i] != 0u;
            next = gol_rule_birth(rules, neighbors) ||
                   (alive && gol_rule_survive(rules, neighbors));
            g->b[i] = next ? 1u : 0u;
        }
    }
    tmp = g->a;
    g->a = g->b;
    g->b = tmp;
}

static inline bool ref_equals_shifted(const RefGrid *cur, const RefGrid *src, int dx, int dy) {
    if (cur->w != src->w || cur->h != src->h) return false;
    for (int y = 0; y < cur->h; y++) {
        for (int x = 0; x < cur->w; x++) {
            bool left = cur->a[(size_t)y * (size_t)cur->w + (size_t)x] != 0u;
            int sx = x - dx;
            int sy = y - dy;
            bool right;
            if (src->torus) {
                sx = (sx + src->w) % src->w;
                sy = (sy + src->h) % src->h;
                right = src->a[(size_t)sy * (size_t)src->w + (size_t)sx] != 0u;
            } else {
                right = (sx >= 0 && sx < src->w && sy >= 0 && sy < src->h) &&
                        src->a[(size_t)sy * (size_t)src->w + (size_t)sx] != 0u;
            }
            if (left != right) return false;
        }
    }
    return true;
}

#endif
