#include "gol.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline uint16_t GolAge(uint16_t v) {
    return (v >> 1) & 0x7FFFu;
}

static inline bool GolAlive(uint16_t v) {
    return (v & 1u) != 0u;
}

GOLRules gol_default_rules(void) {
    GOLRules r = { 0 };
    r.birth = (1u << 3);           // B3
    r.survival = (1u << 2) | (1u << 3); // S23
    return r;
}

bool gol_rule_birth(GOLRules r, int neighbors) {
    if (neighbors < 0 || neighbors > 8) return false;
    return (r.birth >> neighbors) & 1u;
}

bool gol_rule_survive(GOLRules r, int neighbors) {
    if (neighbors < 0 || neighbors > 8) return false;
    return (r.survival >> neighbors) & 1u;
}

void gol_randomize(uint16_t *cells, size_t planeCells, int w, int h, double density) {
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    memset(cells, 0, planeCells * sizeof(uint16_t));
    size_t n = (size_t)w * (size_t)h;
    for (size_t i = 0; i < n; i++) {
        if ((double)rand() / ((double)RAND_MAX + 1.0) < density) {
            cells[i] = (uint16_t)1u;
        }
    }
}

void gol_set_plane(uint16_t *cells, int w, int h, int x, int y, bool alive) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    size_t i = (size_t)y * (size_t)w + (size_t)x;
    uint16_t v = cells[i];
    uint16_t age = GolAge(v);
    if (alive) {
        if (!GolAlive(v) || age == 0u) {
            age = 1u;
        }
        cells[i] = (uint16_t)(1u | (age << 1));
    } else {
        cells[i] = 0u;
    }
}

void gol_step_cpu(const uint16_t *cur, uint16_t *next, int w, int h, GOLRules rules) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int n = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = (x + dx + w) % w;
                    int ny = (y + dy + h) % h;
                    n += GolAlive(cur[(size_t)ny * (size_t)w + (size_t)nx]) ? 1 : 0;
                }
            }
            size_t i = (size_t)y * (size_t)w + (size_t)x;
            uint16_t v = cur[i];
            bool alive = GolAlive(v);
            uint16_t age = GolAge(v);
            if (gol_rule_birth(rules, n) || (alive && gol_rule_survive(rules, n))) {
                uint16_t newAge = alive ? (age >= 32767u ? 32767u : age + 1u) : 1u;
                next[i] = (uint16_t)(1u | (newAge << 1));
            } else {
                next[i] = 0u;
            }
        }
    }
}

void gol_count_alive(const uint16_t *cells, int w, int h, int *alive, int *max_age) {
    int a = 0;
    int ma = 0;
    size_t n = (size_t)w * (size_t)h;
    for (size_t i = 0; i < n; i++) {
        if (GolAlive(cells[i])) {
            a++;
            uint8_t age = GolAge(cells[i]);
            if (age > ma) ma = (int)age;
        }
    }
    if (alive) *alive = a;
    if (max_age) *max_age = ma;
}

// Preset patterns
static void place_block(uint16_t *cells, int w, int h, int cx, int cy) {
    gol_set_plane(cells, w, h, cx, cy, true);
    gol_set_plane(cells, w, h, cx + 1, cy, true);
    gol_set_plane(cells, w, h, cx, cy + 1, true);
    gol_set_plane(cells, w, h, cx + 1, cy + 1, true);
}

static void place_beacon(uint16_t *cells, int w, int h, int cx, int cy) {
    gol_set_plane(cells, w, h, cx, cy, true);
    gol_set_plane(cells, w, h, cx + 1, cy, true);
    gol_set_plane(cells, w, h, cx + 2, cy - 1, true);
    gol_set_plane(cells, w, h, cx + 1, cy - 1, true);
    gol_set_plane(cells, w, h, cx, cy + 1, true);
    gol_set_plane(cells, w, h, cx + 1, cy + 1, true);
}

static void place_glider(uint16_t *cells, int w, int h, int cx, int cy) {
    gol_set_plane(cells, w, h, cx, cy + 1, true);
    gol_set_plane(cells, w, h, cx + 1, cy + 2, true);
    gol_set_plane(cells, w, h, cx + 2, cy, true);
    gol_set_plane(cells, w, h, cx + 2, cy + 1, true);
    gol_set_plane(cells, w, h, cx + 2, cy + 2, true);
}

static void place_blinker(uint16_t *cells, int w, int h, int cx, int cy) {
    gol_set_plane(cells, w, h, cx, cy, true);
    gol_set_plane(cells, w, h, cx + 1, cy, true);
    gol_set_plane(cells, w, h, cx + 2, cy, true);
}

static void place_toad(uint16_t *cells, int w, int h, int cx, int cy) {
    gol_set_plane(cells, w, h, cx + 1, cy, true);
    gol_set_plane(cells, w, h, cx + 2, cy, true);
    gol_set_plane(cells, w, h, cx + 3, cy, true);
    gol_set_plane(cells, w, h, cx, cy + 1, true);
    gol_set_plane(cells, w, h, cx + 1, cy + 1, true);
    gol_set_plane(cells, w, h, cx + 2, cy + 1, true);
}

static void place_r_pentomino(uint16_t *cells, int w, int h, int cx, int cy) {
    gol_set_plane(cells, w, h, cx + 1, cy, true);
    gol_set_plane(cells, w, h, cx + 2, cy, true);
    gol_set_plane(cells, w, h, cx, cy + 1, true);
    gol_set_plane(cells, w, h, cx + 1, cy + 1, true);
    gol_set_plane(cells, w, h, cx + 1, cy + 2, true);
}

static void place_pentadecathlon(uint16_t *cells, int w, int h, int cx, int cy) {
    gol_set_plane(cells, w, h, cx, cy, true);
    gol_set_plane(cells, w, h, cx, cy + 1, true);
    gol_set_plane(cells, w, h, cx + 1, cy + 1, true);
    gol_set_plane(cells, w, h, cx - 1, cy + 1, true);
    gol_set_plane(cells, w, h, cx, cy + 2, true);
    gol_set_plane(cells, w, h, cx, cy + 3, true);
    gol_set_plane(cells, w, h, cx, cy + 4, true);
    gol_set_plane(cells, w, h, cx + 1, cy + 4, true);
    gol_set_plane(cells, w, h, cx - 1, cy + 4, true);
    gol_set_plane(cells, w, h, cx, cy + 5, true);
    gol_set_plane(cells, w, h, cx, cy + 6, true);
}

static void place_spaceship(uint16_t *cells, int w, int h, int cx, int cy) {
    // Lightweight spaceship (glider gun variant - actually a spaceship)
    gol_set_plane(cells, w, h, cx, cy, true);
    gol_set_plane(cells, w, h, cx + 4, cy, true);
    gol_set_plane(cells, w, h, cx + 1, cy + 1, true);
    gol_set_plane(cells, w, h, cx + 4, cy + 1, true);
    gol_set_plane(cells, w, h, cx + 1, cy + 2, true);
    gol_set_plane(cells, w, h, cx + 2, cy + 2, true);
    gol_set_plane(cells, w, h, cx + 3, cy + 2, true);
    gol_set_plane(cells, w, h, cx + 4, cy + 2, true);
    gol_set_plane(cells, w, h, cx + 2, cy + 3, true);
    gol_set_plane(cells, w, h, cx + 3, cy + 3, true);
}

static void place_heptomino(uint16_t *cells, int w, int h, int cx, int cy) {
    gol_set_plane(cells, w, h, cx, cy, true);
    gol_set_plane(cells, w, h, cx + 1, cy, true);
    gol_set_plane(cells, w, h, cx + 2, cy, true);
    gol_set_plane(cells, w, h, cx + 3, cy, true);
    gol_set_plane(cells, w, h, cx + 4, cy, true);
    gol_set_plane(cells, w, h, cx + 5, cy, true);
    gol_set_plane(cells, w, h, cx + 6, cy, true);
}

void gol_apply_preset(uint16_t *cells, int w, int h, const char *preset) {
    memset(cells, 0, (size_t)w * (size_t)h * sizeof(uint16_t));
    
    int cx = w / 2;
    int cy = h / 2;
    
    if (strcmp(preset, "glider") == 0) {
        place_glider(cells, w, h, cx - 2, cy - 2);
    } else if (strcmp(preset, "blinker") == 0) {
        place_blinker(cells, w, h, cx - 1, cy);
    } else if (strcmp(preset, "block") == 0) {
        place_block(cells, w, h, cx, cy);
    } else if (strcmp(preset, "beacon") == 0) {
        place_beacon(cells, w, h, cx - 1, cy);
    } else if (strcmp(preset, "toad") == 0) {
        place_toad(cells, w, h, cx - 1, cy - 1);
    } else if (strcmp(preset, "pentadecathlon") == 0) {
        place_pentadecathlon(cells, w, h, cx, cy - 3);
    } else if (strcmp(preset, "lwss") == 0) {
        place_spaceship(cells, w, h, cx - 2, cy - 2);
    } else if (strcmp(preset, "r-pentomino") == 0) {
        place_r_pentomino(cells, w, h, cx - 1, cy - 1);
    } else if (strcmp(preset, "pentadecathlon") == 0) {
        place_pentadecathlon(cells, w, h, cx, cy - 3);
    } else if (strcmp(preset, "heptomino") == 0) {
        place_heptomino(cells, w, h, cx - 3, cy);
    }
}

void gol_resize_copy(const uint16_t *src, int srcW, int srcH,
                     uint16_t *dst, int dstW, int dstH, size_t dstPlaneCells) {
    memset(dst, 0, dstPlaneCells * sizeof(uint16_t));
    int cw = srcW < dstW ? srcW : dstW;
    int ch = srcH < dstH ? srcH : dstH;
    for (int y = 0; y < ch; y++) {
        for (int x = 0; x < cw; x++) {
            dst[(size_t)y * (size_t)dstW + (size_t)x] =
                src[(size_t)y * (size_t)srcW + (size_t)x];
        }
    }
}
