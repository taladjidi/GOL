#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

#include "gol.h"
#include "refstep.h"

#define GRID_W 64
#define GRID_H 64

#define RLE_BLOCK "2o$2o!"
#define RLE_BLINKER "3o!"
#define RLE_TOAD "b3o$3o!"
#define RLE_BEACON "2o2b$o3b$3bo$2b2o!"
#define RLE_PULSAR "2b3o3b3o2b2$o4bobo4bo$o4bobo4bo$o4bobo4bo$2b3o3b3o2b2$2b3o3b3o2b$o4bobo4bo$o4bobo4bo$o4bobo4bo2$2b3o3b3o!"
#define RLE_PENTADECATHLON "2bo4bo2b$2ob4ob2o$2bo4bo!"
#define RLE_GLIDER "bob$2bo$3o!"
#define RLE_LWSS "bo2bo$o4b$o3bo$4o!"
#define RLE_R_PENTOMINO "b2o$2ob$bo!"
#define RLE_HEPTOMINO "7o!"

static void ref_decode_rle(RefGrid *g, const char *rle, int ox, int oy, int *pw, int *ph) {
    int x = 0;
    int y = 0;
    int count = 0;
    bool have_count = false;
    int max_x = 0;
    int max_y = 0;
    const char *p = rle;

    while (*p) {
        char c = *p++;
        if (c >= '0' && c <= '9') {
            count = have_count ? count * 10 + (c - '0') : (c - '0');
            have_count = true;
        } else if (c == 'o') {
            int n = have_count ? count : 1;
            for (int i = 0; i < n; i++) {
                ref_set(g, ox + x + i, oy + y, true);
            }
            x += n;
            if (x > max_x) max_x = x;
            if (y + 1 > max_y) max_y = y + 1;
            count = 0;
            have_count = false;
        } else if (c == 'b') {
            int n = have_count ? count : 1;
            x += n;
            if (x > max_x) max_x = x;
            if (y + 1 > max_y) max_y = y + 1;
            count = 0;
            have_count = false;
        } else if (c == '$') {
            int n = have_count ? count : 1;
            y += n;
            x = 0;
            if (y > max_y) max_y = y;
            count = 0;
            have_count = false;
        } else if (c == '!') {
            break;
        } else if (c == '#') {
            while (*p && *p != '\n') ++p;
            count = 0;
            have_count = false;
        } else {
            count = 0;
            have_count = false;
        }
    }

    if (pw) *pw = max_x;
    if (ph) *ph = max_y;
}

static void ref_place_rle(RefGrid *g, const char *rle) {
    int pw = 0;
    int ph = 0;
    int ox;
    int oy;
    ref_clear(g);
    ref_decode_rle(g, rle, 0, 0, &pw, &ph);
    ref_clear(g);
    ox = g->w / 2 - pw / 2;
    oy = g->h / 2 - ph / 2;
    ref_decode_rle(g, rle, ox, oy, NULL, NULL);
}

static void test_period(const char *name, const char *rle, int period, int expected_pop) {
    RefGrid g;
    RefGrid copy;
    GOLRules rules;
    printf("Test: %s period %d\n", name, period);
    ref_grid_init(&g, GRID_W, GRID_H, false);
    ref_grid_init(&copy, GRID_W, GRID_H, false);

    ref_place_rle(&g, rle);
    assert(ref_count(&g) == expected_pop);
    ref_copy(&copy, &g);

    rules = gol_default_rules();
    for (int k = 1; k <= period; k++) {
        ref_step(&g, rules);
        if (k < period) {
            assert(!ref_equal(&g, &copy));
        } else {
            assert(ref_equal(&g, &copy));
        }
    }

    ref_grid_free(&g);
    ref_grid_free(&copy);
    printf("  PASS\n\n");
}

static void test_spaceship(const char *name, const char *rle, int dx, int dy, int expected_pop) {
    RefGrid g;
    RefGrid copy;
    GOLRules rules;
    printf("Test: %s translation\n", name);
    ref_grid_init(&g, GRID_W, GRID_H, false);
    ref_grid_init(&copy, GRID_W, GRID_H, false);

    ref_place_rle(&g, rle);
    assert(ref_count(&g) == expected_pop);
    ref_copy(&copy, &g);

    rules = gol_default_rules();
    for (int i = 0; i < 4; i++) {
        ref_step(&g, rules);
    }

    assert(ref_count(&g) == expected_pop);
    assert(ref_equals_shifted(&g, &copy, dx, dy));

    ref_grid_free(&g);
    ref_grid_free(&copy);
    printf("  PASS\n\n");
}

static uint64_t rng_state = 0x123456789abcdef0ULL;

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

static void test_cpu_matches_ref(void) {
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

    printf("Test: CPU matches reference on torus\n");

    rule_count = sizeof(rules) / sizeof(rules[0]);
    grid_count = sizeof(widths) / sizeof(widths[0]);
    density_count = sizeof(densities) / sizeof(densities[0]);

    for (size_t ri = 0; ri < rule_count; ri++) {
        for (size_t gi = 0; gi < grid_count; gi++) {
            for (size_t di = 0; di < density_count; di++) {
                int w;
                int h;
                size_t n;
                uint16_t *buf1;
                uint16_t *buf2;
                RefGrid ref;
                uint16_t *tmp;

                w = widths[gi];
                h = heights[gi];
                n = (size_t)w * (size_t)h;
                buf1 = (uint16_t *)calloc(n, sizeof(uint16_t));
                buf2 = (uint16_t *)calloc(n, sizeof(uint16_t));
                assert(buf1 != NULL && buf2 != NULL);

                ref_grid_init(&ref, w, h, true);

                for (size_t i = 0; i < n; i++) {
                    bool alive = random_alive(densities[di]);
                    buf1[i] = alive ? 1u : 0u;
                    ref.a[i] = alive ? 1u : 0u;
                }

                for (int step = 0; step < 6; step++) {
                    gol_step_cpu(buf1, buf2, w, h, rules[ri]);
                    ref_step(&ref, rules[ri]);
                    for (size_t i = 0; i < n; i++) {
                        int cpu_alive = GolAlive(buf2[i]) ? 1 : 0;
                        if (ref.a[i] != cpu_alive) {
                            printf("  MISMATCH rule=%zu grid=%dx%d density=%.2f step=%d cell=%zu\n",
                                   ri, w, h, densities[di], step, i);
                            assert(false);
                        }
                    }
                    tmp = buf1;
                    buf1 = buf2;
                    buf2 = tmp;
                }

                free(buf1);
                free(buf2);
                ref_grid_free(&ref);
            }
        }
    }

    printf("  PASS\n\n");
}

typedef struct {
    const char *name;
    const char *rle;
} PresetCase;

static const PresetCase preset_cases[] = {
    { "block", RLE_BLOCK },
    { "blinker", RLE_BLINKER },
    { "toad", RLE_TOAD },
    { "beacon", RLE_BEACON },
    { "glider", RLE_GLIDER },
    { "pentadecathlon", RLE_PENTADECATHLON },
    { "lwss", RLE_LWSS },
    { "r-pentomino", RLE_R_PENTOMINO },
    { "heptomino", RLE_HEPTOMINO }
};

static void cpu_to_ref(const uint16_t *cells, int w, int h, RefGrid *g) {
    ref_clear(g);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t i = (size_t)y * (size_t)w + (size_t)x;
            g->a[i] = GolAlive(cells[i]) ? 1u : 0u;
        }
    }
}

static void test_presets_match_rle(void) {
    printf("Test: presets match canonical RLE\n");

    for (size_t i = 0; i < sizeof(preset_cases) / sizeof(preset_cases[0]); i++) {
        size_t n;
        uint16_t *cells;
        RefGrid actual;
        RefGrid expected;

        n = (size_t)GRID_W * (size_t)GRID_H;
        cells = (uint16_t *)calloc(n, sizeof(uint16_t));
        assert(cells != NULL);

        ref_grid_init(&actual, GRID_W, GRID_H, false);
        ref_grid_init(&expected, GRID_W, GRID_H, false);

        gol_apply_preset(cells, GRID_W, GRID_H, preset_cases[i].name);
        cpu_to_ref(cells, GRID_W, GRID_H, &actual);
        ref_place_rle(&expected, preset_cases[i].rle);

        if (!ref_equal(&actual, &expected)) {
            printf("  MISMATCH preset=%s\n", preset_cases[i].name);
            assert(false);
        }

        free(cells);
        ref_grid_free(&actual);
        ref_grid_free(&expected);
    }

    printf("  PASS\n\n");
}

static bool cpu_equal_alive(const uint16_t *a, const uint16_t *b, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t i = (size_t)y * (size_t)w + (size_t)x;
            if (GolAlive(a[i]) != GolAlive(b[i])) return false;
        }
    }
    return true;
}

static bool cpu_equals_shifted(const uint16_t *cur, const uint16_t *src, int w, int h, int dx, int dy) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t i = (size_t)y * (size_t)w + (size_t)x;
            bool left = GolAlive(cur[i]);
            int sx = x - dx;
            int sy = y - dy;
            bool right = false;
            if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
                right = GolAlive(src[(size_t)sy * (size_t)w + (size_t)sx]);
            }
            if (left != right) return false;
        }
    }
    return true;
}

static void test_preset_period_cpu(const char *name, int period) {
    size_t n;
    uint16_t *a;
    uint16_t *b;
    uint16_t *init;
    GOLRules rules;
    uint16_t *tmp;

    printf("Test: CPU preset %s period %d\n", name, period);
    n = (size_t)GRID_W * (size_t)GRID_H;
    a = (uint16_t *)calloc(n, sizeof(uint16_t));
    b = (uint16_t *)calloc(n, sizeof(uint16_t));
    init = (uint16_t *)calloc(n, sizeof(uint16_t));
    assert(a != NULL && b != NULL && init != NULL);

    gol_apply_preset(a, GRID_W, GRID_H, name);
    memcpy(init, a, n * sizeof(uint16_t));

    rules = gol_default_rules();
    for (int k = 1; k <= period; k++) {
        gol_step_cpu(a, b, GRID_W, GRID_H, rules);
        tmp = a;
        a = b;
        b = tmp;
        if (k < period) {
            assert(!cpu_equal_alive(a, init, GRID_W, GRID_H));
        } else {
            assert(cpu_equal_alive(a, init, GRID_W, GRID_H));
        }
    }

    free(a);
    free(b);
    free(init);
    printf("  PASS\n\n");
}

static void test_preset_spaceship_cpu(const char *name, int dx, int dy, int expected_pop) {
    size_t n;
    uint16_t *a;
    uint16_t *b;
    uint16_t *init;
    GOLRules rules;
    uint16_t *tmp;
    int alive = 0;
    int max_age = 0;

    printf("Test: CPU preset %s translation\n", name);
    n = (size_t)GRID_W * (size_t)GRID_H;
    a = (uint16_t *)calloc(n, sizeof(uint16_t));
    b = (uint16_t *)calloc(n, sizeof(uint16_t));
    init = (uint16_t *)calloc(n, sizeof(uint16_t));
    assert(a != NULL && b != NULL && init != NULL);

    gol_apply_preset(a, GRID_W, GRID_H, name);
    memcpy(init, a, n * sizeof(uint16_t));

    rules = gol_default_rules();
    for (int i = 0; i < 4; i++) {
        gol_step_cpu(a, b, GRID_W, GRID_H, rules);
        tmp = a;
        a = b;
        b = tmp;
    }

    gol_count_alive(a, GRID_W, GRID_H, &alive, &max_age);
    assert(alive == expected_pop);
    assert(cpu_equals_shifted(a, init, GRID_W, GRID_H, dx, dy));

    free(a);
    free(b);
    free(init);
    printf("  PASS\n\n");
}

int main(void) {
    printf("=== Reference Implementation Tests ===\n\n");

    test_period("block", RLE_BLOCK, 1, 4);
    test_period("blinker", RLE_BLINKER, 2, 3);
    test_period("toad", RLE_TOAD, 2, 6);
    test_period("beacon", RLE_BEACON, 2, 6);
    test_period("pulsar", RLE_PULSAR, 3, 48);
    test_period("pentadecathlon", RLE_PENTADECATHLON, 15, 12);
    test_spaceship("glider", RLE_GLIDER, 1, 1, 5);
    test_spaceship("lwss", RLE_LWSS, -2, 0, 9);
    test_cpu_matches_ref();
    test_presets_match_rle();
    test_preset_period_cpu("block", 1);
    test_preset_period_cpu("blinker", 2);
    test_preset_period_cpu("toad", 2);
    test_preset_period_cpu("beacon", 2);
    test_preset_period_cpu("pentadecathlon", 15);
    test_preset_spaceship_cpu("glider", 1, 1, 5);
    test_preset_spaceship_cpu("lwss", -2, 0, 9);

    printf("All reference tests passed!\n");
    return 0;
}
