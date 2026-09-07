#ifndef GOL_H
#define GOL_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint16_t birth;     // bitmask: bit n set if birth on n neighbors (0..8)
    uint16_t survival;  // bitmask: bit n set if survive on n neighbors (0..8)
} GOLRules;

// Cell bit packing: bit 0 = alive, bits 1..15 = age (0..32767)
static inline uint16_t GolAge(uint16_t v) { return (v >> 1) & 0x7FFFu; }
static inline bool GolAlive(uint16_t v) { return (v & 1u) != 0u; }

void gol_seed(uint64_t seed);
void gol_randomize(uint16_t *cells, size_t planeCells, int w, int h, double density);
void gol_set_plane(uint16_t *cells, int w, int h, int x, int y, bool alive);
void gol_step_cpu(const uint16_t *cur, uint16_t *next, int w, int h, GOLRules rules);
void gol_resize_copy(const uint16_t *src, int srcW, int srcH,
                     uint16_t *dst, int dstW, int dstH, size_t dstPlaneCells);
void gol_copy_region(const uint16_t *src, int srcW, int srcH,
                     uint16_t *dst, int dstW, int dstH, size_t dstPlaneCells,
                     int srcX, int srcY);
void gol_apply_preset(uint16_t *cells, int w, int h, const char *preset);
void gol_count_alive(const uint16_t *cells, int w, int h, int *alive, int *max_age);

// Rule helpers
GOLRules gol_default_rules(void);
bool gol_rule_birth(GOLRules r, int neighbors);
bool gol_rule_survive(GOLRules r, int neighbors);

#endif
