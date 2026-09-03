#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "gol.h"

#define GRID_W 16
#define GRID_H 16

static uint16_t *create_grid(void) {
    uint16_t *grid = (uint16_t *)calloc((size_t)GRID_W * GRID_H, sizeof(uint16_t));
    assert(grid != NULL);
    return grid;
}

static void clear_grid(uint16_t *grid) {
    memset(grid, 0, (size_t)GRID_W * GRID_H * sizeof(uint16_t));
}

static int count_alive(const uint16_t *grid, int w, int h) {
    int count = 0;
    for (int i = 0; i < w * h; i++) {
        if (GolAlive(grid[i])) count++;
    }
    return count;
}

// Test: Block is a still life - should not change
static void test_block_still_life(void) {
    uint16_t *grid;
    uint16_t *next;
    GOLRules rules;
    int alive_before;
    int alive_after;

    printf("Test: Block still life\n");
    grid = create_grid();
    rules = gol_default_rules();

    // Place a 2x2 block
    gol_set_plane(grid, GRID_W, GRID_H, 7, 7, true);
    gol_set_plane(grid, GRID_W, GRID_H, 8, 7, true);
    gol_set_plane(grid, GRID_W, GRID_H, 7, 8, true);
    gol_set_plane(grid, GRID_W, GRID_H, 8, 8, true);

    alive_before = count_alive(grid, GRID_W, GRID_H);
    assert(alive_before == 4);

    next = create_grid();
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);

    alive_after = count_alive(next, GRID_W, GRID_H);
    printf("  Alive before: %d, after: %d\n", alive_before, alive_after);
    assert(alive_after == 4);

    // Check each cell stayed alive
    assert(GolAlive(next[7 * GRID_W + 7]));
    assert(GolAlive(next[7 * GRID_W + 8]));
    assert(GolAlive(next[8 * GRID_W + 7]));
    assert(GolAlive(next[8 * GRID_W + 8]));

    free(grid);
    free(next);
    printf("  PASS\n\n");
}

// Test: Blinker is an oscillator with period 2
static void test_blinker_oscillator(void) {
    uint16_t *grid;
    uint16_t *next;
    GOLRules rules;
    int alive_before;
    int alive_step1;
    int alive_step2;

    printf("Test: Blinker oscillator (period 2)\n");
    grid = create_grid();
    rules = gol_default_rules();

    // Place a horizontal blinker
    gol_set_plane(grid, GRID_W, GRID_H, 7, 7, true);
    gol_set_plane(grid, GRID_W, GRID_H, 8, 7, true);
    gol_set_plane(grid, GRID_W, GRID_H, 9, 7, true);

    alive_before = count_alive(grid, GRID_W, GRID_H);
    assert(alive_before == 3);

    next = create_grid();
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);

    // After one step, should be vertical
    alive_step1 = count_alive(next, GRID_W, GRID_H);
    printf("  Step 0: %d alive\n", alive_before);
    printf("  Step 1: %d alive\n", alive_step1);
    assert(alive_step1 == 3);

    // Check vertical orientation (centered at x=8)
    // Grid index = y * GRID_W + x
    assert(!GolAlive(next[7 * GRID_W + 7]));  // (7,7) should be dead
    assert(GolAlive(next[6 * GRID_W + 8]));  // (8,6) should be alive
    assert(GolAlive(next[7 * GRID_W + 8]));  // (8,7) should be alive
    assert(GolAlive(next[8 * GRID_W + 8]));  // (8,8) should be alive
    assert(!GolAlive(next[7 * GRID_W + 9]));  // (9,7) should be dead

    // Step again
    clear_grid(grid);
    memcpy(grid, next, (size_t)GRID_W * GRID_H * sizeof(uint16_t));
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);

    alive_step2 = count_alive(next, GRID_W, GRID_H);
    printf("  Step 2: %d alive\n", alive_step2);
    assert(alive_step2 == 3);

    // Should be back to horizontal
    assert(GolAlive(next[7 * GRID_W + 7]));
    assert(GolAlive(next[7 * GRID_W + 8]));
    assert(GolAlive(next[7 * GRID_W + 9]));

    free(grid);
    free(next);
    printf("  PASS\n\n");
}

// Test: Glider moves diagonally
static void test_glider_movement(void) {
    uint16_t *grid;
    uint16_t *next;
    GOLRules rules;
    int alive_before;
    int alive_after;
    int moved;

    printf("Test: Glider movement\n");
    grid = create_grid();
    rules = gol_default_rules();

    // Place a glider at position (7,7)
    gol_set_plane(grid, GRID_W, GRID_H, 7, 8, true);  // (0,1)
    gol_set_plane(grid, GRID_W, GRID_H, 8, 9, true);  // (1,2)
    gol_set_plane(grid, GRID_W, GRID_H, 9, 7, true);  // (2,0)
    gol_set_plane(grid, GRID_W, GRID_H, 9, 8, true);  // (2,1)
    gol_set_plane(grid, GRID_W, GRID_H, 9, 9, true);  // (2,2)

    alive_before = count_alive(grid, GRID_W, GRID_H);
    assert(alive_before == 5);

    next = create_grid();
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);

    alive_after = count_alive(next, GRID_W, GRID_H);
    printf("  Step 0: %d alive\n", alive_before);
    printf("  Step 1: %d alive\n", alive_after);
    assert(alive_after == 5);

    // Glider should move to (8,9) orientation
    // Check that cells moved (not same position)
    moved = 0;
    if (!GolAlive(next[7 * GRID_W + 8])) moved = 1;  // (0,1) -> gone
    if (GolAlive(next[7 * GRID_W + 9])) moved = 1;   // new cell at (0,2)
    if (GolAlive(next[8 * GRID_W + 7])) moved = 1;   // new cell at (1,0)
    if (GolAlive(next[8 * GRID_W + 9])) moved = 1;   // new cell at (1,2)
    if (GolAlive(next[9 * GRID_W + 8])) moved = 1;   // (2,1)

    printf("  Glider moved: %s\n", moved ? "yes" : "no");
    assert(moved);

    free(grid);
    free(next);
    printf("  PASS\n\n");
}

// Test: All cells die with no neighbors
static void test_empty_survival(void) {
    uint16_t *grid;
    uint16_t *next;
    GOLRules rules;
    int alive_before;
    int alive_after;

    printf("Test: No survival without neighbors\n");
    grid = create_grid();
    rules = gol_default_rules();

    // Place a single cell
    gol_set_plane(grid, GRID_W, GRID_H, 8, 8, true);

    alive_before = count_alive(grid, GRID_W, GRID_H);
    assert(alive_before == 1);

    next = create_grid();
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);

    alive_after = count_alive(next, GRID_W, GRID_H);
    printf("  Alive before: %d, after: %d\n", alive_before, alive_after);
    assert(alive_after == 0);

    free(grid);
    free(next);
    printf("  PASS\n\n");
}

// Test: Birth with exactly 3 neighbors
static void test_birth_rule(void) {
    uint16_t *grid;
    uint16_t *next;
    GOLRules rules;
    int born;

    printf("Test: Birth rule (B3)\n");
    grid = create_grid();
    rules = gol_default_rules();

    // Place 3 cells in a row
    gol_set_plane(grid, GRID_W, GRID_H, 7, 8, true);
    gol_set_plane(grid, GRID_W, GRID_H, 8, 8, true);
    gol_set_plane(grid, GRID_W, GRID_H, 9, 8, true);

    next = create_grid();
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);

    // Cell at (8,7) should be born (has 3 neighbors at (7,8), (8,8), (9,8))
    // Grid index = y * GRID_W + x, so (8,7) = 7 * GRID_W + 8
    born = GolAlive(next[7 * GRID_W + 8]) ? 1 : 0;
    printf("  Cell born at (8,7): %s\n", born ? "yes" : "no");
    assert(born);

    free(grid);
    free(next);
    printf("  PASS\n\n");
}

// Test: Custom rules (HighLife B36/S23)
static void test_custom_rules(void) {
    GOLRules rules;

    printf("Test: Custom rules (HighLife B36/S23)\n");
    rules = gol_default_rules();
    rules.birth = (1u << 3) | (1u << 6);  // B36
    rules.survival = (1u << 2) | (1u << 3);  // S23

    assert(gol_rule_birth(rules, 3));
    assert(gol_rule_birth(rules, 6));
    assert(!gol_rule_birth(rules, 4));
    assert(gol_rule_survive(rules, 2));
    assert(gol_rule_survive(rules, 3));
    assert(!gol_rule_survive(rules, 4));

    printf("  PASS\n\n");
}

// Test: B2/S rules - cells born with exactly 2 neighbors
static void test_b2_rules(void) {
    GOLRules rules;
    uint16_t *grid;
    uint16_t *next;
    int alive;

    printf("Test: B2/S rules\n");
    rules = gol_default_rules();
    rules.birth = (1u << 2);  // B2
    rules.survival = 0;       // no survival

    assert(gol_rule_birth(rules, 2));
    assert(!gol_rule_birth(rules, 3));
    assert(!gol_rule_survive(rules, 2));
    assert(!gol_rule_survive(rules, 3));

    // Place 2 adjacent cells - should birth at (8,7) and (8,9)
    grid = create_grid();
    gol_set_plane(grid, GRID_W, GRID_H, 7, 8, true);
    gol_set_plane(grid, GRID_W, GRID_H, 8, 8, true);

    next = create_grid();
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);

    // Both original cells die (no survival), new cells born where there are exactly 2 neighbors
    alive = count_alive(next, GRID_W, GRID_H);
    printf("  Alive before: 2, after: %d\n", alive);
    assert(alive == 4);

    free(grid);
    free(next);
    printf("  PASS\n\n");
}

// Test: Dynamic rule change - start with B3/S23, switch to B2/S
static void test_dynamic_rule_change(void) {
    uint16_t *grid;
    uint16_t *next;
    GOLRules rules;
    int alive;

    printf("Test: Dynamic rule change\n");

    grid = create_grid();
    next = create_grid();

    // Place a block (still life under B3/S23)
    gol_set_plane(grid, GRID_W, GRID_H, 7, 7, true);
    gol_set_plane(grid, GRID_W, GRID_H, 8, 7, true);
    gol_set_plane(grid, GRID_W, GRID_H, 7, 8, true);
    gol_set_plane(grid, GRID_W, GRID_H, 8, 8, true);

    // Step under B3/S23 - block should survive
    rules = gol_default_rules();  // B3/S23
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);
    assert(count_alive(next, GRID_W, GRID_H) == 4);

    // Switch rules to B2/S (no survival, birth on 2)
    rules.birth = (1u << 2);
    rules.survival = 0;

    // Copy next back to grid and step again
    clear_grid(grid);
    memcpy(grid, next, (size_t)GRID_W * GRID_H * sizeof(uint16_t));
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);

    // Under B2/S, the block cells each have 3 neighbors (die) but new cells born
    // where there are exactly 2 neighbors (above and below the block)
    alive = count_alive(next, GRID_W, GRID_H);
    printf("  After rule change (B3/S23 -> B2/S): %d alive\n", alive);
    assert(alive > 0);

    // Switch back to B3/S23
    rules = gol_default_rules();  // B3/S23

    clear_grid(grid);
    memcpy(grid, next, (size_t)GRID_W * GRID_H * sizeof(uint16_t));
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);

    alive = count_alive(next, GRID_W, GRID_H);
    printf("  After switching back to B3/S23: %d alive\n", alive);
    assert(alive > 0);

    free(grid);
    free(next);
    printf("  PASS\n\n");
}

// Test: Glider under B3/S23 after rule changes and recovery
static void test_glider_recovery(void) {
    uint16_t *grid;
    uint16_t *next;
    GOLRules rules;
    int alive1;
    int alive2;
    int alive3;
    int a;

    printf("Test: Glider recovery after rule changes\n");

    grid = create_grid();
    next = create_grid();

    // Place a glider
    gol_set_plane(grid, GRID_W, GRID_H, 7, 8, true);
    gol_set_plane(grid, GRID_W, GRID_H, 8, 9, true);
    gol_set_plane(grid, GRID_W, GRID_H, 9, 7, true);
    gol_set_plane(grid, GRID_W, GRID_H, 9, 8, true);
    gol_set_plane(grid, GRID_W, GRID_H, 9, 9, true);

    rules = gol_default_rules();  // B3/S23

    // Step once under B3/S23
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);
    alive1 = count_alive(next, GRID_W, GRID_H);
    assert(alive1 == 5);

    // Change to a different ruleset
    rules.birth = (1u << 3) | (1u << 6);
    rules.survival = (1u << 2) | (1u << 3);

    clear_grid(grid);
    memcpy(grid, next, (size_t)GRID_W * GRID_H * sizeof(uint16_t));
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);
    alive2 = count_alive(next, GRID_W, GRID_H);
    printf("  After switching to B36/S23: %d alive\n", alive2);

    // Change back to B3/S23
    rules = gol_default_rules();

    clear_grid(grid);
    memcpy(grid, next, (size_t)GRID_W * GRID_H * sizeof(uint16_t));
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);
    alive3 = count_alive(next, GRID_W, GRID_H);
    printf("  After switching back to B3/S23: %d alive\n", alive3);

    // The pattern should still be a valid GOL pattern (not all dead)
    assert(alive3 > 0);

    // Step a few more times under B3/S23 to verify normal evolution
    for (int i = 0; i < 5; i++) {
        clear_grid(grid);
        memcpy(grid, next, (size_t)GRID_W * GRID_H * sizeof(uint16_t));
        gol_step_cpu(grid, next, GRID_W, GRID_H, rules);
        a = count_alive(next, GRID_W, GRID_H);
        assert(a > 0 && a < GRID_W * GRID_H);
    }

    printf("  Glider recovery: PASS\n\n");

    free(grid);
    free(next);
}

// Test: Range rules - birth on 2-4 neighbors, survive on 1-3
static void test_range_rules(void) {
    GOLRules rules;

    printf("Test: Range rules (B2-4/S1-3)\n");

    rules = gol_default_rules();
    // B2-4: birth on 2, 3, or 4 neighbors
    rules.birth = (1u << 2) | (1u << 3) | (1u << 4);
    // S1-3: survive on 1, 2, or 3 neighbors
    rules.survival = (1u << 1) | (1u << 2) | (1u << 3);

    assert(gol_rule_birth(rules, 2));
    assert(gol_rule_birth(rules, 3));
    assert(gol_rule_birth(rules, 4));
    assert(!gol_rule_birth(rules, 1));
    assert(!gol_rule_birth(rules, 5));

    assert(gol_rule_survive(rules, 1));
    assert(gol_rule_survive(rules, 2));
    assert(gol_rule_survive(rules, 3));
    assert(!gol_rule_survive(rules, 0));
    assert(!gol_rule_survive(rules, 4));

    printf("  PASS\n\n");
}

static void test_copy_region(void) {
    uint16_t *src;
    uint16_t *dst;
    int x;
    int y;
    int alive;

    printf("Test: Copy region with origin\n");
    src = (uint16_t *)calloc(8 * 8, sizeof(uint16_t));
    dst = (uint16_t *)calloc(256, sizeof(uint16_t));
    assert(src != NULL);
    assert(dst != NULL);

    for (y = 0; y < 8; y++) {
        for (x = 0; x < 8; x++) {
            src[(size_t)y * 8 + (size_t)x] = (x == 2 && y == 3) ? (uint16_t)1u : (uint16_t)0u;
        }
    }

    gol_copy_region(src, 8, 8, dst, 10, 10, 256, -1, -2);
    assert(GolAlive(dst[(size_t)5 * 10 + (size_t)3]));
    alive = count_alive(dst, 10, 10);
    printf("  Alive after shifted copy: %d\n", alive);
    assert(alive == 1);

    gol_copy_region(src, 8, 8, dst, 4, 4, 256, 4, 4);
    alive = count_alive(dst, 4, 4);
    printf("  Alive after cropped copy: %d\n", alive);
    assert(alive == 0);

    free(src);
    free(dst);
    printf("  PASS\n\n");
}

int main(void) {
    printf("=== Game of Life Tests ===\n\n");

    test_block_still_life();
    test_blinker_oscillator();
    test_glider_movement();
    test_empty_survival();
    test_birth_rule();
    test_custom_rules();
    test_b2_rules();
    test_dynamic_rule_change();
    test_glider_recovery();
    test_range_rules();
    test_copy_region();

    printf("All tests passed!\n");
    return 0;
}
