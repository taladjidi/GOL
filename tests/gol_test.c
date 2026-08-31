#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "gol.h"

#define GRID_W 16
#define GRID_H 16

static uint16_t *create_grid(void) {
    uint16_t *grid = calloc((size_t)GRID_W * GRID_H, sizeof(uint16_t));
    assert(grid != NULL);
    return grid;
}

static void clear_grid(uint16_t *grid) {
    memset(grid, 0, (size_t)GRID_W * GRID_H * sizeof(uint16_t));
}

static void print_grid(const uint16_t *grid, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            printf("%c", GolAlive(grid[y * w + x]) ? '#' : '.');
        }
        printf("\n");
    }
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
    printf("Test: Block still life\n");
    uint16_t *grid = create_grid();
    GOLRules rules = gol_default_rules();
    
    // Place a 2x2 block
    gol_set_plane(grid, GRID_W, GRID_H, 7, 7, true);
    gol_set_plane(grid, GRID_W, GRID_H, 8, 7, true);
    gol_set_plane(grid, GRID_W, GRID_H, 7, 8, true);
    gol_set_plane(grid, GRID_W, GRID_H, 8, 8, true);
    
    int alive_before = count_alive(grid, GRID_W, GRID_H);
    assert(alive_before == 4);
    
    uint16_t *next = create_grid();
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);
    
    int alive_after = count_alive(next, GRID_W, GRID_H);
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
    printf("Test: Blinker oscillator (period 2)\n");
    uint16_t *grid = create_grid();
    GOLRules rules = gol_default_rules();
    
    // Place a horizontal blinker
    gol_set_plane(grid, GRID_W, GRID_H, 7, 7, true);
    gol_set_plane(grid, GRID_W, GRID_H, 8, 7, true);
    gol_set_plane(grid, GRID_W, GRID_H, 9, 7, true);
    
    int alive_before = count_alive(grid, GRID_W, GRID_H);
    assert(alive_before == 3);
    
    uint16_t *next = create_grid();
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);
    
    // After one step, should be vertical
    int alive_step1 = count_alive(next, GRID_W, GRID_H);
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
    
    int alive_step2 = count_alive(next, GRID_W, GRID_H);
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
    printf("Test: Glider movement\n");
    uint16_t *grid = create_grid();
    GOLRules rules = gol_default_rules();
    
    // Place a glider at position (7,7)
    gol_set_plane(grid, GRID_W, GRID_H, 7, 8, true);  // (0,1)
    gol_set_plane(grid, GRID_W, GRID_H, 8, 9, true);  // (1,2)
    gol_set_plane(grid, GRID_W, GRID_H, 9, 7, true);  // (2,0)
    gol_set_plane(grid, GRID_W, GRID_H, 9, 8, true);  // (2,1)
    gol_set_plane(grid, GRID_W, GRID_H, 9, 9, true);  // (2,2)
    
    int alive_before = count_alive(grid, GRID_W, GRID_H);
    assert(alive_before == 5);
    
    uint16_t *next = create_grid();
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);
    
    int alive_after = count_alive(next, GRID_W, GRID_H);
    printf("  Step 0: %d alive\n", alive_before);
    printf("  Step 1: %d alive\n", alive_after);
    assert(alive_after == 5);
    
    // Glider should move to (8,9) orientation
    // Check that cells moved (not same position)
    int moved = 0;
    if (!GolAlive(next[7 * GRID_W + 8])) moved = 1;  // (0,1) -> gone
    if (GolAlive(next[7 * GRID_W + 9])) moved = 1;   // new cell at (0,2)
    if (GolAlive(next[8 * GRID_W + 7])) moved = 1;   // new cell at (1,0)
    if (GolAlive(next[8 * GRID_W + 9])) moved = 1;   // new cell at (1,2)
    if (GolAlive(next[9 * GRID_W + 8])) moved = 1;   // new cell at (2,1)
    
    printf("  Glider moved: %s\n", moved ? "yes" : "no");
    assert(moved);
    
    free(grid);
    free(next);
    printf("  PASS\n\n");
}

// Test: All cells die with no neighbors
static void test_empty_survival(void) {
    printf("Test: No survival without neighbors\n");
    uint16_t *grid = create_grid();
    GOLRules rules = gol_default_rules();
    
    // Place a single cell
    gol_set_plane(grid, GRID_W, GRID_H, 8, 8, true);
    
    int alive_before = count_alive(grid, GRID_W, GRID_H);
    assert(alive_before == 1);
    
    uint16_t *next = create_grid();
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);
    
    int alive_after = count_alive(next, GRID_W, GRID_H);
    printf("  Alive before: %d, after: %d\n", alive_before, alive_after);
    assert(alive_after == 0);
    
    free(grid);
    free(next);
    printf("  PASS\n\n");
}

// Test: Birth with exactly 3 neighbors
static void test_birth_rule(void) {
    printf("Test: Birth rule (B3)\n");
    uint16_t *grid = create_grid();
    GOLRules rules = gol_default_rules();
    
    // Place 3 cells in a row
    gol_set_plane(grid, GRID_W, GRID_H, 7, 8, true);
    gol_set_plane(grid, GRID_W, GRID_H, 8, 8, true);
    gol_set_plane(grid, GRID_W, GRID_H, 9, 8, true);
    
    uint16_t *next = create_grid();
    gol_step_cpu(grid, next, GRID_W, GRID_H, rules);
    
    // Cell at (8,7) should be born (has 3 neighbors at (7,8), (8,8), (9,8))
    // Grid index = y * GRID_W + x, so (8,7) = 7 * GRID_W + 8
    int born = GolAlive(next[7 * GRID_W + 8]) ? 1 : 0;
    printf("  Cell born at (8,7): %s\n", born ? "yes" : "no");
    assert(born);
    
    free(grid);
    free(next);
    printf("  PASS\n\n");
}

// Test: Custom rules (HighLife B36/S23)
static void test_custom_rules(void) {
    printf("Test: Custom rules (HighLife B36/S23)\n");
    GOLRules rules = gol_default_rules();
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

int main(void) {
    printf("=== Game of Life Tests ===\n\n");
    
    test_block_still_life();
    test_blinker_oscillator();
    test_glider_movement();
    test_empty_survival();
    test_birth_rule();
    test_custom_rules();
    
    printf("All tests passed!\n");
    return 0;
}
