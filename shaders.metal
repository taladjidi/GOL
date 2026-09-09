#include <metal_stdlib>
using namespace metal;

// Shared CPU/GPU uniform values.
// curOffset is a cell index, not a byte offset. It selects the active plane
// inside the full grid buffer for the fragment shader.
struct Uniforms {
    uint gridW;
    uint gridH;
    uint curOffset;
    uint pad;
    // Rule bitmasks (bit n set if the rule acts on n neighbors, n in 0..8)
    ushort birth;
    ushort survival;
    float viewScaleX;
    float viewScaleY;
    float viewOffsetX;
    float viewOffsetY;
    float viewWidth;
    float viewHeight;
    uint displayMode;
    uint palette;
    float glow;
    float cursorX;
    float cursorY;
    float brushR;
    float pad2;
};

struct VSOut {
    float4 pos [[position]];
    float4 uv;
};

// Fullscreen quad corners.
constant float2 kCornerPos[4] = {
    float2(-1.0f, -1.0f),
    float2( 1.0f, -1.0f),
    float2(-1.0f,  1.0f),
    float2( 1.0f,  1.0f),
};

constant float2 kCornerUV[4] = {
    float2(0.0f, 0.0f),
    float2(1.0f, 0.0f),
    float2(0.0f, 1.0f),
    float2(1.0f, 1.0f),
};

[[vertex]] VSOut vs_main(uint vid [[vertex_id]]) {
    VSOut o;
    o.pos = float4(kCornerPos[vid], 0.0f, 1.0f);
    o.uv = float4(kCornerUV[vid], 0.0f, 0.0f);
    return o;
}

// Approximate matplotlib "viridis" colormap.
// Index 0 is dark blue/purple, index 9 is yellow.
constant float3 kViridis[10] = {
    float3(0.2667f, 0.0039f, 0.3294f),
    float3(0.2824f, 0.1569f, 0.4706f),
    float3(0.2431f, 0.2863f, 0.5373f),
    float3(0.1922f, 0.4078f, 0.5569f),
    float3(0.1490f, 0.5098f, 0.5569f),
    float3(0.1294f, 0.5686f, 0.5490f),
    float3(0.2078f, 0.7176f, 0.4745f),
    float3(0.3686f, 0.7882f, 0.3843f),
    float3(0.7098f, 0.8706f, 0.1686f),
    float3(0.9922f, 0.9059f, 0.1451f),
};

static float3 Viridis(float t) {
    t = clamp(t, 0.0f, 1.0f);
    float scaled = t * 9.0f;
    int i0 = static_cast<int>(scaled);
    int i1 = min(i0 + 1, 9);
    float f = scaled - static_cast<float>(i0);
    return mix(kViridis[i0], kViridis[i1], f);
}

// Piecewise-linear blend through four color stops (t in [0,1]).
static float3 StopMix(float t, float3 c0, float3 c1, float3 c2, float3 c3) {
    t = clamp(t, 0.0f, 1.0f);
    float x = t * 3.0f;
    if (x < 1.0f) {
        return mix(c0, c1, x);
    }
    if (x < 2.0f) {
        return mix(c1, c2, x - 1.0f);
    }
    return mix(c2, c3, x - 2.0f);
}

static float3 Inferno(float t) {
    return StopMix(t,
        float3(0.019f, 0.025f, 0.078f),
        float3(0.510f, 0.024f, 0.490f),
        float3(0.938f, 0.361f, 0.267f),
        float3(0.988f, 0.992f, 0.667f));
}

static float3 Plasma(float t) {
    return StopMix(t,
        float3(0.246f, 0.012f, 0.518f),
        float3(0.647f, 0.120f, 0.737f),
        float3(0.930f, 0.427f, 0.396f),
        float3(0.961f, 0.968f, 0.412f));
}

static float3 Turbo(float t) {
    return StopMix(t,
        float3(0.486f, 0.000f, 1.000f),
        float3(0.000f, 0.467f, 0.894f),
        float3(0.173f, 0.894f, 0.361f),
        float3(1.000f, 0.945f, 0.000f));
}

// Selects the active colormap by palette index.
static float3 Ramp(float t, uint palette) {
    if (palette == 1u) {
        return Inferno(t);
    }
    if (palette == 2u) {
        return Plasma(t);
    }
    if (palette == 3u) {
        return Turbo(t);
    }
    return Viridis(t);
}

constant uint kDisplayTrails = 1u;
constant uint kDisplayHeatmap = 2u;

constexpr sampler s_linear(filter::linear, address::clamp_to_edge);
constexpr sampler s_mip(filter::linear, mip_filter::linear, address::clamp_to_edge);

// Renders the active grid plane into a small cell texture.
// Cell packing: bit 0 = alive, bits 1..15 = age.
[[fragment]] float4 fs_main(VSOut in [[stage_in]],
                            device const ushort *cells [[buffer(0)]],
                             constant Uniforms &u [[buffer(1)]]) {

    // Flip uv.y so grid row 0 is stored at the top of the cell texture.
    float2 cellF = float2(in.uv.x, 1.0f - in.uv.y) *
        float2(static_cast<float>(u.gridW), static_cast<float>(u.gridH));
    int2 icell = int2(floor(cellF));
    icell = clamp(icell, int2(0, 0),
                  int2(static_cast<int>(u.gridW) - 1, static_cast<int>(u.gridH) - 1));
    uint idx = u.curOffset + static_cast<uint>(icell.y) * u.gridW + static_cast<uint>(icell.x);
    ushort v = cells[idx];

    if ((v & 1u) == 0u) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float age = static_cast<float>((v >> 1) & 0x7FFFu);
    float3 col;
    if (u.displayMode == kDisplayHeatmap) {
        // Heat builds up and saturates quickly.
        col = Ramp(min(age / 30.0f, 1.0f), u.palette);
    } else {
        // Age: newest cells at the bright end, dimming as they age.
        col = Ramp(1.0f - min(age / 100.0f, 1.0f), u.palette);
    }
    return float4(col, 1.0f);
}

// Brush cursor preview ring: a soft circle of radius u.brushR (in cells)
// centered on the cursor, one pixel wide at any zoom. Returns the blend
// amount toward the ring color; 0 when the cursor is hidden.
static float BrushRing(float2 gridF, float2 scale, constant Uniforms &u) {
    float s;
    float d;

    if (u.cursorX < -1e8f) {
        return 0.0f;
    }
    s = max(min(scale.x, scale.y), 1e-6f);
    d = abs(length(gridF - float2(u.cursorX, u.cursorY)) - u.brushR) / s;
    return (1.0f - smoothstep(0.0f, 1.0f, d)) * 0.6f;
}

// Scales the small cell texture to the drawable and draws cell gaps.
[[fragment]] float4 fs_scale(VSOut in [[stage_in]],
                              texture2d<float> tex [[texture(0)]],
                              texture2d<float> trail [[texture(1)]],
                              constant Uniforms &u [[buffer(0)]]) {
    const float4 bg = float4(0.04f, 0.05f, 0.08f, 1.0f);
    float2 viewSize = float2(max(u.viewWidth, 1.0f), max(u.viewHeight, 1.0f));
    float2 viewPos = float2(in.uv.x, 1.0f - in.uv.y) * viewSize;
    float2 scale = float2(max(u.viewScaleX, 0.0f), max(u.viewScaleY, 0.0f));
    float2 gridF = (viewPos - float2(u.viewOffsetX, u.viewOffsetY)) * scale;
    if (gridF.x < 0.0f || gridF.x >= static_cast<float>(u.gridW) ||
        gridF.y < 0.0f || gridF.y >= static_cast<float>(u.gridH)) {
        return float4(mix(bg.rgb, float3(0.9f), BrushRing(gridF, scale, u)), 1.0f);
    }
    float2 f = fract(gridF);
    bool subpixel = (u.viewScaleX > 1.0f) || (u.viewScaleY > 1.0f);
    int2 icell = int2(floor(gridF));
    icell = clamp(icell, int2(0, 0),
                  int2(static_cast<int>(u.gridW) - 1, static_cast<int>(u.gridH) - 1));
    uint2 tcoord = uint2(static_cast<uint>(icell.x), static_cast<uint>(icell.y));
    float4 sample;
    if (subpixel) {
        // Multiple cells per pixel: linear filtering keeps shrunken patterns smooth.
        float2 tcoordF = (float2(icell) + 0.5f) /
            float2(static_cast<float>(u.gridW), static_cast<float>(u.gridH));
        if (u.displayMode == kDisplayTrails) {
            float t = trail.sample(s_linear, tcoordF).r;
            sample = float4(Ramp(t, u.palette), t);
        } else {
            // Mipmapped tap at the exact fractional position: a fixed bilinear
            // tap sees only a few of the many cells under a pixel and sparkles;
            // the lod matching the zoom averages them into a smooth density map.
            float2 uv = gridF /
                float2(static_cast<float>(u.gridW), static_cast<float>(u.gridH));
            float lod = log2(max(max(u.viewScaleX, u.viewScaleY), 1.0f));
            sample = tex.sample(s_mip, uv, level(lod));
        }
        float3 col = bg.rgb * (1.0f - sample.a) + sample.rgb;
        col += sample.rgb * u.glow * 0.3f;
        col = mix(col, float3(0.9f), BrushRing(gridF, scale, u));
        return float4(col, 1.0f);
    }
    if (u.displayMode == kDisplayTrails) {
        float t = trail.read(tcoord, 0).r;
        sample = float4(Ramp(t, u.palette), t);
    } else {
        sample = tex.read(tcoord, 0);
    }
    // Soft edge: keep a small gap, then fade the cell over ~1px near its border
    // instead of a hard cut, so cells look rounded rather than aliased squares.
    const float gap = 0.16f;
    float e = min(min(f.x, 1.0f - f.x), min(f.y, 1.0f - f.y));
    float px = clamp(min(u.viewScaleX, u.viewScaleY), 0.0f, (0.5f - gap) * 0.9f);
    float bodyA = sample.a * smoothstep(gap, gap + max(px, 1e-3f), e);
    // Radial glow: a soft luminous halo that fades out from the cell center.
    float d = length(f - 0.5f);
    float glowA = sample.a * (1.0f - smoothstep(0.0f, 0.55f, d)) * u.glow;
    float a = clamp(bodyA + glowA, 0.0f, 1.0f);
    if (a < 0.004f) {
        return float4(mix(bg.rgb, float3(0.9f), BrushRing(gridF, scale, u)), 1.0f);
    }
    float3 col = bg.rgb * (1.0f - a) + sample.rgb * a;
    col += sample.rgb * glowA * 0.3f;
    col = mix(col, float3(0.9f), BrushRing(gridF, scale, u));
    return float4(col, 1.0f);
}

// Wraps a possibly-negative coordinate back into [0, size) for the torus.
static inline uint wrapCoord(int v, uint size) {
    int r = v % static_cast<int>(size);
    if (r < 0) { r += static_cast<int>(size); }
    return static_cast<uint>(r);
}

// Advances one generation: reads cur plane, writes next plane.
// Each 16x16 threadgroup cooperatively loads an 18x18 tile (the 16x16 block
// plus a one-cell torus-wrapped halo) into shared memory, so the eight neighbor
// reads hit fast threadgroup storage instead of nine strided global loads.
// Out-of-range threads still load the tile and reach the barrier and reduction,
// contributing zeros; each simdgroup then adds its alive count and max age into
// stats[0]/stats[1]. The tile is sized for a 16x16 threadgroup.
[[kernel]] void gol_step(device const ushort *cur [[buffer(0)]],
                         device ushort *next [[buffer(1)]],
                         constant Uniforms &u [[buffer(2)]],
                         device atomic_uint *stats [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]],
                         uint2 tid [[thread_position_in_threadgroup]],
                         uint simdLane [[thread_index_in_simdgroup]]) {
    threadgroup ushort tile[18][18];

    bool inRange = (gid.x < u.gridW) && (gid.y < u.gridH);
    uint outAlive = 0u;
    uint outAge = 0u;

    uint gw = u.gridW;
    uint gh = u.gridH;
    int a = static_cast<int>(tid.x);
    int b = static_cast<int>(tid.y);
    int gx = static_cast<int>(gid.x);   // this thread's center cell (global)
    int gy = static_cast<int>(gid.y);

    // Load the 18x18 tile; each of its 324 cells is written exactly once
    // across the 256 threads (center by all, edges/corners by border threads).
    tile[b + 1][a + 1] = cur[wrapCoord(gy, gh) * gw + wrapCoord(gx, gw)];
    if (b == 0) {
        tile[0][a + 1] = cur[wrapCoord(gy - 1, gh) * gw + wrapCoord(gx, gw)];
    } else if (b == 15) {
        tile[17][a + 1] = cur[wrapCoord(gy + 1, gh) * gw + wrapCoord(gx, gw)];
    }
    if (a == 0) {
        tile[b + 1][0] = cur[wrapCoord(gy, gh) * gw + wrapCoord(gx - 1, gw)];
    } else if (a == 15) {
        tile[b + 1][17] = cur[wrapCoord(gy, gh) * gw + wrapCoord(gx + 1, gw)];
    }
    if (a == 0 && b == 0) {
        tile[0][0] = cur[wrapCoord(gy - 1, gh) * gw + wrapCoord(gx - 1, gw)];
    } else if (a == 15 && b == 0) {
        tile[0][17] = cur[wrapCoord(gy - 1, gh) * gw + wrapCoord(gx + 1, gw)];
    } else if (a == 0 && b == 15) {
        tile[17][0] = cur[wrapCoord(gy + 1, gh) * gw + wrapCoord(gx - 1, gw)];
    } else if (a == 15 && b == 15) {
        tile[17][17] = cur[wrapCoord(gy + 1, gh) * gw + wrapCoord(gx + 1, gw)];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (inRange) {
        int n = (tile[b][a] & 1) + (tile[b][a + 1] & 1) + (tile[b][a + 2] & 1)
              + (tile[b + 1][a] & 1) + (tile[b + 1][a + 2] & 1)
              + (tile[b + 2][a] & 1) + (tile[b + 2][a + 1] & 1) + (tile[b + 2][a + 2] & 1);

        ushort v = tile[b + 1][a + 1];
        bool alive = (v & 1u) != 0u;
        uint age = (v >> 1) & 0x7FFFu;

        uint out = 0u;
        uchar birthBit = static_cast<uchar>((u.birth >> n) & 1u);
        uchar survBit = static_cast<uchar>((u.survival >> n) & 1u);
        if (birthBit || (alive && survBit)) {
            uint newAge = alive ? (age >= 32767u ? 32767u : age + 1u) : 1u;
            out = 1u | (newAge << 1);
        }

        uint idx = gid.y * gw + gid.x;
        next[idx] = static_cast<ushort>(out);
        outAlive = out & 1u;
        outAge = (out >> 1) & 0x7FFFu;
    }

    uint groupAlive = simd_sum(outAlive);
    uint groupMaxAge = simd_max(outAge);
    if (simdLane == 0u) {
        atomic_fetch_add_explicit(&stats[0], groupAlive, memory_order_relaxed);
        atomic_fetch_max_explicit(&stats[1], groupMaxAge, memory_order_relaxed);
    }
}

// Fades the persistent trail intensity and records the current cell's liveness.
[[kernel]] void trail_step(texture2d<float, access::read> cell [[texture(0)]],
                           texture2d<float, access::read> trailRead [[texture(1)]],
                           texture2d<float, access::write> trailWrite [[texture(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint2 size = uint2(trailWrite.get_width(), trailWrite.get_height());
    if (gid.x >= size.x || gid.y >= size.y) return;
    float old = trailRead.read(gid).r * 0.94f;
    float cur = cell.read(gid).a;
    float out = max(old, cur);
    trailWrite.write(float4(out, 0.0f, 0.0f, 1.0f), gid);
}

// Zeroes the persistent trail texture.
[[kernel]] void trail_clear(texture2d<float, access::write> trail [[texture(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint2 size = uint2(trail.get_width(), trail.get_height());
    if (gid.x >= size.x || gid.y >= size.y) return;
    trail.write(float4(0.0f, 0.0f, 0.0f, 0.0f), gid);
}
