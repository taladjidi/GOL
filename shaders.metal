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
    // Rule bitmasks
    uchar birth;
    uchar survival;
    uchar pad2;
    uchar pad3;
    float viewScaleX;
    float viewScaleY;
    float viewOffsetX;
    float viewOffsetY;
    float viewWidth;
    float viewHeight;
    uint displayMode;
    uint pad4;
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

constant uint kDisplayTrails = 1u;
constant uint kDisplayHeatmap = 2u;

constexpr sampler s_linear(filter::linear, address::clamp_to_edge);

static float3 Heat(float t) {
    float3 cool = mix(float3(0.05f, 0.15f, 0.65f), float3(0.95f, 0.25f, 0.10f), t);
    return mix(cool, float3(1.0f, 0.95f, 0.25f), smoothstep(0.65f, 1.0f, t));
}

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
    float age = static_cast<float>((v >> 1) & 0x7FFFu) / 100.0f;
    float t = 1.0f - min(age, 1.0f);
    if (u.displayMode == kDisplayHeatmap) {
        return float4(Heat(t), 1.0f);
    }
    return float4(Viridis(t), 1.0f);
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
        return bg;
    }
    float2 f = fract(gridF);
    const float gap = 0.16f;
    bool subpixel = (u.viewScaleX > 1.0f) || (u.viewScaleY > 1.0f);
    if (!subpixel && (f.x < gap || f.x > 1.0f - gap || f.y < gap || f.y > 1.0f - gap)) {
        return bg;
    }
    int2 icell = int2(floor(gridF));
    icell = clamp(icell, int2(0, 0),
                  int2(static_cast<int>(u.gridW) - 1, static_cast<int>(u.gridH) - 1));
    uint2 tcoord = uint2(static_cast<uint>(icell.x), static_cast<uint>(icell.y));
    float4 sample;
    if (subpixel) {
        float2 tcoordF = (float2(icell) + 0.5f) /
            float2(static_cast<float>(u.gridW), static_cast<float>(u.gridH));
        sample = (u.displayMode == kDisplayTrails) ? trail.sample(s_linear, tcoordF) :
                                                       tex.sample(s_linear, tcoordF);
        float3 col = bg.rgb * (1.0f - sample.a) + sample.rgb;
        return float4(col, 1.0f);
    }
    sample = (u.displayMode == kDisplayTrails) ? trail.read(tcoord, 0) : tex.read(tcoord, 0);
    if (sample.a < 0.5f) {
        return bg;
    }
    return float4(sample.r, sample.g, sample.b, 1.0f);
}

// Advances one generation: reads cur plane, writes next plane.
// One thread handles one cell.
[[kernel]] void gol_step(device const ushort *cur [[buffer(0)]],
                         device ushort *next [[buffer(1)]],
                         constant Uniforms &u [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= u.gridW || gid.y >= u.gridH) return;

    int w = static_cast<int>(u.gridW);
    int h = static_cast<int>(u.gridH);
    int x = static_cast<int>(gid.x);
    int y = static_cast<int>(gid.y);

    int xL = (x > 0) ? x - 1 : w - 1;
    int xR = (x + 1 < w) ? x + 1 : 0;
    int yU = (y > 0) ? y - 1 : h - 1;
    int yD = (y + 1 < h) ? y + 1 : 0;

    uint ux = static_cast<uint>(x);
    uint rowU = static_cast<uint>(yU) * u.gridW;
    uint rowC = static_cast<uint>(y) * u.gridW;
    uint rowD = static_cast<uint>(yD) * u.gridW;

    int n = static_cast<int>(cur[rowC + static_cast<uint>(xL)] & 1u)
          + static_cast<int>(cur[rowC + ux] & 1u)
          + static_cast<int>(cur[rowC + static_cast<uint>(xR)] & 1u)
          + static_cast<int>(cur[rowU + static_cast<uint>(xL)] & 1u)
          + static_cast<int>(cur[rowU + ux] & 1u)
          + static_cast<int>(cur[rowU + static_cast<uint>(xR)] & 1u)
          + static_cast<int>(cur[rowD + static_cast<uint>(xL)] & 1u)
          + static_cast<int>(cur[rowD + ux] & 1u)
          + static_cast<int>(cur[rowD + static_cast<uint>(xR)] & 1u);
    n -= static_cast<int>(cur[rowC + ux] & 1u);

    uint idx = rowC + ux;
    ushort v = cur[idx];
    bool alive = (v & 1u) != 0u;
    uint age = (v >> 1) & 0x7FFFu;

    uint out = 0u;
    uchar birthBit = static_cast<uchar>((u.birth >> n) & 1u);
    uchar survBit = static_cast<uchar>((u.survival >> n) & 1u);
    if (birthBit || (alive && survBit)) {
        uint newAge = alive ? (age >= 32767u ? 32767u : age + 1u) : 1u;
        out = 1u | (newAge << 1);
    }
    next[idx] = static_cast<ushort>(out);
}

// Fades the persistent trail texture and adds the current cell color.
[[kernel]] void trail_step(texture2d<float, access::read> cell [[texture(0)]],
                           texture2d<float, access::read> trailRead [[texture(1)]],
                           texture2d<float, access::write> trailWrite [[texture(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint2 size = uint2(trailWrite.get_width(), trailWrite.get_height());
    if (gid.x >= size.x || gid.y >= size.y) return;
    float3 old = trailRead.read(gid).rgb * 0.94f;
    float4 cur = cell.read(gid);
    float3 added = cur.rgb * cur.a;
    float3 out = max(old, added);
    trailWrite.write(float4(out, 1.0f), gid);
}

// Zeroes the persistent trail texture.
[[kernel]] void trail_clear(texture2d<float, access::write> trail [[texture(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint2 size = uint2(trail.get_width(), trail.get_height());
    if (gid.x >= size.x || gid.y >= size.y) return;
    trail.write(float4(0.0f, 0.0f, 0.0f, 0.0f), gid);
}
