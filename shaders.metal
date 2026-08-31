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
};

struct VSOut {
    float4 pos [[position]];
    float2 uv;
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

vertex VSOut vs_main(uint vid [[vertex_id]]) {
    VSOut o;
    o.pos = float4(kCornerPos[vid], 0.0f, 1.0f);
    o.uv = kCornerUV[vid];
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

float3 Viridis(float t) {
    t = clamp(t, 0.0f, 1.0f);
    float scaled = t * 9.0f;
    int i0 = (int)scaled;
    int i1 = min(i0 + 1, 9);
    float f = scaled - float(i0);
    return mix(kViridis[i0], kViridis[i1], f);
}

// Renders the active grid plane into a small cell texture.
// Cell packing: bit 0 = alive, bits 1..15 = age.
fragment float4 fs_main(VSOut in [[stage_in]],
                        device const ushort *cells [[buffer(0)]],
                        constant Uniforms &u [[buffer(1)]]) {
    const float4 bg = float4(0.04f, 0.05f, 0.08f, 1.0f);

    // Flip uv.y so grid row 0 is stored at the top of the cell texture.
    float2 cellF = float2(in.uv.x, 1.0f - in.uv.y) * float2((float)u.gridW, (float)u.gridH);
    int2 icell = int2(floor(cellF));
    icell = clamp(icell, int2(0, 0), int2((int)u.gridW - 1, (int)u.gridH - 1));
    uint idx = u.curOffset + (uint)icell.y * u.gridW + (uint)icell.x;
    ushort v = cells[idx];

    if ((v & 1u) == 0u) {
        return bg;
    }
    float age = float((v >> 1) & 0x7FFFu) / 32767.0f;
    return float4(Viridis(1.0f - age), 1.0f);
}

// Scales the small cell texture to the drawable and draws cell gaps.
fragment float4 fs_scale(VSOut in [[stage_in]],
                         texture2d<float> tex [[texture(0)]],
                         constant Uniforms &u [[buffer(0)]]) {
    const float4 bg = float4(0.04f, 0.05f, 0.08f, 1.0f);
    float2 cellF = float2(in.uv.x, 1.0f - in.uv.y) * float2((float)u.gridW, (float)u.gridH);
    float2 f = fract(cellF);
    const float gap = 0.16f;
    if (f.x < gap || f.x > 1.0f - gap || f.y < gap || f.y > 1.0f - gap) {
        return bg;
    }
    int2 icell = int2(floor(cellF));
    icell = clamp(icell, int2(0, 0), int2((int)u.gridW - 1, (int)u.gridH - 1));
    return tex.read(uint2(icell), 0);
}

// Advances one generation: reads cur plane, writes next plane.
// One thread handles one cell.
kernel void gol_step(device const ushort *cur [[buffer(0)]],
                     device ushort *next [[buffer(1)]],
                     constant Uniforms &u [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= u.gridW || gid.y >= u.gridH) return;

    int w = (int)u.gridW;
    int h = (int)u.gridH;
    int x = (int)gid.x;
    int y = (int)gid.y;

    int xL = (x > 0) ? x - 1 : w - 1;
    int xR = (x + 1 < w) ? x + 1 : 0;
    int yU = (y > 0) ? y - 1 : h - 1;
    int yD = (y + 1 < h) ? y + 1 : 0;

    uint ux = (uint)x;
    uint rowU = (uint)yU * u.gridW;
    uint rowC = (uint)y * u.gridW;
    uint rowD = (uint)yD * u.gridW;

    int n = (int)(cur[rowC + (uint)xL] & 1u)
          + (int)(cur[rowC + ux] & 1u)
          + (int)(cur[rowC + (uint)xR] & 1u)
          + (int)(cur[rowU + (uint)xL] & 1u)
          + (int)(cur[rowU + ux] & 1u)
          + (int)(cur[rowU + (uint)xR] & 1u)
          + (int)(cur[rowD + (uint)xL] & 1u)
          + (int)(cur[rowD + ux] & 1u)
          + (int)(cur[rowD + (uint)xR] & 1u);
    n -= (int)(cur[rowC + ux] & 1u);

    uint idx = rowC + ux;
    ushort v = cur[idx];
    bool alive = (v & 1u) != 0u;
    uint age = (v >> 1) & 0x7FFFu;

    uint out = 0u;
    uchar birthBit = (uchar)(u.birth >> n);
    uchar survBit = (uchar)(u.survival >> n);
    if (birthBit || (alive && survBit)) {
        uint newAge = alive ? (age >= 32767u ? 32767u : age + 1u) : 1u;
        out = 1u | (newAge << 1);
    }
    next[idx] = (ushort)out;
}
