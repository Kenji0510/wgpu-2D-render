struct Uniforms {
    resolution: vec2<f32>,
};
@group(0) @binding(0)
var<uniform> u: Uniforms;

struct TimeUniform {
    time: f32,
};
@group(0) @binding(1)
var<uniform> u_time: TimeUniform;

// 頂点→フラグメント受け渡し用
struct VSOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VSOut {
    let coords = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0),  // 左下
        vec2( 1.0, -1.0),  // 右下
        vec2(-1.0,  1.0),  // 左上
        // 第二三角形
        vec2(-1.0,  1.0),  // 左上
        vec2( 1.0, -1.0),  // 右下
        vec2( 1.0,  1.0),  // 右上
    );
    var out: VSOut;
    // クリップ空間へ
    out.position = vec4<f32>(coords[idx], 0.0, 1.0);
    return out;
}

// // フラグメント入力
// struct FSIn {
//     @location(0)       color:         vec3<f32>,
//     @builtin(position) frag_coord: vec4<f32>,
// };

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    var t = (frag_coord.xy * 2.0 - u.resolution.xy) / u.resolution.y;

    var col = vec3(1.0, 2.0, 3.0);

    var d = length(t.xy);
    d = sin(d * 8.0 + u_time.time) / 8.0;
    d = abs(d);

    // d = smoothstep(0.0, 0.1, d);
    d = 0.02 / d;

    col *= d;

    // return vec4(t.x, t.y, 0.0, 1.0);
    return vec4(col, 1.0);
}
