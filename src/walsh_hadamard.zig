/// Design note: should not import nor depend on "tables" module, since tables.zig imports it.
const gf = @import("gf.zig");
const utils = @import("utilities.zig");

/// Performs an in-place fast walsh-hadamard transform (FWHT) over GF(2^m).
///
/// This transformation is self-inverse, meaning another FWHT restores the original data.
///
/// This implementation performs the FWHT iteratively in stages, applying 4-point
/// "butterfly" transforms (fwht4) at increasing distances between elements. Each
/// fwht4 internally applies two 2-point butterflies (fwht2) to form the full
/// 4-element combination.
///
/// The pattern of strides and recursive butterflies ensures logarithmic complexity
/// O(N log N), same as radix-4 FFT.
pub fn fwht(data: *[gf.order]u16, m: u64) void {
    var dist: u64 = 1; // distance between elements in the first stage
    var stride: u64 = 4; // radix-4 stride increment

    while (stride <= gf.order) {
        var r: u64 = 0;
        while (r < m) : (r += stride) {
            for (r..r + dist) |offset| {
                fwht4(data, @truncate(offset), @truncate(dist));
            }
        }
        dist = stride;
        stride *= 4; // next stage operates at 4x the stride
    }
}

/// Perform a 4-point FWHT butterfly stage at a given offset and stride.
///
/// The 4-point structure combines values spaced at `stride` apart,
/// computing pairwise sums and differences in a recursive pattern.
fn fwht4(data: *[gf.order]u16, offset: u64, stride: u64) void {
    // indices for this butterfly group
    const x0: u64 = offset + stride * 0;
    const x1: u64 = offset + stride * 1;
    const x2: u64 = offset + stride * 2;
    const x3: u64 = offset + stride * 3;

    // pairwise butterflies
    const s0, const d0 = fwht2(data[x0], data[x1]);
    const s1, const d1 = fwht2(data[x2], data[x3]);
    const s2, const d2 = fwht2(s0, s1);
    const s3, const d3 = fwht2(d0, d1);

    data[x0] = s2;
    data[x1] = s3;
    data[x2] = d2;
    data[x3] = d3;
}

/// 2-point Walsh-Hadamard butterfly.
///
/// Each butterfly computes the field sum and difference of two points.
fn fwht2(a: u16, b: u16) struct { u16, u16 } {
    const sum = utils.addMod(a, b);
    const dif = utils.subMod(a, b);
    return .{ sum, dif };
}
