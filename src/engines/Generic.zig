//! An engine that implements the fft and ifft using generic `@Vector`s.

const std = @import("std");
const tables = @import("tables");

const walsh_hadamard = @import("../walsh_hadamard.zig");

const root = @import("../root.zig");
const gf = @import("../gf.zig");
const utils = @import("../utilities.zig");

const V = @Vector(32, u8);

pub fn fft(a: anytype, pos: u64, size: u64, truncated_size: u64, skew_delta: u64) void {
    const shards: *const root.Shards = &a.shards;

    var distance = size >> 2;
    var distance_4 = size;
    while (distance != 0) {
        var r: u64 = 0;
        while (r < truncated_size) : (r += distance_4) {
            const base = r + distance + skew_delta - 1;

            const log_m01 = tables.skew[base + distance * 0];
            const log_m02 = tables.skew[base + distance * 1];
            const log_m23 = tables.skew[base + distance * 2];

            for (r..r + distance) |i| {
                const position = pos + i;

                const s0 = shards.data[(position + distance * 0) * shards.shard_length ..][0..shards.shard_length];
                const s1 = shards.data[(position + distance * 1) * shards.shard_length ..][0..shards.shard_length];
                const s2 = shards.data[(position + distance * 2) * shards.shard_length ..][0..shards.shard_length];
                const s3 = shards.data[(position + distance * 3) * shards.shard_length ..][0..shards.shard_length];

                // first layer
                if (log_m02 == gf.modulus) {
                    utils.xor(s2, s0);
                    utils.xor(s3, s1);
                } else {
                    fftPartial(s0, s2, log_m02);
                    fftPartial(s1, s3, log_m02);
                }

                // second layer
                if (log_m01 == gf.modulus) {
                    utils.xor(s1, s0);
                } else {
                    fftPartial(s0, s1, log_m01);
                }

                if (log_m23 == gf.modulus) {
                    utils.xor(s3, s2);
                } else {
                    fftPartial(s2, s3, log_m23);
                }
            }
        }
        distance_4 = distance;
        distance >>= 2;
    }

    if (distance_4 == 2) {
        var r: usize = 0;
        while (r < truncated_size) : (r += 2) {
            const log_m = tables.skew[r + skew_delta];
            const s0 = shards.data[(pos + r) * shards.shard_length ..][0..shards.shard_length];
            const s1 = shards.data[(pos + r + 1) * shards.shard_length ..][0..shards.shard_length];

            if (log_m == gf.modulus) {
                utils.xor(s1, s0);
            } else {
                fftPartial(s0, s1, log_m);
            }
        }
    }
}

pub fn ifft(a: anytype, pos: u64, size: u64, truncated_size: u64, skew_delta: u64) void {
    const shards: *const root.Shards = &a.shards;

    var distance: u64 = 1;
    var distance_4: u64 = 4;
    while (distance_4 <= size) {
        var r: u64 = 0;
        while (r < truncated_size) : (r += distance_4) {
            const base = r + distance + skew_delta - 1;

            const log_m01 = tables.skew[base + distance * 0];
            const log_m02 = tables.skew[base + distance * 1];
            const log_m23 = tables.skew[base + distance * 2];

            for (r..r + distance) |i| {
                const position = pos + i;

                const s0 = shards.data[(position + distance * 0) * shards.shard_length ..][0..shards.shard_length];
                const s1 = shards.data[(position + distance * 1) * shards.shard_length ..][0..shards.shard_length];
                const s2 = shards.data[(position + distance * 2) * shards.shard_length ..][0..shards.shard_length];
                const s3 = shards.data[(position + distance * 3) * shards.shard_length ..][0..shards.shard_length];

                // first layer
                if (log_m01 == gf.modulus) {
                    utils.xor(s1, s0);
                } else {
                    ifftPartial(s0, s1, log_m01);
                }

                if (log_m23 == gf.modulus) {
                    utils.xor(s3, s2);
                } else {
                    ifftPartial(s2, s3, log_m23);
                }

                // second layer
                if (log_m02 == gf.modulus) {
                    utils.xor(s2, s0);
                    utils.xor(s3, s1);
                } else {
                    ifftPartial(s0, s2, log_m02);
                    ifftPartial(s1, s3, log_m02);
                }
            }
        }
        distance = distance_4;
        distance_4 <<= 2;
    }

    // final odd layer

    if (distance < size) {
        const log_m = tables.skew[distance + skew_delta - 1];

        if (log_m == gf.modulus) {
            const s0 = shards.data[(pos + distance) * shards.shard_length ..][0..distance];
            const s1 = shards.data[pos * shards.shard_length ..][0..distance];
            utils.xor(s0, s1);
        } else {
            for (0..distance) |i| {
                // TODO simplify this slicing
                const s0 = shards.data[0 .. (pos + distance) * shards.shard_length][(pos + i) * shards.shard_length ..][0..shards.shard_length];
                const s1 = shards.data[(pos + distance) * shards.shard_length ..][i * shards.shard_length ..][0..shards.shard_length];
                ifftPartial(s0, s1, log_m);
            }
        }
    }
}

fn fftPartial(x: [][64]u8, y: [][64]u8, log_m: u16) void {
    const lut = tables.mul_128[log_m];

    for (x, y) |*a, *b| {
        var x_lo: V = @bitCast(a[0..32].*);
        var x_hi: V = @bitCast(a[32..64].*);

        var y_lo: V = @bitCast(b[0..32].*);
        var y_hi: V = @bitCast(b[32..64].*);

        x_lo, x_hi = mulAdd(x_lo, x_hi, y_lo, y_hi, lut);

        a[0..32].* = @bitCast(x_lo);
        a[32..64].* = @bitCast(x_hi);

        y_lo ^= x_lo;
        y_hi ^= x_hi;

        b[0..32].* = @bitCast(y_lo);
        b[32..64].* = @bitCast(y_hi);
    }
}

fn ifftPartial(x: [][64]u8, y: [][64]u8, log_m: u16) void {
    const lut = tables.mul_128[log_m];

    for (x, y) |*a, *b| {
        var x_lo: V = @bitCast(a[0..32].*);
        var x_hi: V = @bitCast(a[32..64].*);

        var y_lo: V = @bitCast(b[0..32].*);
        var y_hi: V = @bitCast(b[32..64].*);

        y_lo ^= x_lo;
        y_hi ^= x_hi;

        b[0..32].* = @bitCast(y_lo);
        b[32..64].* = @bitCast(y_hi);

        x_lo, x_hi = mulAdd(x_lo, x_hi, y_lo, y_hi, lut);

        a[0..32].* = @bitCast(x_lo);
        a[32..64].* = @bitCast(x_hi);
    }
}

/// Evalute the erasure locator polynomial across all positions in the field.
///
/// Our goal is to compute, for each symbol position, the contribution
/// of all known erasures according to the erasure locator polynomial.
///
/// This implementation uses multiple rounds of walsh-hadamard transformations.
pub fn evalPoly(erasures: *[gf.order]u16, truncated_size: u64) void {
    // move the erasure indicators into the "spectral" domain,
    // where linear operations (adds/XORs) become commponent-wise.
    walsh_hadamard.fwht(erasures, truncated_size);
    // multiple each transformed coefficient by a precomputed factor
    // (log-domain walsh weight). this scales each spectral component
    // to reflect the polynomial evaluation at all field points.
    for (erasures, tables.log_walsh) |*e, factor| {
        const product = @as(u32, e.*) * @as(u32, factor);
        e.* = utils.addMod(product & 0xFFFF, product >> gf.bits);
    }
    // perform an inverse walsh-hadamard transformation back from
    // spectral domain to the standard domain, yielding the evaluated
    // erasure polynomial across all field elements.
    walsh_hadamard.fwht(erasures, gf.order);
}

/// Mutliplies a slice of 64-byte chunks by a finite field scalar `log_m`.
///
/// Each 64-byte chunk represents 512 bits (or 64 GF(2^8) symbols).
pub fn mulScalar(x: [][64]u8, log_m: u16) void {
    const lut = tables.mul_128[log_m];

    for (x) |*chunk| {
        var x_lo: V = @bitCast(chunk[0..32].*);
        var x_hi: V = @bitCast(chunk[32..64].*);

        x_lo, x_hi = mul(x_lo, x_hi, lut);

        chunk[0..32].* = @bitCast(x_lo);
        chunk[32..64].* = @bitCast(x_hi);
    }
}

/// Fused multiply-add operation in GF(2^8).
fn mulAdd(x_lo: V, x_hi: V, y_lo: V, y_hi: V, lut: tables.Lut) struct { V, V } {
    const prod_lo, const prod_hi = mul(y_lo, y_hi, lut);
    return .{
        x_lo ^ prod_lo,
        x_hi ^ prod_hi,
    };
}

fn mul(lo: V, hi: V, lut: tables.Lut) struct { V, V } {
    var prod_lo: V = undefined;
    var prod_hi: V = undefined;

    const clr_mask: V = @splat(0x0f);

    const data_0 = lo & clr_mask;
    prod_lo = shuffle256epi8(broadcastU128(lut[0][0]), data_0);
    prod_hi = shuffle256epi8(broadcastU128(lut[1][0]), data_0);

    const data_1 = (lo >> @splat(4)) & clr_mask;
    prod_lo ^= shuffle256epi8(broadcastU128(lut[0][1]), data_1);
    prod_hi ^= shuffle256epi8(broadcastU128(lut[1][1]), data_1);

    const data_2 = hi & clr_mask;
    prod_lo ^= shuffle256epi8(broadcastU128(lut[0][2]), data_2);
    prod_hi ^= shuffle256epi8(broadcastU128(lut[1][2]), data_2);

    const data_3 = (hi >> @splat(4)) & clr_mask;
    prod_lo ^= shuffle256epi8(broadcastU128(lut[0][3]), data_3);
    prod_hi ^= shuffle256epi8(broadcastU128(lut[1][3]), data_3);

    return .{ prod_lo, prod_hi };
}

// TODO optimize
fn broadcastU128(x: u128) V {
    const lo: [16]u8 = @bitCast(x);
    var res: V = undefined;

    for (0..16) |i| {
        res[i] = lo[i];
        res[i + 16] = lo[i];
    }

    return res;
}

// TODO optimize
fn shuffle256epi8(a: V, b: V) V {
    var res: V = @splat(0);
    for (0..16) |i| {
        if ((b[i] & 0x80) == 0)
            res[i] = a[b[i] % 16];

        if ((b[i + 16] & 0x80) == 0)
            res[i + 16] = a[b[i + 16] % 16 + 16];
    }
    return res;
}

test ifftPartial {
    var arr: [2][64]u8 = .{
        .{ 21, 20, 23, 22, 17, 16, 19, 18, 29, 28, 31, 30, 25, 24, 27, 26, 5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10, 230, 231, 228, 229, 226, 227, 224, 225, 238, 239, 236, 237, 234, 235, 232, 233, 246, 247, 244, 245, 242, 243, 240, 241, 254, 255, 252, 253, 250, 251, 248, 249 },
        .{ 85, 84, 87, 86, 81, 80, 83, 82, 93, 92, 95, 94, 89, 88, 91, 90, 69, 68, 71, 70, 65, 64, 67, 66, 77, 76, 79, 78, 73, 72, 75, 74, 166, 167, 164, 165, 162, 163, 160, 161, 174, 175, 172, 173, 170, 171, 168, 169, 182, 183, 180, 181, 178, 179, 176, 177, 190, 191, 188, 189, 186, 187, 184, 185 },
    };
    {
        var iota: [256]u8 = std.simd.iota(u8, 256);
        try ifftPartialTest(
            @ptrCast(iota[0..128]),
            @ptrCast(iota[128..256]),
            0xDDDD,
            &arr,
            &[2][64]u8{ @splat(128), @splat(128) },
        );
    }

    {
        var input: [2][64]u8 = .{
            .{ 27, 26, 25, 24, 31, 30, 29, 28, 19, 18, 17, 16, 23, 22, 21, 20, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30 },
            .{ 91, 90, 89, 88, 95, 94, 93, 92, 83, 82, 81, 80, 87, 86, 85, 84, 75, 74, 73, 72, 79, 78, 77, 76, 67, 66, 65, 64, 71, 70, 69, 68, 65, 64, 67, 66, 69, 68, 71, 70, 73, 72, 75, 74, 77, 76, 79, 78, 81, 80, 83, 82, 85, 84, 87, 86, 89, 88, 91, 90, 93, 92, 95, 94 },
        };
        const expected_y: [64]u8 = @as([32]u8, @splat(0xE)) ++ @as([32]u8, @splat(0xE7));
        try ifftPartialTest(
            &arr,
            &input,
            0x4444,
            &.{
                .{ 142, 143, 140, 141, 138, 139, 136, 137, 134, 135, 132, 133, 130, 131, 128, 129, 158, 159, 156, 157, 154, 155, 152, 153, 150, 151, 148, 149, 146, 147, 144, 145, 71, 70, 69, 68, 67, 66, 65, 64, 79, 78, 77, 76, 75, 74, 73, 72, 87, 86, 85, 84, 83, 82, 81, 80, 95, 94, 93, 92, 91, 90, 89, 88 },
                .{ 206, 207, 204, 205, 202, 203, 200, 201, 198, 199, 196, 197, 194, 195, 192, 193, 222, 223, 220, 221, 218, 219, 216, 217, 214, 215, 212, 213, 210, 211, 208, 209, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 24 },
            },
            &.{ expected_y, expected_y },
        );
    }
}

fn ifftPartialTest(
    x: [][64]u8,
    y: [][64]u8,
    log_m: u16,
    expected_x: []const [64]u8,
    expected_y: []const [64]u8,
) !void {
    ifftPartial(x, y, log_m);

    try std.testing.expectEqualSlices(u8, &expected_x[0], &x[0]);
    try std.testing.expectEqualSlices(u8, &expected_y[0], &y[0]);
    try std.testing.expectEqualSlices(u8, &expected_x[1], &x[1]);
    try std.testing.expectEqualSlices(u8, &expected_y[1], &y[1]);
}

test mulAdd {
    const x_lo_res, const x_hi_res = mulAdd(
        @bitCast(@Vector(4, u64){ 506097522914230528, 1084818905618843912, 1663540288323457296, 2242261671028070680 }),
        @bitCast(@Vector(4, u64){ 2820983053732684064, 3399704436437297448, 3978425819141910832, 4557147201846524216 }),
        @splat(0x80),
        @splat(0x80),
        tables.mul_128[0x7777],
    );

    const expected_prod_lo: V = @bitCast(@Vector(4, u64){ 2025808526283708955, 1447087143579095571, 868365760874482187, 289644378169868803 });
    const expected_prod_hi: V = @bitCast(@Vector(4, u64){ 434320308619640833, 1013041691324254217, 1591763074028867601, 2170484456733480985 });

    try std.testing.expectEqual(x_lo_res, expected_prod_lo);
    try std.testing.expectEqual(x_hi_res, expected_prod_hi);
}

test mul {
    {
        const prod_lo, const prod_hi = mul(
            @splat(0x80),
            @splat(0x80),
            tables.mul_128[0x7777],
        );

        const expected_prod_lo: V = @splat(0x1B);
        const expected_prod_hi: V = @splat(0x21);

        try std.testing.expectEqual(prod_lo, expected_prod_lo);
        try std.testing.expectEqual(prod_hi, expected_prod_hi);
    }
    {
        const prod_lo, const prod_hi = mul(
            @splat(0x0E),
            @splat(0xE7),
            tables.mul_128[0x4444],
        );

        const expected_prod_lo: V = @splat(0x9B);
        const expected_prod_hi: V = @splat(0xA1);

        try std.testing.expectEqual(prod_lo, expected_prod_lo);
        try std.testing.expectEqual(prod_hi, expected_prod_hi);
    }
    {
        const prod_lo, const prod_hi = mul(
            @splat(0x80),
            @splat(0x80),
            tables.mul_128[0xDDDD],
        );

        const expected_prod_lo: V = @splat(0x15);
        const expected_prod_hi: V = @splat(0xC6);

        try std.testing.expectEqual(prod_lo, expected_prod_lo);
        try std.testing.expectEqual(prod_hi, expected_prod_hi);
    }
    {
        const prod_lo, const prod_hi = mul(
            @splat(0),
            @splat(0),
            tables.mul_128[0x8888],
        );

        const expected_prod_lo: V = @splat(0);
        const expected_prod_hi: V = @splat(0);

        try std.testing.expectEqual(prod_lo, expected_prod_lo);
        try std.testing.expectEqual(prod_hi, expected_prod_hi);
    }
}
