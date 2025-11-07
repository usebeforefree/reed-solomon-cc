const std = @import("std");
const tables = @import("tables");
const gf = @import("gf.zig");

const SHARD_BYTES = 1024;

pub fn main() !void {
    const count = 256;

    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();

    var original: [count][]const u8 = undefined;
    defer for (original) |o| allocator.free(o);
    for (&original) |*a| {
        const slice = try allocator.alloc(u8, SHARD_BYTES);
        random.bytes(slice);
        a.* = slice;
    }

    const recovery = try encode(allocator, count, count, &original);
    _ = recovery;
}

fn encode(allocator: std.mem.Allocator, original_count: u64, recovery_count: u64, original: []const []const u8) !void {
    if (original.len == 0) return error.TooFewOriginalShards;
    const shard_bytes = original[0].len;

    var encoder: Encoder = try .init(allocator, original_count, recovery_count, shard_bytes);
    defer encoder.deinit(allocator);

    for (original) |o| try encoder.addOriginalShard(o);

    const result = try encoder.encode();
    _ = result;
}

const Encoder = struct {
    work: Work,

    const V = @Vector(32, u8);

    const Work = struct {
        original_count: u64,
        recovery_count: u64,
        shard_bytes: usize,

        original_received_count: u64,
        shards: Shards,

        fn deinit(w: *Work, allocator: std.mem.Allocator) void {
            w.shards.deinit(allocator);
        }
    };

    const Shards = struct {
        shard_count: u64,
        /// 64 byte chunks
        shard_length: u64,
        /// Slice of `shard_count * shard_length * 64` bytes.
        data: [][64]u8,

        fn init(allocator: std.mem.Allocator, shard_count: u64, shard_length: u64) !Shards {
            const data = try allocator.alloc([64]u8, shard_count * shard_length);
            errdefer allocator.free(data);
            @memset(data, @splat(0));

            return .{
                .shard_count = shard_count,
                .shard_length = shard_length,
                .data = data,
            };
        }

        fn deinit(s: *Shards, allocator: std.mem.Allocator) void {
            allocator.free(s.data);
        }

        fn insert(s: *Shards, index: u64, shard: []const u8) void {
            std.debug.assert(shard.len % 2 == 0);

            const whole_chunk_count = shard.len / 64;
            const tail_length = shard.len % 64;

            const source_chunks = shard[0 .. shard.len - tail_length];

            const dst = s.data[index * s.shard_length ..][0..s.shard_length];
            @memcpy(std.mem.sliceAsBytes(dst[0..whole_chunk_count]), source_chunks);

            if (tail_length > 0) {
                @panic("TODO");
            }
        }

        /// Zeroes shards from `start_index..end_index`.
        fn zero(s: *Shards, start_index: u64, end_index: u64) void {
            const start = start_index * s.shard_length;
            const end = end_index * s.shard_length;
            @memset(std.mem.sliceAsBytes(s.data[start..end]), 0);
        }
    };

    fn init(
        allocator: std.mem.Allocator,
        original_count: u64,
        recovery_count: u64,
        shard_bytes: usize,
    ) !Encoder {
        const high_rate = try useHighRate(original_count, recovery_count);

        if (high_rate) {
            if (shard_bytes == 0 or shard_bytes & 1 != 0) return error.InvalidShardSize;

            const chunk_size = try std.math.ceilPowerOfTwo(u64, recovery_count);
            const work_count = std.mem.alignForward(u64, original_count, chunk_size);

            const work: Work = .{
                .original_count = original_count,
                .recovery_count = recovery_count,
                .shard_bytes = shard_bytes,
                .original_received_count = 0,
                .shards = try .init(
                    allocator,
                    work_count,
                    try std.math.divCeil(u64, shard_bytes, 64),
                ),
            };

            return .{ .work = work };
        } else {
            @panic("TODO");
        }
    }

    fn deinit(e: *Encoder, allocator: std.mem.Allocator) void {
        e.work.deinit(allocator);
    }

    fn addOriginalShard(e: *Encoder, original_shard: []const u8) !void {
        const work = &e.work;

        if (work.original_received_count == work.original_count) return error.TooManyOriginalShards;
        if (original_shard.len != work.shard_bytes) return error.DifferentShardSize;

        work.shards.insert(work.original_received_count, original_shard);
        work.original_received_count += 1;
    }

    fn encode(e: *Encoder) !void {
        const work = &e.work;
        if (work.original_received_count != work.original_count) return error.TooFewOriginalShards;

        const chunk_size = try std.math.ceilPowerOfTwo(u64, work.recovery_count);

        // first chunk
        const first_count = @min(work.original_count, chunk_size);
        work.shards.zero(first_count, chunk_size);
        e.ifft(0, chunk_size, first_count, chunk_size);
    }

    fn ifftPartial(x: [][64]u8, y: [][64]u8, log_m: u16) void {
        const lut = tables.mul128[log_m];

        for (x, y) |*a, *b| {
            var x_lo: V = @bitCast(a[0..32].*);
            var x_hi: V = @bitCast(a[32..64].*);

            var y_lo: V = @bitCast(b[0..32].*);
            var y_hi: V = @bitCast(b[32..64].*);

            y_lo ^= x_lo;
            y_hi ^= x_hi;

            b[0..32].* = @bitCast(y_lo);
            b[32..64].* = @bitCast(y_hi);

            x_lo, x_hi = mulAdd256(x_lo, x_hi, y_lo, y_hi, lut);

            a[0..32].* = @bitCast(x_lo);
            a[32..64].* = @bitCast(x_hi);
        }
    }

    fn ifft(e: *Encoder, pos: u64, size: u64, truncated_size: u64, skew_delta: u64) void {
        const work = &e.work;
        const shards = &work.shards;

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
                        xor(s1, s0);
                    } else {
                        ifftPartial(s0, s1, log_m01);
                    }

                    if (log_m23 == gf.modulus) {
                        xor(s3, s2);
                    } else {
                        ifftPartial(s2, s3, log_m23);
                    }

                    // second layer
                    if (log_m02 == gf.modulus) {
                        xor(s2, s0);
                        xor(s3, s1);
                    } else {
                        ifftPartial(s0, s2, log_m02);
                        ifftPartial(s1, s3, log_m02);
                    }
                }
            }
            distance = distance_4;
            distance_4 <<= 2;
        }

        // FINAL ODD LAYER

        if (distance < size) {
            const log_m = tables.skew[distance + skew_delta - 1];

            if (log_m == gf.modulus) {
                const s0 = shards.data[(pos + distance) * shards.shard_length ..][0..shards.shard_length];
                const s1 = shards.data[pos * shards.shard_length ..][0..shards.shard_length];
                xor(s0, s1);
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

    fn xor(a: [][64]u8, b: [][64]u8) void {
        std.debug.assert(a.len == b.len);
        for (a, b) |*ac, bc| {
            for (ac, bc) |*x, y| x.* ^= y;
        }
    }

    fn mul256(lo: V, hi: V, lut: tables.Lut) struct { V, V } {
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

    fn mulAdd256(x_lo: V, x_hi: V, y_lo: V, y_hi: V, lut: tables.Lut) struct { V, V } {
        const prod_lo, const prod_hi = mul256(y_lo, y_hi, lut);
        return .{
            x_lo ^ prod_lo,
            x_hi ^ prod_hi,
        };
    }
};

fn useHighRate(original: u64, recovery: u64) !bool {
    if (original > gf.order or recovery > gf.order) return error.UnsupportedShardCount;

    const original_pow2 = try std.math.ceilPowerOfTwo(u64, original);
    const recovery_pow2 = try std.math.ceilPowerOfTwo(u64, recovery);

    const smaller = @min(original_pow2, recovery_pow2);
    const larger = @max(original, recovery);

    if (original == 0 or recovery == 0 or smaller + larger > gf.order) {
        return error.UnsupportedShardCount;
    }

    return switch (std.math.order(original_pow2, recovery_pow2)) {
        .lt => false,
        .gt => true,
        .eq => original <= recovery,
    };
}

test "Encoder.ifftPartial" {
    try ifftPartialTest(
        @constCast(&[2][64]u8{ .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63 }, .{ 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127 } }),
        @constCast(&[2][64]u8{ .{ 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191 }, .{ 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255 } }),
        56797,
        @constCast(&[2][64]u8{ .{ 21, 20, 23, 22, 17, 16, 19, 18, 29, 28, 31, 30, 25, 24, 27, 26, 5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10, 230, 231, 228, 229, 226, 227, 224, 225, 238, 239, 236, 237, 234, 235, 232, 233, 246, 247, 244, 245, 242, 243, 240, 241, 254, 255, 252, 253, 250, 251, 248, 249 }, .{ 85, 84, 87, 86, 81, 80, 83, 82, 93, 92, 95, 94, 89, 88, 91, 90, 69, 68, 71, 70, 65, 64, 67, 66, 77, 76, 79, 78, 73, 72, 75, 74, 166, 167, 164, 165, 162, 163, 160, 161, 174, 175, 172, 173, 170, 171, 168, 169, 182, 183, 180, 181, 178, 179, 176, 177, 190, 191, 188, 189, 186, 187, 184, 185 } }),
        @constCast(&[2][64]u8{ @splat(128), @splat(128) }),
    );

    try ifftPartialTest(
        @constCast(&[2][64]u8{ .{ 21, 20, 23, 22, 17, 16, 19, 18, 29, 28, 31, 30, 25, 24, 27, 26, 5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10, 230, 231, 228, 229, 226, 227, 224, 225, 238, 239, 236, 237, 234, 235, 232, 233, 246, 247, 244, 245, 242, 243, 240, 241, 254, 255, 252, 253, 250, 251, 248, 249 }, .{ 85, 84, 87, 86, 81, 80, 83, 82, 93, 92, 95, 94, 89, 88, 91, 90, 69, 68, 71, 70, 65, 64, 67, 66, 77, 76, 79, 78, 73, 72, 75, 74, 166, 167, 164, 165, 162, 163, 160, 161, 174, 175, 172, 173, 170, 171, 168, 169, 182, 183, 180, 181, 178, 179, 176, 177, 190, 191, 188, 189, 186, 187, 184, 185 } }),
        @constCast(&[2][64]u8{ .{ 27, 26, 25, 24, 31, 30, 29, 28, 19, 18, 17, 16, 23, 22, 21, 20, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30 }, .{ 91, 90, 89, 88, 95, 94, 93, 92, 83, 82, 81, 80, 87, 86, 85, 84, 75, 74, 73, 72, 79, 78, 77, 76, 67, 66, 65, 64, 71, 70, 69, 68, 65, 64, 67, 66, 69, 68, 71, 70, 73, 72, 75, 74, 77, 76, 79, 78, 81, 80, 83, 82, 85, 84, 87, 86, 89, 88, 91, 90, 93, 92, 95, 94 } }),
        17476,
        @constCast(&[2][64]u8{ .{ 142, 143, 140, 141, 138, 139, 136, 137, 134, 135, 132, 133, 130, 131, 128, 129, 158, 159, 156, 157, 154, 155, 152, 153, 150, 151, 148, 149, 146, 147, 144, 145, 71, 70, 69, 68, 67, 66, 65, 64, 79, 78, 77, 76, 75, 74, 73, 72, 87, 86, 85, 84, 83, 82, 81, 80, 95, 94, 93, 92, 91, 90, 89, 88 }, .{ 206, 207, 204, 205, 202, 203, 200, 201, 198, 199, 196, 197, 194, 195, 192, 193, 222, 223, 220, 221, 218, 219, 216, 217, 214, 215, 212, 213, 210, 211, 208, 209, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 24 } }),
        @constCast(&[2][64]u8{ .{ 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231 }, .{ 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231 } }),
    );
}

fn ifftPartialTest(
    x: [][64]u8,
    y: [][64]u8,
    log_m: u16,
    expected_x: [][64]u8,
    expected_y: [][64]u8,
) !void {
    Encoder.ifftPartial(x, y, log_m);

    try std.testing.expect(std.mem.eql(u8, &x[0], &expected_x[0]));
    try std.testing.expect(std.mem.eql(u8, &y[0], &expected_y[0]));
    try std.testing.expect(std.mem.eql(u8, &x[1], &expected_x[1]));
    try std.testing.expect(std.mem.eql(u8, &y[1], &expected_y[1]));
}

test "Encoder.mulAdd256" {
    try mulAdd256Test(@Vector(4, u64){ 506097522914230528, 1084818905618843912, 1663540288323457296, 2242261671028070680 }, @Vector(4, u64){ 2820983053732684064, 3399704436437297448, 3978425819141910832, 4557147201846524216 }, @Vector(4, u64){ 9259542123273814144, 9259542123273814144, 9259542123273814144, 9259542123273814144 }, @Vector(4, u64){ 9259542123273814144, 9259542123273814144, 9259542123273814144, 9259542123273814144 }, 30583, @Vector(4, u64){ 2025808526283708955, 1447087143579095571, 868365760874482187, 289644378169868803 }, @Vector(4, u64){ 434320308619640833, 1013041691324254217, 1591763074028867601, 2170484456733480985 });
}

fn mulAdd256Test(
    x_lo: @Vector(4, u64),
    x_hi: @Vector(4, u64),
    y_lo: @Vector(4, u64),
    y_hi: @Vector(4, u64),
    table_index: usize,
    expected_lo: @Vector(4, u64),
    expected_hi: @Vector(4, u64),
) !void {
    const x_lo_res, const x_hi_res = Encoder.mulAdd256(
        @bitCast(x_lo),
        @bitCast(x_hi),
        @bitCast(y_lo),
        @bitCast(y_hi),
        tables.mul128[table_index],
    );

    const expected_prod_lo: Encoder.V = @bitCast(expected_lo);
    const expected_prod_hi: Encoder.V = @bitCast(expected_hi);

    try std.testing.expectEqual(x_lo_res, expected_prod_lo);
    try std.testing.expectEqual(x_hi_res, expected_prod_hi);
}

test "Encoder.mul256" {
    try mul256Test(@Vector(4, u64){ 9259542123273814144, 9259542123273814144, 9259542123273814144, 9259542123273814144 }, @Vector(4, u64){ 9259542123273814144, 9259542123273814144, 9259542123273814144, 9259542123273814144 }, 30583, @Vector(4, u64){ 1953184666628070171, 1953184666628070171, 1953184666628070171, 1953184666628070171 }, @Vector(4, u64){ 2387225703656530209, 2387225703656530209, 2387225703656530209, 2387225703656530209 });

    try mul256Test(@Vector(4, u64){ 1012762419733073422, 1012762419733073422, 1012762419733073422, 1012762419733073422 }, @Vector(4, u64){ 16710579925595711463, 16710579925595711463, 16710579925595711463, 16710579925595711463 }, 17476, @Vector(4, u64){ 11212726789901884315, 11212726789901884315, 11212726789901884315, 11212726789901884315 }, @Vector(4, u64){ 11646767826930344353, 11646767826930344353, 11646767826930344353, 11646767826930344353 });

    try mul256Test(@Vector(4, u64){ 9259542123273814144, 9259542123273814144, 9259542123273814144, 9259542123273814144 }, @Vector(4, u64){ 9259542123273814144, 9259542123273814144, 9259542123273814144, 9259542123273814144 }, 56797, @Vector(4, u64){ 1519143629599610133, 1519143629599610133, 1519143629599610133, 1519143629599610133 }, @Vector(4, u64){ 14323354221939181254, 14323354221939181254, 14323354221939181254, 14323354221939181254 });

    try mul256Test(@Vector(4, u64){ 0, 0, 0, 0 }, @Vector(4, u64){ 0, 0, 0, 0 }, 34952, @Vector(4, u64){ 0, 0, 0, 0 }, @Vector(4, u64){ 0, 0, 0, 0 });
}

fn mul256Test(
    y_lo: @Vector(4, u64),
    y_hi: @Vector(4, u64),
    table_index: usize,
    expected_lo: @Vector(4, u64),
    expected_hi: @Vector(4, u64),
) !void {
    const prod_lo, const prod_hi = Encoder.mul256(
        @bitCast(y_lo),
        @bitCast(y_hi),
        tables.mul128[table_index],
    );

    const expected_prod_lo: Encoder.V = @bitCast(expected_lo);
    const expected_prod_hi: Encoder.V = @bitCast(expected_hi);

    try std.testing.expectEqual(prod_lo, expected_prod_lo);
    try std.testing.expectEqual(prod_hi, expected_prod_hi);
}
