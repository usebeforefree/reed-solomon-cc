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

    fn ifft(e: *Encoder, pos: u64, size: u64, truncated_size: u64, skew_delta: u64) void {
        const S = struct {
            fn partial(x: [][64]u8, y: [][64]u8, log_m: u16) void {
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
        };

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
                        S.partial(s0, s1, log_m01);
                    }

                    if (log_m23 == gf.modulus) {
                        xor(s3, s2);
                    } else {
                        S.partial(s2, s3, log_m23);
                    }

                    // second layer
                    if (log_m02 == gf.modulus) {
                        xor(s2, s0);
                        xor(s3, s1);
                    } else {
                        S.partial(s0, s2, log_m02);
                        S.partial(s1, s3, log_m02);
                    }
                }
            }
            distance = distance_4;
            distance_4 <<= 2;
        }

        // FINAL ODD LAYER

        if (distance < size) {
            const log_m = tables.skew[distance + skew_delta - 1];

            const s0 = shards.data[0 .. (pos + distance) * shards.shard_length][0..shards.shard_length];
            const s1 = shards.data[(pos + distance) * shards.shard_length ..][0..shards.shard_length];

            if (log_m == gf.modulus) {
                xor(s0, s1);
            } else {
                S.partial(s0, s1, log_m);
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
        const prod_lo, const prod_hi = mul256(@bitCast(y_lo), @bitCast(y_hi), lut);
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
