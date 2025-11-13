const std = @import("std");
const builtin = @import("builtin");

const tables = @import("tables");
const util = @import("utilities.zig");
const gf = @import("gf.zig");

const V = @Vector(32, u8);

pub const Engine = switch (builtin.target.cpu.arch) {
    else => @import("engines/Generic.zig"),
};

pub fn encode(
    allocator: std.mem.Allocator,
    original_count: u64,
    recovery_count: u64,
    original: []const []const u8,
) ![]const [64]u8 {
    if (original.len == 0) return error.TooFewOriginalShards;
    const shard_bytes = original[0].len;

    var encoder: Encoder = try .init(allocator, original_count, recovery_count, shard_bytes);
    errdefer encoder.deinit(allocator);

    for (original) |o| try encoder.addOriginalShard(o);

    const result = try encoder.encode();
    return result;
}

pub fn decode(
    allocator: std.mem.Allocator,
    original_count: u64,
    recovery_count: u64,
    original: []const ?[]const u8,
    recovery: []const ?[64]u8,
) ![]const [64]u8 {
    const shard_bytes = blk: {
        for (recovery) |rec| if (rec) |r| break :blk r.len;

        // no recovery shards
        var original_received_count: u64 = 0;
        for (original) |ori| {
            if (ori != null) original_received_count += 1;
        }

        // original data is complete
        if (original_received_count == original_count) {
            const result = try allocator.alloc([64]u8, original_count);
            errdefer allocator.free(result);

            for (0..original_count) |i| {
                @memcpy(&result[i], original[i].?);
            }

            return result;
        } else return error.NotEnoughShards;
    };

    var decoder: Decoder = try .init(allocator, original_count, recovery_count, shard_bytes);
    defer decoder.deinit(allocator);

    for (0..original_count) |i| {
        if (original[i]) |o| try decoder.addOriginalShard(i, o);
    }
    for (0..recovery_count) |i| {
        if (recovery[i]) |r| try decoder.addRecoveryShard(i, &r);
    }

    const data = try decoder.decode();

    const result = try allocator.alloc([64]u8, original_count);
    errdefer allocator.free(result);

    for (0..original_count) |i| {
        if (original[i]) |o|
            @memcpy(&result[i], o)
        else
            @memcpy(&result[i], &data[decoder.original_base_pos + i]);
    }

    return result;
}

const Encoder = struct {
    original_count: u64,
    recovery_count: u64,
    shard_bytes: usize,

    original_received_count: u64 = 0,
    shards: Shards,

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

            return .{
                .original_count = original_count,
                .recovery_count = recovery_count,
                .shard_bytes = shard_bytes,
                .shards = try .init(
                    allocator,
                    work_count,
                    try std.math.divCeil(u64, shard_bytes, 64),
                ),
            };
        } else {
            @panic("TODO");
        }
    }

    fn deinit(e: *Encoder, allocator: std.mem.Allocator) void {
        e.shards.deinit(allocator);
    }

    fn addOriginalShard(e: *Encoder, original_shard: []const u8) !void {
        if (e.original_received_count == e.original_count) return error.TooManyOriginalShards;
        if (original_shard.len != e.shard_bytes) return error.DifferentShardSize;

        e.shards.insert(e.original_received_count, original_shard);
        e.original_received_count += 1;
    }

    fn encode(e: *Encoder) ![]const [64]u8 {
        const shards = &e.shards;

        if (e.original_received_count != e.original_count) return error.TooFewOriginalShards;

        const chunk_size = try std.math.ceilPowerOfTwo(u64, e.recovery_count);

        // first chunk
        const first_count = @min(e.original_count, chunk_size);
        shards.zero(first_count, chunk_size);
        Engine.ifft(e, 0, chunk_size, first_count, chunk_size);

        if (e.original_count > chunk_size) {
            // full chunks
            var chunk_start = chunk_size;
            while (chunk_start + chunk_size < e.original_count) : (chunk_start += chunk_size) {
                Engine.ifft(e, chunk_start, chunk_size, chunk_size, chunk_start + chunk_size);
                const s0 = shards.data[0..chunk_size];
                const s1 = shards.data[chunk_start * shards.shard_length ..][0..chunk_size];
                util.xor(s0, s1);
            }

            // final partial chunk
            const last_count = e.original_count % chunk_size;
            if (last_count > 0) {
                shards.zero(chunk_start + last_count, shards.data.len);
                Engine.ifft(e, chunk_start, chunk_size, last_count, chunk_start + chunk_size);
                const s0 = shards.data[0..chunk_size];
                const s1 = shards.data[chunk_start * shards.shard_length ..][0..chunk_size];
                util.xor(s0, s1);
            }
        }

        Engine.fft(e, 0, chunk_size, e.recovery_count, 0);
        undoLastChunkEncoding(e, 0, e.recovery_count);

        return shards.data;
    }
};

const Decoder = struct {
    original_count: u64,
    recovery_count: u64,
    shard_bytes: usize,

    original_base_pos: u64,
    recovery_base_pos: u64 = 0,

    original_received_count: u64 = 0,
    recovery_received_count: u64 = 0,

    erasures: [gf.order]u16 = @splat(0),

    received: []bool,
    shards: Shards,

    fn init(
        allocator: std.mem.Allocator,
        original_count: u64,
        recovery_count: u64,
        shard_bytes: usize,
    ) !Decoder {
        const high_rate = try useHighRate(original_count, recovery_count);

        if (high_rate) {
            if (shard_bytes == 0 or shard_bytes & 1 != 0) return error.InvalidShardSize;

            const chunk_size = try std.math.ceilPowerOfTwo(u64, recovery_count);
            const work_count = try std.math.ceilPowerOfTwo(u64, chunk_size + original_count);

            var shards: Shards = try .init(
                allocator,
                work_count,
                try std.math.divCeil(u64, shard_bytes, 64),
            );
            errdefer shards.deinit(allocator);

            const received = try allocator.alloc(bool, original_count + recovery_count);
            errdefer allocator.free(received);
            @memset(received, false);

            return .{
                .original_count = original_count,
                .recovery_count = recovery_count,
                .shard_bytes = shard_bytes,
                .original_base_pos = chunk_size,
                .received = received,
                .shards = shards,
            };
        } else {
            @panic("TODO");
        }
    }

    fn deinit(d: *Decoder, allocator: std.mem.Allocator) void {
        allocator.free(d.received);
        d.shards.deinit(allocator);
    }

    fn addOriginalShard(d: *Decoder, index: u64, original_shard: []const u8) !void {
        const pos = d.original_base_pos + index;

        if (index >= d.original_count) {
            return error.InvalidShardIndex;
        } else if (d.received[pos]) return error.DuplicateShardIndex;
        if (d.original_received_count == d.original_count) return error.TooManyShards;
        if (original_shard.len != d.shard_bytes) return error.DifferentShardSize;

        d.shards.insert(pos, original_shard);
        d.original_received_count += 1;
        d.received[pos] = true;
    }

    fn addRecoveryShard(d: *Decoder, index: u64, recovery_shard: []const u8) !void {
        const pos = d.recovery_base_pos + index;

        if (index >= d.recovery_count) {
            return error.InvalidShardIndex;
        } else if (d.received[pos]) {
            return error.DuplicateShardIndex;
        } else if (d.recovery_received_count == d.recovery_count) {
            return error.TooManyShards;
        } else if (recovery_shard.len != d.shard_bytes)
            return error.DifferentShardSize;

        d.shards.insert(pos, recovery_shard);
        d.recovery_received_count += 1;
        d.received[pos] = true;
    }

    /// Performs polynomial interpolation in over GF to reconstruct missing shards.
    fn decode(d: *Decoder) ![][64]u8 {
        const shards = &d.shards;

        if (d.original_received_count + d.recovery_received_count < d.original_count)
            return error.NotEnoughShards;

        const chunk_size = try std.math.ceilPowerOfTwo(u64, d.recovery_count);
        const original_end = chunk_size + d.original_count;

        // mark missing recovery shards / erasures
        for (0..d.recovery_count) |i| {
            if (!d.received[i]) d.erasures[i] = 1;
        }

        @memset(d.erasures[d.recovery_count..chunk_size], 0);

        // mark missing original shreds
        for (chunk_size..original_end) |i| {
            if (!d.received[i]) d.erasures[i] = 1;
        }

        Engine.evalPoly(&d.erasures, original_end);

        // apply erasure masks to all chunks
        for (0..d.recovery_count) |i| {
            const chunk = shards.data[i * shards.shard_length ..][0..shards.shard_length];
            if (d.received[i]) Engine.mulScalar(chunk, d.erasures[i]) else @memset(chunk, @splat(0));
        }
        shards.zero(d.recovery_count, chunk_size);

        // original region
        for (chunk_size..original_end) |i| {
            const chunk = shards.data[i * shards.shard_length ..][0..shards.shard_length];
            if (d.received[i]) Engine.mulScalar(chunk, d.erasures[i]) else @memset(chunk, @splat(0));
        }
        shards.zero(original_end, shards.data.len);

        // convert from freq to time domain
        Engine.ifft(d, 0, shards.data.len, original_end, 0);

        // formal derivative (forney's algorithm)
        for (1..shards.data.len) |i| {
            // intCast is safe because i cannot be 0 nor usize max
            const width: u64 = @as(u64, 1) << @intCast(@ctz(i));
            const s0 = shards.data[(i - width) * shards.shard_length ..][0..width];
            const s1 = shards.data[i * shards.shard_length ..][0..width];
            util.xor(s0, s1);
        }

        // return to freq domain
        Engine.fft(d, 0, shards.data.len, original_end, 0);

        // restore the missing (erased) shards
        for (chunk_size..original_end) |i| if (!d.received[i]) {
            Engine.mulScalar(
                shards.data[i * shards.shard_length ..][0..shards.shard_length],
                gf.modulus - d.erasures[i],
            );
        };

        undoLastChunkEncoding(
            d,
            d.original_base_pos,
            d.original_base_pos + d.original_count,
        );

        return shards.data;
    }
};

fn undoLastChunkEncoding(e: anytype, start: usize, end: usize) void {
    const whole_chunk_count = e.shard_bytes / 64;
    const tail_len = e.shard_bytes % 64;

    if (tail_len == 0) return;

    for (start..end) |i| {
        var last_chunk = e.shards.data[i * e.shards.shard_length ..][0..e.shards.shard_length][whole_chunk_count];
        @memmove(last_chunk[tail_len / 2 ..], last_chunk[32..][0 .. tail_len / 2]);
    }
}

pub const Shards = struct {
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

test {
    _ = Engine;
}
