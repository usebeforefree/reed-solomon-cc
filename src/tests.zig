const std = @import("std");
const testing = std.testing;

const reedsol = @import("reedsol");
const encode = reedsol.encode;
const decode = reedsol.decode;

fn encodeDecodeCycle(
    allocator: std.mem.Allocator,
    comptime count: usize,
    comptime SHARD_BYTES: usize,
    input: [count * SHARD_BYTES]u8,
    original_shards_present: [count]bool,
    recovery_shards_present: [count]bool,
) !void {
    var original: [count][]const u8 = undefined;
    for (&original, 0..) |*shard, i| {
        const start = i * SHARD_BYTES;
        const end = start + SHARD_BYTES;
        shard.* = input[start..end];
    }

    const recovery = try encode(
        allocator,
        count,
        count,
        &original,
    );
    defer allocator.free(recovery);

    var original_recovery_shards: [count]?[]const u8 = undefined;
    for (0..count) |i| {
        original_recovery_shards[i] = if (original_shards_present[i])
            original[i]
        else
            null;
    }

    var recovery_shards: [count]?[SHARD_BYTES]u8 = undefined;
    for (0..count) |i| {
        recovery_shards[i] = if (recovery_shards_present[i])
            recovery[i]
        else
            null;
    }

    const recovered = try decode(
        allocator,
        count,
        count,
        &original_recovery_shards,
        &recovery_shards,
    );
    defer allocator.free(recovered);

    for (0..count) |i| for (0..SHARD_BYTES) |j| {
        try std.testing.expectEqual(original[i][j], recovered[i][j]);
    };
}

test "encode and decode cycles with all possible shard combinations" {
    const allocator = std.testing.allocator;
    const count = 5;
    const SHARD_BYTES = 64;

    var input: [SHARD_BYTES * count]u8 = undefined;
    for (0..input.len) |i| input[i] = @intCast(i % 256);

    var original_shards_present: [count]bool = undefined;
    var recovery_shards_present: [count]bool = undefined;

    const total_combinations = 1 << (count * 2);

    for (0..total_combinations) |mask| {
        original_shards_present = @splat(true);
        recovery_shards_present = @splat(true);

        for (0..count * 2) |i| {
            if ((mask & (@as(usize, 1) << @as(u6, @intCast(i)))) != 0) {
                if (i < count) {
                    original_shards_present[i] = false;
                } else {
                    recovery_shards_present[i - count] = false;
                }
            }
        }

        const result = encodeDecodeCycle(
            allocator,
            count,
            SHARD_BYTES,
            input,
            original_shards_present,
            recovery_shards_present,
        );
        if (@popCount(mask) <= count)
            try result
        else
            try testing.expectError(error.NotEnoughShards, result);
    }
}

test "encode" {
    const allocator = std.testing.allocator;
    const count = 16;
    const SHARD_BYTES = 64;

    var input: [SHARD_BYTES * count]u8 = undefined;
    for (0..input.len) |i| input[i] = @intCast(i % 256);

    var original: [count][]const u8 = undefined;
    for (&original, 0..) |*shard, i| {
        const start = i * SHARD_BYTES;
        const end = start + SHARD_BYTES;
        shard.* = input[start..end];
    }

    const recovery = try encode(
        allocator,
        count,
        count,
        &original,
    );
    defer allocator.free(recovery);

    const expected: [16][64]u8 = @import("tests/encode_data.zon");
    for (expected, recovery) |e, r| try testing.expectEqual(e, r);
}
