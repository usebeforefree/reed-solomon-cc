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

test "encode leaks" {
    const S = struct {
        const input = [_]u8{
            0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
            22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
            44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
            66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,
            88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
            132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
            154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
            176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
            198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
            220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
            242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 0,   1,   2,   3,   4,   5,   6,   7,
            8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
            30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
            52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
        };

        fn doTest(allocator: std.mem.Allocator) !void {
            try encodeDecodeCycle(allocator, 5, 64, input, @splat(true), @splat(false));
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, S.doTest, .{});
}
