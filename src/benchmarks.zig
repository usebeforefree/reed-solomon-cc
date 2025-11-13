/// Benchmarks based off of:
/// https://github.com/AndersTrier/reed-solomon-simd/blob/df1b4fee7f28e2ea9b02e05169b7b93150fdf932/benches/benchmarks.rs
const std = @import("std");
const reedsol = @import("reedsol");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    try roundtrip(allocator);
}

const SHARD_BYTES = 1024;
const ITERATIONS = 10_000;

fn roundtrip(gpa: std.mem.Allocator) !void {
    var stdout_buffer: [0x100]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var progress = std.Progress.start(.{});

    inline for ([_]struct { u32, u32 }{
        .{ 32, 32 },
        .{ 64, 64 },
    }) |entry| {
        const original_count, const recovery_count = entry;

        const bytes = try arena.alloc(u8, original_count * SHARD_BYTES);
        defer arena.free(bytes);
        std.crypto.random.bytes(bytes);
        const original_bytes: *[original_count][SHARD_BYTES]u8 = @ptrCast(bytes);
        var original: [original_count][]const u8 = undefined;
        for (&original, original_bytes) |*slice, *o| slice.* = o;

        {
            var name_buffer: [100]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buffer, "encode:{d}/{d}", .{ original_count, recovery_count });
            const node = progress.start(name, ITERATIONS);
            defer node.end();

            var total_ns: u64 = 0;

            for (0..ITERATIONS) |i| {
                defer node.completeOne();
                std.mem.doNotOptimizeAway(i);

                var encoder = try reedsol.Encoder.init(arena, original_count, recovery_count, SHARD_BYTES);
                defer encoder.deinit(arena);

                var start = try std.time.Timer.start();
                for (original) |o| std.mem.doNotOptimizeAway(try encoder.addOriginalShard(o));
                std.mem.doNotOptimizeAway(try encoder.encode());
                total_ns += start.read();
            }

            const average = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(ITERATIONS));
            try stdout.print("{s} - average of {d}us per encode\n", .{ name, @floor(average) / std.time.ns_per_us });
            try stdout.flush();
        }

        // const recovery = try reedsol.encode(
        //     arena,
        //     original_count,
        //     recovery_count,
        //     &original,
        // );
        // defer arena.free(recovery);
    }
}
