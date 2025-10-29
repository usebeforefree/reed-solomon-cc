const std = @import("std");
const gf = @import("gf.zig");

pub fn main() !void {
    var stdout_buffer: [0x1000]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch @panic("failed to write to stdout");

    try stdout.writeAll(
        \\const ExpLog = struct {
        \\    exp: [65536]u16,
        \\    log: [65536]u16,
        \\};
        \\
        \\const exp_log: ExpLog = .{
        \\    .exp = .{ 
    );

    var exp: [gf.order]u16 = @splat(0);
    var log: [gf.order]u16 = @splat(0);

    var state: u64 = 1;
    for (0..gf.modulus) |i| {
        exp[state] = @intCast(i);
        state <<= 1;
        if (state >= gf.order) state ^= gf.polynomial;
    }
    exp[0] = gf.modulus;

    // transform into cantor basis

    log[0] = 0;
    for (0..16) |i| {
        const width = @as(u64, 1) << @intCast(i);
        for (0..width) |j| {
            log[j + width] = log[j] ^ gf.cantor_basis[i];
        }
    }

    for (0..gf.order) |i| log[i] = exp[log[i]];
    for (0..gf.order) |i| exp[log[i]] = @intCast(i);
    exp[gf.modulus] = exp[0];

    for (exp) |e| try stdout.print("{d}, ", .{e});
    try stdout.writeAll(
        \\    },
        \\    .log = .{ 
    );
    for (log) |e| try stdout.print("{d}, ", .{e});
    try stdout.writeAll(
        \\    },
        \\};
        \\
        \\pub const skew: [65535]u16 = .{ 
    );

    var skew: [gf.modulus]u16 = @splat(0);
    var temp: [@bitSizeOf(u16) - 1]u16 = @splat(0);

    for (1..16) |i| temp[i - 1] = @as(u16, 1) << @intCast(i);

    for (0..15) |m| {
        const step = @as(u64, 1) << @intCast(m + 1);
        const backwards = (@as(u16, 1) << @intCast(m)) - 1;
        skew[backwards] = 0;

        for (m..15) |i| {
            const s = @as(u16, 1) << @intCast(i + 1);
            var j = backwards;
            while (j < s) {
                skew[j + s] = skew[j] ^ temp[i];
                j += @intCast(step);
            }
        }

        temp[m] = gf.modulus - log[mul(temp[m], log[temp[m] ^ 1], &exp, &log)];

        for (m + 1..15) |i| {
            const sum = addMod(log[temp[i] ^ 1], temp[m]);
            temp[i] = mul(temp[i], sum, &exp, &log);
        }
    }

    for (0..gf.modulus) |i| skew[i] = log[skew[i]];

    for (skew) |e| try stdout.print("{d},\n", .{e});
    try stdout.writeAll(
        \\};
    );
}

fn mul(x: u16, log_m: u16, exp: *const [65536]u16, log: *const [65536]u16) u16 {
    if (x == 0) return 0;
    return exp[addMod(log[x], log_m)];
}

fn addMod(x: u32, y: u32) u16 {
    const sum = x + y;
    return @truncate(sum + (sum >> 16));
}
