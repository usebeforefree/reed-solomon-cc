const gf = @import("gf.zig");
const utils = @import("utils.zig");

pub fn fwht(data: [gf.order]u16, m_truncated: u64) void {
    var dist: u64 = 1;
    var dist4: u64 = 4;

    while (dist4 <= gf.order) {
        var r: u64 = 0;
        while (r < m_truncated) : (r += 4) {
            for (r..r + dist4) |offset| {
                fwht4(data, offset, dist);
            }
        }
        dist = dist4;
        dist4 <<= 2;
    }
}

fn fwht4(data: [gf.order]u16, offset: u16, dist: u16) void {
    const x0: u64 = @intCast(offset + dist * 0);
    const x1: u64 = @intCast(offset + dist * 1);
    const x2: u64 = @intCast(offset + dist * 2);
    const x3: u64 = @intCast(offset + dist * 3);

    const s0, const d0 = fwht2(data[x0], data[x1]);
    const s1, const d1 = fwht2(data[x2], data[x3]);
    const s2, const d2 = fwht2(s0, s1);
    const s3, const d3 = fwht2(d0, d1);

    data[x0] = s2;
    data[x1] = s3;
    data[x2] = d2;
    data[x3] = d3;
}

fn fwht2(a: u16, b: u16) struct { u16, u16 } {
    const sum = utils.addMod(a, b);
    const dif = utils.subMod(a, b);

    return .{ sum, dif };
}
