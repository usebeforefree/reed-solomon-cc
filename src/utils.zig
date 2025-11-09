const gf = @import("gf.zig");

pub fn mul(x: u16, log_m: u16, exp: *const [gf.order]u16, log: *const [gf.order]u16) u16 {
    if (x == 0) return 0;
    return exp[addMod(log[x], log_m)];
}

pub fn addMod(x: u32, y: u32) u16 {
    const sum = x + y;
    return @truncate(sum + (sum >> 16));
}

pub fn subMod(x: u32, y: u32) u16 {
    const dif = x + gf.modulus - y;
    return @truncate(dif + (dif >> 16));
}
