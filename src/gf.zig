//! Describes the galois field over which reed-solomon operates.

pub const order = 65536;
pub const modulus = order - 1;
pub const polynomial = 65581;

pub const cantor_basis: [16]u16 = .{
    0x0001, 0xACCA, 0x3C0E, 0x163E,
    0xC582, 0xED2E, 0x914C, 0x4012,
    0x6C98, 0x10D8, 0x6A72, 0xB900,
    0xFDB8, 0xFB34, 0xFF38, 0x991E,
};
