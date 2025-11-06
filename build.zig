const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const tables_exe = b.addExecutable(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tables.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
        .name = "tables",
    });
    const run_tables = b.addRunArtifact(tables_exe);
    const tables_mod = b.createModule(.{ .root_source_file = run_tables.captureStdOut() });
    run_tables.captured_stdout.?.basename = "output.zig"; // work around before zig 0.16

    const exe = b.addExecutable(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tables", .module = tables_mod },
            },
        }),
        .name = "example",
    });
    b.installArtifact(exe);

    const run_step = b.step("run", "");
    run_step.dependOn(&b.addRunArtifact(exe).step);

    const main_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tables", .module = tables_mod },
            },
        }),
    });
    const run_tests = b.addRunArtifact(main_tests);
    b.step("test", "Run main tests").dependOn(&run_tests.step);
}
