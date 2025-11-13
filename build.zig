const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const use_llvm = b.option(bool, "use-llvm", "Compile library and tests with LLVM backend");

    const tables_exe = b.addExecutable(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tables.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
        .use_llvm = use_llvm,
        .name = "tables",
    });
    const run_tables = b.addRunArtifact(tables_exe);
    const tables_mod = b.createModule(.{ .root_source_file = run_tables.captureStdOut() });
    run_tables.captured_stdout.?.basename = "table.zig"; // work around before zig 0.16

    const reedsol = b.addModule("reedsol", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "tables", .module = tables_mod }},
    });

    const test_step = b.step("test", "Run tests");
    inline for (.{
        .{ "encode", "src/tests.zig" },
        .{ "engine", "src/root.zig" },
    }) |entry| {
        const name, const path = entry;

        const tests = b.addTest(.{
            .name = name,
            .use_llvm = use_llvm,
            .root_module = b.createModule(.{
                .root_source_file = b.path(path),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "reedsol", .module = reedsol },
                    .{ .name = "tables", .module = tables_mod },
                },
            }),
        });

        const run_tests = b.addRunArtifact(tests);
        test_step.dependOn(&run_tests.step);
    }
}
