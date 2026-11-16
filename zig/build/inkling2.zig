//! Inkling 2's CUDA family, focused host checks and native layer-oracle tool.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/inkling2.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

pub fn host(b: *std.Build, m: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-inkling2-host", "Inkling geometry, exact cache accounting and native argument layouts").dependOn(&run.step);
    all.dependOn(&run.step);
}

pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) void {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/dump_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("inkling2", module(b, target, optimize, cuda, core, lanes));
    const exe = b.addExecutable(.{ .name = "tf-inkling-cuda", .root_module = m });
    b.step("tf-inkling-cuda", "Native Inkling layer/checkpoint oracle dump").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const checks = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/inkling_contract_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    checks.addImport("cuda", cuda);
    checks.addImport("core", core);
    const contract = b.addExecutable(.{ .name = "tf-inkling-contract", .root_module = checks });
    b.step("tf-inkling-contract", "Inkling shared CUDA contract gates and live limits").dependOn(&b.addInstallArtifact(contract, .{}).step);
}
