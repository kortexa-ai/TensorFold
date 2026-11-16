//! The shared Inkling contract fixtures and limits without loading unrelated model families.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;

fn image(name: []const u8) error{UnknownContractImage}![]const u8 {
    if (std.mem.eql(u8, name, "inkling2_draft")) return cuda.kernels.inkling2_draft;
    if (std.mem.eql(u8, name, "inkling2_verify")) return cuda.kernels.inkling2_verify;
    return error.UnknownContractImage;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const only: []const []const u8 = if (args.len > 2) args[2..] else &.{ "inkling draft", "inkling verify" };
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    var stream = try cuda.Stream.init(&driver, false);
    defer stream.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(init.gpa);
    const draft = (try @import("contract_inkling_draft_steps.zig").Set.init(gpu, stream, init.gpa, &kernels)) orelse return error.InklingNeedsGb10;
    defer draft.deinit();
    const verify = (try @import("contract_inkling_verify.zig").Set.init(gpu, stream, init.gpa, &kernels)) orelse return error.InklingNeedsGb10;
    defer verify.deinit();
    @import("contract_after.zig").keep(&kernels, only);
    var tally: g.Tally = .{};
    try g.runAll(init.gpa, kernels.items, 0x696b6472, &tally);
    const limits = try @import("contract_limits.zig").liveWith(gpu, only, image);
    std.debug.print("SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed; {d} limit problems\n", .{ tally.kernels, tally.passed, tally.failed, tally.known, tally.controls, tally.caught, tally.missed, limits });
    return if (tally.problems() + limits == 0) 0 else 1;
}
