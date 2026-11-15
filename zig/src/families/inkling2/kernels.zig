//! Inkling's own operators and the shared EXL3 routed dot, loaded by the native CUDA driver.
const std = @import("std");
const cuda = @import("cuda");

pub const Op = enum { mix, finish, rank_sum, logits, quant, swiglu, route, moe_sum, conv, conv_commit, prep, relative, attention, attention_decode, rotate, unrotate, embed, norm, top1 };

pub const Mix = extern struct {
    partial: u64,
    blocks: u64,
    attention: u64,
    query: u64,
    norm: u64,
    out: u64,
    rows: i32,
    width: i32,
    block_stride: i32,
    blocks_count: i32,
    queries: i32,
    mode: i32,
};
pub const Attention = extern struct {
    q: u64,
    k: u64,
    v: u64,
    relative: u64,
    gate: u64,
    sink: u64,
    out: u64,
    rows: i32,
    pos: i32,
    hq: i32,
    hk: i32,
    capacity: i32,
    window: i32,
};
pub const Dot = extern struct {
    x0: u64,
    x1: u64,
    tp0: u64,
    tp1: u64,
    k2_0: u64,
    k2_1: u64,
    uids: u64,
    ucount: u64,
    members: u64,
    z: u64,
    k: i32,
    n: i32,
    p: i32,
    sk: i32,
    maxm: i32,
    slots: i32,
    nu: i32,
    mats: i32,
    ns: i32,
    standard: i32,
    skip: i32,
};
pub const Pick = extern struct { value: f32, token: u32, nonfinite: u32, pad: u32 };

pub const Kernels = struct {
    mod: cuda.Module,
    experts_mod: cuda.Module,
    plan: cuda.grouped.Router,
    x3: cuda.exl3_experts.Experts,
    functions: std.EnumArray(Op, cuda.Function),
    split_decode: bool,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2);
        errdefer mod.unload();
        var experts = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer experts.unload();
        var x3 = try cuda.exl3_experts.Experts.load(d);
        errdefer x3.unload();
        var functions: std.EnumArray(Op, cuda.Function) = undefined;
        inline for (comptime std.meta.tags(Op)) |op| {
            const symbol = comptime "tf_ik_" ++ @tagName(op);
            functions.set(op, try mod.function(symbol));
        }
        return .{ .mod = mod, .experts_mod = experts, .plan = try cuda.grouped.Router.resolve(experts), .x3 = x3, .functions = functions, .split_decode = (try d.currentCapability()) == 121 };
    }

    pub fn deinit(k: *Kernels) void {
        k.x3.unload();
        k.experts_mod.unload();
        k.mod.unload();
    }

    pub fn go(k: *const Kernels, comptime op: Op, s: cuda.Stream, grid: cuda.Dim3, threads: u32, values: anytype) !void {
        var a: cuda.Args = .{};
        inline for (values) |v| a.add(v);
        try cuda.launch.launch(k.functions.get(op), .{ .grid = grid, .block = .{ .x = threads } }, s, &a);
    }

    pub fn dot(k: *const Kernels, s: cuda.Stream, a: Dot) !void {
        var args: cuda.Args = .{};
        args.add(a);
        try cuda.launch.launch(k.x3.grouped[2], .{ .grid = .{ .x = @intCast(a.nu), .y = @intCast(@divExact(a.n, 128)), .z = @intCast(a.sk * @divTrunc(a.maxm + 15, 16)) }, .block = .{ .x = 128 } }, s, &args);
    }
};

pub fn int(v: usize) i32 {
    return @intCast(v);
}
pub fn ceil(n: usize, d: usize) u32 {
    return @intCast((n + d - 1) / d);
}

test "Inkling CUDA by-value argument layouts match the kernel's C structs" {
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(Mix));
    try std.testing.expectEqual(@as(usize, 80), @sizeOf(Attention));
    try std.testing.expectEqual(@as(usize, 128), @sizeOf(Dot));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Pick));
}
