//! Exact BF16 ring writes and every convolution-prefix snapshot, including exceptional payload bits.
const std = @import("std");
const cuda = @import("cuda");
const math = @import("contract_moss_image_ref.zig");
const c = math.c;
const g = math.g;
const Form = enum { store, snapshot };
fn depth(_: c.Shape) usize {
    return 0;
}
const served = [_][2]c.Shape{
    .{ .{ .rows = 8, .n = 512, .k = 1 }, .{ .rows = 256, .n = 512, .k = 1 } },
    .{ .{ .rows = 8, .n = 512, .k = 1 }, .{ .rows = 8, .n = 4608, .k = 1 } },
};

pub const Case = struct {
    rig: @import("contract_rig.zig").Rig,
    form: Form,
    function: cuda.Function,
    expected: std.ArrayList(u16) = .empty,
    shape: c.Shape = undefined,
    input: u64 = 0,
    history: u64 = 0,
    out: u64 = 0,

    const vt = g.hooks(Case);
    pub fn kernel(t: *Case) g.Kernel {
        return .{ .spec = .{ .name = if (t.form == .store) "cuda inkling draft BF16 context ring" else "cuda inkling verify convolution prefixes", .accum = .none, .depth = depth, .out = .raw, .served = &served[@backingInt(t.form)], .tiles = .{ .{ .size = 8 }, .{ .size = 256 }, .{} }, .nan = .{ .stated = "copies every BF16 payload bit, including nonfinite values" } }, .ctx = t, .vt = &vt };
    }
    pub fn deinit(t: *Case) void {
        t.expected.deinit(t.rig.gpu.gpa);
        t.rig.deinit();
    }
    pub fn count(t: *Case) usize {
        return t.expected.items.len * 2;
    }
    pub fn prepare(t: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > (if (t.form == .store) @as(usize, 256) else 8) or shape.n == 0) return false;
        t.rig.reset();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const x = try a.alloc(u16, shape.rows * shape.n);
        defer a.free(x);
        const old = try a.alloc(u16, 3 * shape.n);
        defer a.free(old);
        var rng = std.Random.DefaultPrng.init(seed);
        for (x) |*v| v.* = rng.random().int(u16);
        for (old) |*v| v.* = rng.random().int(u16);
        if (fill == .nonfinite) for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }, 0..) |v, i| {
            x[i % x.len] = v;
        };
        const size = if (t.form == .store) x.len else (shape.rows + 1) * 3 * shape.n;
        try t.expected.resize(a, size);
        if (t.form == .store) {
            for (0..shape.rows) |r| @memcpy(t.expected.items[((1234 + r) % shape.rows) * shape.n ..][0..shape.n], x[r * shape.n ..][0..shape.n]);
        } else for (0..shape.rows + 1) |prefix| for (0..3) |tap| {
            const src: isize = @as(isize, @intCast(prefix + tap)) - 3;
            const row = if (src < 0) old[@as(usize, @intCast(src + 3)) * shape.n ..][0..shape.n] else x[@as(usize, @intCast(src)) * shape.n ..][0..shape.n];
            @memcpy(t.expected.items[(prefix * 3 + tap) * shape.n ..][0..shape.n], row);
        };
        t.input = try t.rig.input(std.mem.sliceAsBytes(x));
        t.history = try t.rig.input(std.mem.sliceAsBytes(old));
        t.out = try t.rig.output(size * 2);
        return true;
    }
    pub fn run(t: *Case) !void {
        var args: cuda.Args = .{};
        if (t.form == .store) {
            inline for (.{ t.input, t.out, @as(i32, @intCast(t.shape.n)), @as(i32, 1234), @as(i32, @intCast(t.shape.rows)), @as(i32, @intCast(t.shape.rows)) }) |v| args.add(v);
        } else {
            inline for (.{ t.input, t.history, t.out, @as(i32, @intCast(t.shape.rows)), @as(i32, @intCast(t.shape.n)), @as(i32, @intCast(3 * t.shape.n)) }) |v| args.add(v);
        }
        const items = if (t.form == .store) t.shape.rows * t.shape.n else t.shape.n;
        try cuda.launch.launch(t.function, .{ .grid = .{ .x = @intCast((items + 255) / 256), .y = if (t.form == .store) 1 else @intCast(t.shape.rows + 1) }, .block = .{ .x = 256 } }, t.rig.s, &args);
    }
    pub fn check(t: *Case, _: g.Gate) !?[]const u8 {
        if (!std.mem.eql(u8, try t.rig.outputBytes(), std.mem.sliceAsBytes(t.expected.items))) return "BF16 payload or prefix index differs from independent copy reference";
        return null;
    }
    pub fn reference(_: *Case, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
};
