//! DSpark's noncausal eight-query blocks and fixed split merge against the shared independent FP64 attention oracle.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const helper = @import("contract_attn.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const served = [_]c.Shape{
    .{ .rows = 1, .n = 24, .k = 8 },      .{ .rows = 1, .n = 24, .k = 65 },
    .{ .rows = 1, .n = 24, .k = 1023 },   .{ .rows = 1, .n = 24, .k = 1024 },
    .{ .rows = 1, .n = 24, .k = 1032 },   .{ .rows = 1, .n = 24, .k = 1280 },
    .{ .rows = 1, .n = 24, .k = 8192 },   .{ .rows = 1, .n = 24, .k = 32768 },
    .{ .rows = 1, .n = 24, .k = 524288 },
};
fn depth(s: c.Shape) usize {
    return s.k + 128;
}

pub const Case = struct {
    rig: @import("contract_rig.zig").Rig,
    attention: cuda.Function,
    merge: cuda.Function,
    local: bool,
    shape: c.Shape = undefined,
    capacity: usize = 0,
    first: usize = 0,
    q: std.ArrayList(u16) = .empty,
    k: std.ArrayList(u16) = .empty,
    v: std.ArrayList(u16) = .empty,
    terms: std.ArrayList(f64) = .empty,
    dev: [3]u64 = @splat(0),
    partials: u64 = 0,
    out: u64 = 0,
    ref: ?helper.Ref = null,
    stress: bool = false,

    pub fn kernel(t: *Case) g.Kernel {
        return .{ .spec = .{ .name = if (t.local) "cuda inkling draft local block attention and merge" else "cuda inkling draft global block attention and merge", .accum = .f32, .depth = depth, .out = .bf16, .ops = 16, .served = &served, .tiles = .{ .{}, .{ .size = 24, .step = 6, .min = 6 }, .{ .size = 64, .min = 8 } }, .nan = .{ .stated = "visible V exceptions propagate; context before the anchor window is excluded" } }, .ctx = t, .vt = &g.hooks(Case) };
    }
    pub fn deinit(t: *Case) void {
        for ([_]*std.ArrayList(u16){ &t.q, &t.k, &t.v }) |p| p.deinit(t.rig.gpu.gpa);
        t.terms.deinit(t.rig.gpu.gpa);
        if (t.ref) |*r| r.deinit();
        t.rig.deinit();
    }
    // Shape.rows counts one checkpoint block; every physical launch has its fixed eight query rows.
    pub fn count(t: *Case) usize {
        return 8 * t.shape.n * 128;
    }
    pub fn prepare(t: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows != 1 or shape.n < 6 or shape.n % 6 != 0 or shape.k < 8) return false;
        t.rig.reset();
        if (t.ref) |*r| r.deinit();
        t.ref = null;
        t.shape = shape;
        t.capacity = if (t.local) @min(shape.k, 1280) else shape.k;
        t.first = if (t.local) shape.k - 8 -| 1024 else 0;
        t.stress = fill == .stress;
        const a = t.rig.gpu.gpa;
        const keys = shape.n / 6;
        try t.q.resize(a, t.count());
        try t.k.resize(a, t.capacity * keys * 128);
        try t.v.resize(a, t.k.items.len);
        var rng = std.Random.DefaultPrng.init(seed);
        for ([_][]u16{ t.q.items, t.k.items, t.v.items }) |values| {
            for (values) |*v| v.* = if (t.stress) 0 else c.toBf16((rng.random().float(f32) - 0.5) * 0.6);
        }
        if (t.stress) {
            const active = @min(shape.k - t.first, 2048);
            try t.terms.resize(a, active);
            const half = active / 2;
            for (0..half) |i| {
                t.terms.items[i] = if (i < 8) 1 else 4096;
                t.terms.items[half + i] = if (i < 8) 0 else -4096;
            }
            if (active % 2 != 0) t.terms.items[active - 1] = 0;
            for (t.terms.items, 0..) |v, i| @memset(t.v.items[((t.first + i) % t.capacity) * keys * 128 ..][0 .. keys * 128], c.toBf16(@floatCast(v)));
        } else if (fill == .nonfinite) {
            const offset = (t.first % t.capacity) * keys * 128;
            t.v.items[offset] = c.toBf16(std.math.nan(f32));
            t.v.items[offset + 1] = c.toBf16(std.math.inf(f32));
            t.v.items[offset + 2] = c.toBf16(-std.math.inf(f32));
        }
        if (t.local and t.first > 0 and !t.stress) t.v.items[((t.first - 1) % t.capacity) * keys * 128 + 7] = c.toBf16(std.math.nan(f32));
        t.dev = .{ try t.rig.input(std.mem.sliceAsBytes(t.q.items)), try t.rig.input(std.mem.sliceAsBytes(t.k.items)), try t.rig.input(std.mem.sliceAsBytes(t.v.items)) };
        t.partials = try t.rig.scratch(8 * shape.n * 32 * 130 * 4);
        t.out = try t.rig.output(t.count() * 2);
        t.ref = try helper.Ref.init(a, 128, shape.k - t.first);
        return true;
    }
    pub fn run(t: *Case) !void {
        var args: cuda.Args = .{};
        inline for (.{ t.dev[0], t.dev[1], t.dev[2], t.partials, @as(i32, @intCast(t.shape.k - 8)), @as(i32, @intCast(t.shape.n)), @as(i32, @intCast(t.shape.n / 6)), @as(i32, @intCast(t.capacity)), @as(i32, @intFromBool(t.local)) }) |v| args.add(v);
        try cuda.launch.launch(t.attention, .{ .grid = .{ .x = @intCast(t.shape.n), .y = 32 }, .block = .{ .x = 256 } }, t.rig.s, &args);
        args = .{};
        args.add(t.partials);
        args.add(t.out);
        args.add(@as(i32, @intCast(t.shape.n)));
        try cuda.launch.launch(t.merge, .{ .grid = .{ .x = @intCast(t.shape.n), .y = 8 }, .block = .{ .x = 128 } }, t.rig.s, &args);
    }
    pub fn load(t: *Case, i: usize, h: *helper.Head) !void {
        const head = i % t.shape.n;
        const keys = t.shape.n / 6;
        for (h.q, 0..) |*v, j| v.* = c.fromBf16(t.q.items[i * 128 + j]);
        h.n = t.shape.k - t.first;
        for (0..h.n) |row| {
            const base = (((t.first + row) % t.capacity) * keys + head / 6) * 128;
            for (0..128) |j| {
                h.k[row * 128 + j] = c.fromBf16(t.k.items[base + j]);
                h.v[row * 128 + j] = c.fromBf16(t.v.items[base + j]);
            }
        }
    }
    pub fn check(t: *Case, at: g.Gate) !?[]const u8 {
        return t.ref.?.check(t, 8 * t.shape.n, 1 / @sqrt(@as(f64, 128)), try t.rig.outputBytes(), at);
    }
    pub fn simulate(t: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        const value = core.contract.rows.sumIn(acc, order, t.terms.items) / @as(f64, @floatFromInt(t.shape.k - t.first));
        @memset(out, value);
    }
    pub fn reference(t: *Case, want: []f64, mag: []f64) !void {
        if (t.stress) {
            try t.simulate(.none, .sequential, want);
            var magnitude: f64 = 0;
            for (t.terms.items) |v| magnitude += @abs(v);
            @memset(mag, magnitude / @as(f64, @floatFromInt(t.shape.k - t.first)));
        } else try t.ref.?.reference(t, 8 * t.shape.n, 1 / @sqrt(@as(f64, 128)), want, mag);
    }
};
