//! Target-feature mixtures and draft QK RoPE with FP64 ranges through their BF16 rounding nodes.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const math = @import("contract_moss_image_ref.zig");
const c = math.c;
const g = math.g;
const R = math.Range;
const Form = enum { mix, query, key, layer };
const served = [_][2]c.Shape{
    .{ .{ .rows = 8, .n = 4608, .k = 4608 }, .{ .rows = 256, .n = 4608, .k = 4608 } },
    .{ .{ .rows = 8, .n = 24, .k = 128 }, .{ .rows = 1, .n = 24, .k = 128 } },
    .{ .{ .rows = 8, .n = 4, .k = 128 }, .{ .rows = 256, .n = 4, .k = 128 } },
    .{ .{ .rows = 8, .n = 4608, .k = 4608 }, .{ .rows = 1, .n = 4608, .k = 4608 } },
};
fn depth(s: c.Shape) usize {
    return s.k + 16;
}
fn sqrt(x: f64) f64 {
    return @sqrt(x);
}

pub const Case = struct {
    rig: @import("contract_rig.zig").Rig,
    form: Form,
    function: cuda.Function,
    ends: math.Ends,
    shape: c.Shape = undefined,
    x: std.ArrayList(u16) = .empty,
    norm: std.ArrayList(u16) = .empty,
    alpha: [8]u16 = @splat(c.toBf16(0.125)),
    dev: [3]u64 = @splat(0),
    out: u64 = 0,
    cache: u64 = 0,

    const vt = g.hooks(Case);
    pub fn kernel(t: *Case) g.Kernel {
        return .{ .spec = .{ .name = switch (t.form) {
            .mix => "cuda inkling draft feature mixture norm",
            .query => "cuda inkling draft query RMS RoPE",
            .key => "cuda inkling draft key RMS RoPE cache",
            .layer => "cuda inkling draft layer RMS norm",
        }, .accum = .f32, .depth = depth, .out = .bf16, .served = &served[@backingInt(t.form)], .tiles = .{ .{ .size = 8 }, .{}, .{} }, .sums = .{ .f16 = 100, .bf16 = 0, .exact = false, .why = "normalized BF16 outputs conceal BF16 square-sum error at some orders" } }, .ctx = t, .vt = &vt };
    }
    pub fn deinit(t: *Case) void {
        t.x.deinit(t.rig.gpu.gpa);
        t.norm.deinit(t.rig.gpu.gpa);
        t.ends.deinit();
        t.rig.deinit();
    }
    fn width(t: *const Case) usize {
        return if (t.form == .mix or t.form == .layer) 4608 else 128;
    }
    fn groups(t: *const Case) usize {
        return if (t.form == .mix or t.form == .layer) t.shape.rows else t.shape.rows * t.shape.n;
    }
    pub fn count(t: *Case) usize {
        return t.groups() * t.width() * (if (t.form == .key) @as(usize, 2) else 1);
    }
    pub fn prepare(t: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 256 or shape.n != served[@backingInt(t.form)][0].n or shape.k != t.width()) return false;
        t.rig.reset();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const size = t.groups() * t.width();
        try t.x.resize(a, size * (if (t.form == .mix) @as(usize, 8) else 1));
        try t.norm.resize(a, t.width());
        var rng = std.Random.DefaultPrng.init(seed);
        for (t.x.items) |*v| v.* = c.toBf16(if (fill == .stress) 32 else rng.random().float(f32) - 0.5);
        for (t.norm.items) |*v| v.* = c.toBf16(if (fill == .stress) 1 else 0.75 + rng.random().float(f32) * 0.5);
        if (fill == .nonfinite) {
            t.x.items[0] = c.toBf16(std.math.nan(f32));
            t.x.items[t.width()] = c.toBf16(std.math.inf(f32));
            t.x.items[2 * t.width()] = c.toBf16(-std.math.inf(f32));
        }
        t.dev = .{ try t.rig.input(std.mem.sliceAsBytes(t.x.items)), try t.rig.input(std.mem.sliceAsBytes(t.norm.items)), try t.rig.input(std.mem.sliceAsBytes(&t.alpha)) };
        t.out = try t.rig.output(size * 2);
        if (t.form == .key) t.cache = try t.rig.output(size * 2);
        try t.ends.size(t.count());
        return true;
    }
    pub fn run(t: *Case) !void {
        var args: cuda.Args = .{};
        if (t.form == .mix) {
            inline for (.{ t.dev[0], t.dev[2], t.dev[1], t.out, @as(i32, @intCast(t.shape.rows * 4608)) }) |v| args.add(v);
        } else if (t.form == .layer) {
            inline for (.{ t.dev[0], t.dev[1], t.out, @as(i32, 4608) }) |v| args.add(v);
        } else {
            inline for (.{ t.dev[0], t.dev[1], t.out, t.cache, @as(i32, @intCast(t.shape.n)), @as(i32, 524288), @as(i32, @intCast(t.shape.rows)) }) |v| args.add(v);
        }
        try cuda.launch.launch(t.function, .{ .grid = .{ .x = if (t.form == .mix or t.form == .layer) @intCast(t.shape.rows) else @intCast(t.shape.n), .y = if (t.form == .mix or t.form == .layer) 1 else @intCast(t.shape.rows) }, .block = .{ .x = if (t.form == .mix or t.form == .layer) 256 else 128 } }, t.rig.s, &args);
    }
    fn value(t: *const Case, group: usize, col: usize) f64 {
        if (t.form != .mix) return c.fromBf16(t.x.items[group * t.width() + col]);
        var v: f64 = 0;
        for (0..8) |tap| v += @as(f64, c.fromBf16(t.x.items[(tap * t.shape.rows + group) * 4608 + col])) * c.fromBf16(t.alpha[tap]);
        return v;
    }
    fn rotated(t: *const Case, group: usize, col: usize, normalized: []const f64) f64 {
        const frequency: f32 = @floatCast(std.math.pow(f64, 1000000, -2 * @as(f64, @floatFromInt(col % 64)) / 128));
        const angle: f32 = @as(f32, @floatFromInt(524288 + group / t.shape.n)) * frequency;
        return normalized[col] * @cos(@as(f64, angle)) + (if (col < 64) -normalized[(col + 64) % 128] else normalized[(col + 64) % 128]) * @sin(@as(f64, angle));
    }
    pub fn reference(t: *Case, want: []f64, mag: []f64) !void {
        const a = t.rig.gpu.gpa;
        const w = t.width();
        const normalized = try a.alloc(R, w);
        defer a.free(normalized);
        const center_norm = try a.alloc(f64, w);
        defer a.free(center_norm);
        const values = try a.alloc(R, w);
        defer a.free(values);
        for (0..t.groups()) |group| {
            var center_square: f64 = 0;
            for (0..w) |col| {
                const x = t.value(group, col);
                center_square += x * x;
            }
            const center_inv = 1 / @sqrt(center_square / @as(f64, @floatFromInt(w)) + 1e-6);
            for (center_norm, 0..) |*v, col| {
                const x = t.value(group, col);
                const norm = c.fromBf16(t.norm.items[col]);
                v.* = if (t.form == .mix) x * center_inv * norm else math.bf(math.bf(x * center_inv) * norm);
            }
            var squares = R.of(0);
            for (values, 0..) |*value_, col| {
                const x = t.value(group, col);
                var magnitude: f64 = @abs(x);
                if (t.form == .mix) {
                    magnitude = 0;
                    for (0..8) |tap| magnitude += @abs(@as(f64, c.fromBf16(t.x.items[(tap * t.shape.rows + group) * w + col])) * c.fromBf16(t.alpha[tap]));
                }
                const error_ = if (t.form == .mix) c.gamma(8, 0x1p-24) * magnitude else 0;
                value_.* = .{ .lo = x - error_, .hi = x + error_ };
                const low = if (value_.lo <= 0 and value_.hi >= 0) 0 else @min(value_.lo * value_.lo, value_.hi * value_.hi);
                squares = squares.add(.{ .lo = low, .hi = @max(value_.lo * value_.lo, value_.hi * value_.hi) });
            }
            const sum_error = c.gamma(@floatFromInt(w), 0x1p-24) * @max(@abs(squares.lo), @abs(squares.hi));
            squares = .{ .lo = @max(0, squares.lo - sum_error), .hi = squares.hi + sum_error };
            const inv = squares.mul(R.of(1 / @as(f64, @floatFromInt(w)))).fp32(0.5).add(R.of(1e-6)).fp32(0.5).up(sqrt).recip().fp32(2);
            for (normalized, values, 0..) |*v, x, col| {
                const n = R.of(c.fromBf16(t.norm.items[col]));
                const scaled = x.mul(inv).fp32(0.5);
                v.* = if (t.form == .mix) scaled.mul(n).fp32(0.5) else scaled.bf16().mul(n).fp32(0.5).bf16();
            }
            for (normalized, 0..) |value_, col| {
                const at = group * w + col;
                var range = value_;
                if (t.form == .query or t.form == .key) {
                    const frequency: f32 = @floatCast(std.math.pow(f64, 1000000, -2 * @as(f64, @floatFromInt(col % 64)) / 128));
                    const angle: f32 = @as(f32, @floatFromInt(524288 + group / t.shape.n)) * frequency;
                    const cosine = R.of(@cos(@as(f64, angle))).fp32(2);
                    const sine = R.of(@sin(@as(f64, angle))).fp32(2);
                    const other = normalized[(col + 64) % 128];
                    const rotate = if (col < 64) R{ .lo = -other.hi, .hi = -other.lo } else other;
                    range = value_.mul(cosine).fp32(0.5).add(rotate.mul(sine).fp32(0.5)).fp32(0.5);
                }
                range = range.bf16();
                const center = math.bf(if (t.form == .query or t.form == .key) t.rotated(group, col, center_norm) else center_norm[col]);
                t.ends.put(want, mag, at, center, range);
                if (t.form == .key) {
                    const row = group / t.shape.n;
                    const cache_at = t.groups() * w + (((524288 + row) % t.shape.rows) * t.shape.n + group % t.shape.n) * w + col;
                    t.ends.put(want, mag, cache_at, center, range);
                }
            }
        }
    }
    pub fn bounds(t: *Case, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Case, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
    pub fn simulate(t: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const w = t.width();
        const terms = try a.alloc(f64, w);
        defer a.free(terms);
        const normalized = try a.alloc(f64, w);
        defer a.free(normalized);
        for (0..t.groups()) |group| {
            for (terms, 0..) |*v, col| {
                const x = t.value(group, col);
                v.* = x * x;
            }
            const inv = 1 / @sqrt(core.contract.rows.sumIn(acc, order, terms) / @as(f64, @floatFromInt(w)) + 1e-6);
            for (normalized, 0..) |*v, col| {
                const x = t.value(group, col);
                const n = c.fromBf16(t.norm.items[col]);
                v.* = if (t.form == .mix) x * inv * n else math.bf(math.bf(x * inv) * n);
            }
            for (0..w) |col| {
                const v = math.bf(if (t.form == .mix or t.form == .layer) normalized[col] else t.rotated(group, col, normalized));
                out[group * w + col] = v;
                if (t.form == .key) out[t.groups() * w + (((524288 + group / t.shape.n) % t.shape.rows) * t.shape.n + group % t.shape.n) * w + col] = v;
            }
        }
    }
};

test "contract image: draft RMS class oracle propagates NaN and distinguishes infinite norms" {
    const a = std.testing.allocator;
    var t: Case = .{ .rig = .{ .gpu = .{ .gpa = a, .ctx = undefined, .d = undefined, .io = undefined }, .s = undefined }, .form = .layer, .function = undefined, .ends = .{ .a = a }, .shape = .{ .rows = 5, .n = 4608, .k = 4608 } };
    defer t.x.deinit(a);
    defer t.norm.deinit(a);
    defer t.ends.deinit();
    try t.x.resize(a, 5 * 4608);
    try t.norm.resize(a, 4608);
    @memset(t.x.items, c.toBf16(1));
    @memset(t.norm.items, c.toBf16(1));
    t.x.items[0] = c.toBf16(std.math.nan(f32));
    t.x.items[4608] = c.toBf16(std.math.inf(f32));
    t.x.items[2 * 4608] = c.toBf16(-std.math.inf(f32));
    try t.ends.size(t.count());
    const want = try a.alloc(f64, t.count());
    defer a.free(want);
    const magnitude = try a.alloc(f64, t.count());
    defer a.free(magnitude);
    try t.reference(want, magnitude);
    for (want[0..4608]) |v| try std.testing.expect(std.math.isNan(v));
    try std.testing.expect(std.math.isNan(want[4608]));
    try std.testing.expect(std.math.isNan(want[2 * 4608]));
    try std.testing.expectEqual(@as(f64, 0), want[4609]);
    try std.testing.expectEqual(@as(f64, 0), want[2 * 4608 + 1]);
    try std.testing.expectEqual(@as(f64, 1), want[3 * 4608]);
}
