//! Inkling's fixed one-row projection classes against native cuBLASLt, plus guarded FP32 accumulation fixtures.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const Form = enum { bf16, output, head, control };
const served = [_][1]c.Shape{
    .{.{ .rows = 1, .n = 3072, .k = 4608 }},
    .{.{ .rows = 1, .n = 4608, .k = 3072 }},
    .{.{ .rows = 1, .n = 100512, .k = 4608 }},
    .{.{ .rows = 1, .n = 3072, .k = 4608 }},
};
fn depth(s: c.Shape) usize {
    return s.k / 16 + 4;
}

const Case = struct {
    rig: rig_.Rig,
    form: Form,
    function: cuda.Function,
    lt: *const cuda.cublaslt.Library,
    plan: ?cuda.cublaslt.Linear = null,
    shape: c.Shape = undefined,
    x: std.ArrayList(u16) = .empty,
    w: std.ArrayList(u16) = .empty,
    terms: std.ArrayList(f64) = .empty,
    xd: u64 = 0,
    wd: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    workspace: u64 = 0,

    fn bf16(self: *const Case) bool {
        return self.form == .bf16 or self.form == .control;
    }
    fn output(self: *const Case) c.Elem {
        return if (self.bf16()) .bf16 else .f32;
    }
    fn kernel(self: *Case) g.Kernel {
        const name = switch (self.form) {
            .bf16 => "cuda inkling dense bf16 Q and gate",
            .output => "cuda inkling dense f32 output",
            .head => "cuda inkling dense f32 vocabulary head",
            .control => "control inkling dense missing output tile",
        };
        return .{ .spec = .{ .name = name, .accum = .f32, .depth = depth, .out = self.output(), .served = &served[@backingInt(self.form)], .replaces = "native Linear first cuBLASLt heuristic", .tiles = .{ .{ .size = 1 }, .{ .size = 8 }, .{ .size = 256 } }, .nan = .{ .stated = "IEEE dot classes on finite BF16 inputs and exceptional weight rows" } }, .ctx = self, .vt = &vt, .control = if (self.form == .control) .shapes else null };
    }
    const vt = g.hooks(Case);
    fn deinit(self: *Case) void {
        if (self.plan) |*p| p.deinit();
        for (.{ &self.x, &self.w }) |a| a.deinit(self.rig.gpu.gpa);
        self.terms.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    /// Shape.rows counts independent single-row calls; n and k are the exact served projection dimensions.
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const qualified = served[@backingInt(self.form)][0];
        if (shape.rows < 1 or shape.rows > 8 or shape.n != qualified.n or shape.k != qualified.k) return false;
        if (self.plan) |*p| p.deinit();
        self.plan = null;
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        try self.x.resize(a, shape.rows * shape.k);
        try self.w.resize(a, shape.n * shape.k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.x.items) |*v| v.* = c.toBf16(rng.float(f32) - 0.5);
        for (self.w.items) |*v| v.* = c.toBf16(rng.float(f32) - 0.5);
        if (fill == .stress) {
            var row = try core.contract.rows.Rows.init(a, .dot, .{ .rows = 1, .n = 1, .k = shape.k });
            defer row.deinit();
            row.fill(.stress, seed);
            const half = shape.k / 2;
            for (0..half) |i| row.x[half + i] = c.toBf16(c.fromBf16(row.x[i]) + @as(f32, if (i == 0) 1 else 0));
            row.w[0] = c.toBf16(1);
            row.w[half] = c.toBf16(-1);
            for (0..shape.rows) |r| @memcpy(self.x.items[r * shape.k ..][0..shape.k], row.x);
            for (0..shape.n) |j| @memcpy(self.w.items[j * shape.k ..][0..shape.k], row.w);
            try self.terms.resize(a, shape.k);
            for (self.terms.items, 0..) |*v, i| v.* = c.fromBf16(row.x[i]) * @as(f64, c.fromBf16(row.w[i]));
        } else if (fill == .nonfinite) {
            for (0..shape.rows) |r| {
                self.x.items[r * shape.k] = c.toBf16(1);
                self.x.items[r * shape.k + 16] = c.toBf16(1);
            }
            self.w.items[0] = c.toBf16(std.math.nan(f32));
            self.w.items[shape.k] = c.toBf16(std.math.inf(f32));
            self.w.items[2 * shape.k] = c.toBf16(-std.math.inf(f32));
            self.w.items[3 * shape.k] = c.toBf16(std.math.inf(f32));
            self.w.items[3 * shape.k + 16] = c.toBf16(-std.math.inf(f32));
        }
        self.xd = try self.rig.input(std.mem.sliceAsBytes(self.x.items));
        self.wd = try self.rig.input(std.mem.sliceAsBytes(self.w.items));
        self.out = try self.rig.output(self.count() * c.size(self.output()));
        self.base = try self.rig.scratch(self.count() * c.size(self.output()));
        self.workspace = try self.rig.scratch(64 << 20);
        self.plan = try cuda.cublaslt.Linear.init(self.lt, 1, shape.n, shape.k, if (self.bf16()) .bf16 else .f32, 64 << 20);
        return true;
    }

    pub fn count(self: *Case) usize {
        return self.shape.rows * self.shape.n;
    }
    pub fn run(self: *Case) !void {
        for (0..self.shape.rows) |r| {
            var args: cuda.Args = .{};
            args.add(self.xd + r * self.shape.k * 2);
            args.add(self.wd);
            args.add(self.out + r * self.shape.n * c.size(self.output()));
            args.add(@as(i32, @intCast(self.shape.k)));
            try cuda.launch.launch(self.function, .{
                .grid = .{ .x = @intCast(self.shape.n / 8 - @intFromBool(self.form == .control)) },
                .block = .{ .x = 32 },
            }, self.rig.s, &args);
        }
    }
    pub fn replaced(self: *Case) ![]const u8 {
        for (0..self.shape.rows) |r| try self.plan.?.run(self.xd + r * self.shape.k * 2, self.wd, self.base + r * self.shape.n * c.size(self.output()), self.workspace, 64 << 20, self.rig.s.handle);
        return self.rig.bytes(self.base, self.count() * c.size(self.output()));
    }

    /// Gate 1's finite sum has magnitude below k; four exceptional weight rows determine the nonfinite classes.
    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const bytes = try self.rig.outputBytes();
        for (0..self.count()) |i| {
            const want: f64 = switch (i % self.shape.n) {
                0, 3 => std.math.nan(f64),
                1 => std.math.inf(f64),
                2 => -std.math.inf(f64),
                else => 0,
            };
            if (c.classOf(c.read(self.output(), bytes, i)) != c.classOf(want)) return "dot class differs from FP64";
        }
        return null;
    }
    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        var sum: f64 = 0;
        var magnitude: f64 = 0;
        for (self.terms.items) |t| {
            sum += t;
            magnitude += @abs(t);
        }
        @memset(want, sum);
        @memset(mag, magnitude);
    }
    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        @memset(out, core.contract.rows.sumIn(acc, order, self.terms.items));
    }
};

pub const Set = struct {
    gpu: check_.Gpu,
    bf16: cuda.Module,
    f32: cuda.Module,
    lt: cuda.cublaslt.Library,
    cases: [4]Case,

    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !?*Set {
        if (try gpu.ctx.capability() != 121) return null;
        var lt = try cuda.cublaslt.Library.open();
        errdefer lt.close();
        if (lt.api.cublasLtGetVersion() != 130101) {
            lt.close();
            return null;
        }
        const set = try a.create(Set);
        errdefer a.destroy(set);
        var bf16 = try cuda.Module.load(gpu.d, cuda.kernels.drama3_staged_gemv);
        errdefer bf16.unload();
        var f32_module = try cuda.Module.load(gpu.d, cuda.kernels.inkling2_dense);
        errdefer f32_module.unload();
        set.* = .{ .gpu = gpu, .bf16 = bf16, .f32 = f32_module, .lt = lt, .cases = undefined };
        const b = try bf16.function("drama3_staged_gemv");
        const f = try f32_module.function("tf_ik_dense_gemv_f32");
        for (&set.cases, 0..) |*x, i| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = @fromBackingInt(@intCast(i)), .function = if (i == 0 or i == 3) b else f, .lt = &set.lt };
        errdefer for (&set.cases) |*x| x.deinit();
        for (&set.cases) |*x| try list.append(a, x.kernel());
        return set;
    }
    pub fn deinit(set: *Set) void {
        for (&set.cases) |*x| x.deinit();
        set.f32.unload();
        set.bf16.unload();
        set.lt.close();
        set.gpu.gpa.destroy(set);
    }
};
