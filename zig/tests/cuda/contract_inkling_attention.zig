//! One-token Inkling output partitions: original bytes, guarded calls, exceptional values and FP32 P.V stress.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;

const Args = extern struct {
    q: u64,
    k: u64,
    v: u64,
    relative: u64,
    gate: u64,
    sink: u64,
    out: u64,
    rows: i32 = 1,
    pos: i32,
    hq: i32,
    hk: i32,
    capacity: i32,
    window: i32,
};
const Form = struct { local: bool = false, hidden: bool = false, control: bool = false };
const forms = [_]Form{ .{}, .{ .local = true }, .{ .local = true, .hidden = true }, .{ .control = true } };
const served = [_]c.Shape{
    .{ .rows = 1, .n = 24, .k = 65 },     .{ .rows = 1, .n = 24, .k = 1023 },
    .{ .rows = 1, .n = 24, .k = 1024 },   .{ .rows = 1, .n = 24, .k = 1025 },
    .{ .rows = 1, .n = 24, .k = 2050 },   .{ .rows = 1, .n = 24, .k = 8192 },
    .{ .rows = 1, .n = 24, .k = 32768 },  .{ .rows = 1, .n = 48, .k = 65536 },
    .{ .rows = 1, .n = 24, .k = 524288 },
};
fn depth(s: c.Shape) usize {
    return s.k + 128;
}

const Case = struct {
    rig: rig_.Rig,
    form: Form,
    candidate: cuda.Function,
    baseline: cuda.Function,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    host: [6]std.ArrayList(u16) = @splat(.empty),
    terms: std.ArrayList(f64) = .empty,
    params: std.ArrayList(Args) = .empty,
    base: u64 = 0,
    stress: bool = false,
    msg: [200]u8 = undefined,

    fn kernel(self: *Case) g.Kernel {
        const name = if (self.form.control) "control inkling attention missing one output partition" else if (self.form.hidden) "cuda inkling decode attention hidden masked V" else if (self.form.local) "cuda inkling decode attention local ring" else "cuda inkling decode attention global";
        return .{ .spec = .{ .name = name, .accum = .f32, .depth = depth, .out = .bf16, .ops = 16, .served = &served, .replaces = "tf_ik_attention on identical one-query inputs", .tiles = .{ .{ .size = 1 }, .{ .size = 24, .step = 6, .min = 6 }, .{ .size = 64 } }, .nan = .{ .stated = "visible V exceptions follow Inkling's original rule; masked values are excluded" } }, .ctx = self, .vt = &vt, .control = if (self.form.control) .shapes else null };
    }
    const vt = g.hooks(Case);

    fn deinit(self: *Case) void {
        for (&self.host) |*x| x.deinit(self.rig.gpu.gpa);
        self.params.deinit(self.rig.gpu.gpa);
        self.terms.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    /// Shape.rows counts independent one-query fixtures, not a multi-row attention launch.
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.rows == 0 or shape.rows > 8 or shape.n < 6 or shape.n % 6 != 0 or shape.k == 0) return false;
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        self.stress = fill == .stress;
        const cap = if (self.form.local) @min(shape.k, 1280) else shape.k;
        const kh = shape.n / 6;
        const sizes = [_]usize{ shape.n * 128, cap * kh * 128, cap * kh * 128, shape.n * 1024, shape.n * 128, shape.n };
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (&self.host, sizes) |*h, size| {
            try h.resize(a, shape.rows * size);
            for (h.items) |*x| x.* = c.toBf16((rng.float(f32) - 0.5) * 0.6);
        }
        if (self.stress) {
            for (&self.host) |*h| @memset(h.items, 0);
            @memset(self.host[5].items, c.toBf16(-std.math.inf(f32)));
            const visible = if (self.form.local) @min(shape.k, 1024) else shape.k;
            const active = @min(visible, 2048);
            try self.terms.resize(a, active);
            const half = active / 2;
            for (0..half) |i| {
                self.terms.items[i] = if (i < 8) 1 else 4096;
                self.terms.items[half + i] = if (i < 8) 0 else -4096;
            }
            if (active % 2 != 0) self.terms.items[active - 1] = 0;
            const first = shape.k - visible;
            for (0..shape.rows) |r| for (self.terms.items, 0..) |v, j| {
                const offset = r * sizes[2] + ((first + j) % cap) * kh * 128;
                @memset(self.host[2].items[offset..][0 .. kh * 128], c.toBf16(@floatCast(v)));
            };
        } else if (fill == .nonfinite) self.plant(sizes, cap);
        if (self.form.hidden and !self.stress) {
            const first = shape.k - @min(shape.k, 1024);
            if (first > 0 and first % 64 != 0) for (0..shape.rows) |r| {
                self.host[2].items[r * sizes[2] + ((first - 1) % cap) * kh * 128] = c.toBf16(std.math.nan(f32));
            };
        }
        var dev: [6]u64 = undefined;
        for (&dev, &self.host) |*d, h| d.* = try self.rig.input(std.mem.sliceAsBytes(h.items));
        const out = try self.rig.output(self.count() * 2);
        self.base = try self.rig.scratch(self.count() * 2);
        try self.params.resize(a, shape.rows);
        for (self.params.items, 0..) |*p, r| p.* = .{
            .q = dev[0] + r * sizes[0] * 2,
            .k = dev[1] + r * sizes[1] * 2,
            .v = dev[2] + r * sizes[2] * 2,
            .relative = dev[3] + r * sizes[3] * 2,
            .gate = dev[4] + r * sizes[4] * 2,
            .sink = dev[5] + r * sizes[5] * 2,
            .out = out + r * sizes[0] * 2,
            .pos = @intCast(shape.k - 1),
            .hq = @intCast(shape.n),
            .hk = @intCast(kh),
            .capacity = @intCast(cap),
            .window = if (self.form.local) 1023 else 0,
        };
        return true;
    }

    fn plant(self: *Case, sizes: [6]usize, cap: usize) void {
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const first = if (self.form.local) self.shape.k - @min(self.shape.k, 1024) else 0;
        const kv = (first % cap) * (self.shape.n / 6) * 128;
        self.host[0].items[0] = nan;
        if (self.shape.rows > 1) self.host[2].items[sizes[2] + kv] = inf;
        if (self.shape.rows > 2) self.host[2].items[2 * sizes[2] + kv + 1] = nan;
        if (self.shape.rows > 3) {
            self.host[2].items[3 * sizes[2] + kv + 2] = inf;
            self.host[2].items[3 * sizes[2] + ((first + 1) % cap) * (self.shape.n / 6) * 128 + 2] = c.toBf16(-std.math.inf(f32));
        }
        if (self.shape.rows > 4) self.host[4].items[4 * sizes[4] + 3] = nan;
    }

    pub fn run(self: *Case) !void {
        for (self.params.items) |p| {
            var a: cuda.Args = .{};
            a.add(p);
            try cuda.launch.launch(self.candidate, .{ .grid = .{ .x = 1, .y = @intCast(p.hq), .z = if (self.form.control) 3 else 4 }, .block = .{ .x = 128 } }, self.rig.s, &a);
        }
    }

    pub fn replaced(self: *Case) ![]const u8 {
        for (self.params.items, 0..) |p, r| {
            var b = p;
            b.out = self.base + r * self.shape.n * 128 * 2;
            var a: cuda.Args = .{};
            a.add(b);
            try cuda.launch.launch(self.baseline, .{ .grid = .{ .x = 1, .y = @intCast(p.hq) }, .block = .{ .x = 128 } }, self.rig.s, &a);
        }
        return self.rig.bytes(self.base, self.count() * 2);
    }

    pub fn count(self: *Case) usize {
        return self.shape.rows * self.shape.n * 128;
    }

    /// The exact cancellation stress is uniform attention; the active integer terms are below 2^24 in FP32.
    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        const n = if (self.form.local) @min(self.shape.k, 1024) else self.shape.k;
        @memset(out, core.contract.rows.sumIn(acc, order, self.terms.items) / @as(f64, @floatFromInt(n)) * 0.5);
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        if (self.stress) {
            try self.simulate(.none, .sequential, want);
            var total: f64 = 0;
            for (self.terms.items) |v| total += @abs(v);
            const n = if (self.form.local) @min(self.shape.k, 1024) else self.shape.k;
            @memset(mag, total / @as(f64, @floatFromInt(n)) * 0.5);
            return;
        }
        const a = self.rig.gpu.gpa;
        const first = if (self.form.local) self.shape.k - @min(self.shape.k, 1024) else 0;
        const cap = if (self.form.local) @min(self.shape.k, 1280) else self.shape.k;
        const kh = self.shape.n / 6;
        const probs = try a.alloc(f64, self.shape.k - first);
        defer a.free(probs);
        for (0..self.shape.rows) |r| for (0..self.shape.n) |h| {
            var m: f64 = -std.math.inf(f64);
            for (probs, 0..) |*p, j| {
                var dot: f64 = 0;
                const at = (r * cap + (first + j) % cap) * kh * 128 + h / 6 * 128;
                for (0..128) |e| dot += c.fromBf16(self.host[0].items[(r * self.shape.n + h) * 128 + e]) * @as(f64, c.fromBf16(self.host[1].items[at + e]));
                const dist = self.shape.k - 1 - (first + j);
                p.* = dot * 0.08838834764831845;
                if (dist < 1024) p.* += c.fromBf16(self.host[3].items[(r * self.shape.n + h) * 1024 + dist]);
                m = if (std.math.isNan(m) or std.math.isNan(p.*)) std.math.nan(f64) else @max(m, p.*);
            }
            var z: f64 = 0;
            for (probs) |*p| {
                p.* = @exp(p.* - m);
                z += p.*;
            }
            const snk: f64 = c.fromBf16(self.host[5].items[r * self.shape.n + h]);
            const factor = sigmoid(m + @log(z) - snk);
            for (0..128) |e| {
                var sum: f64 = 0;
                var magnitude: f64 = 0;
                var nan = false;
                var pos = false;
                var neg = false;
                for (probs, 0..) |p, j| {
                    const at = (r * cap + (first + j) % cap) * kh * 128 + h / 6 * 128 + e;
                    const v: f64 = c.fromBf16(self.host[2].items[at]);
                    nan = nan or std.math.isNan(v);
                    pos = pos or (std.math.isInf(v) and v > 0);
                    neg = neg or (std.math.isInf(v) and v < 0);
                    if (std.math.isFinite(v)) {
                        sum += p * v;
                        magnitude += p * @abs(v);
                    }
                }
                const raw = if (!std.math.isFinite(m) or !std.math.isFinite(z) or nan or (pos and neg)) std.math.nan(f64) else if (pos) std.math.inf(f64) else if (neg) -std.math.inf(f64) else sum / z;
                const at = (r * self.shape.n + h) * 128 + e;
                const gate: f64 = c.fromBf16(self.host[4].items[at]);
                want[at] = raw * sigmoid(gate) * factor;
                mag[at] = magnitude / z * sigmoid(gate) * factor;
            }
        };
    }

    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        if (!std.mem.eql(u8, out, try self.replaced())) return "exceptional outputs differ from original bytes";
        const a = self.rig.gpu.gpa;
        const buf = try a.alloc(f64, self.count() * 2);
        defer a.free(buf);
        try self.reference(buf[0..self.count()], buf[self.count()..]);
        var nonfinite: usize = 0;
        for (buf[0..self.count()], 0..) |w, i| {
            if (c.classOf(w) != .finite) nonfinite += 1;
            if (c.classOf(w) != c.classOf(c.read(.bf16, out, i))) return "an output class differs from the FP64 formula";
        }
        if (nonfinite == 0 or nonfinite == self.count()) return "nonfinite fixture proves no finite/nonfinite mixture";
        return null;
    }
};

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

pub const Set = struct {
    gpu: check_.Gpu,
    module: cuda.Module,
    cases: [forms.len]Case,

    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !?*Set {
        if (try gpu.ctx.capability() < 120) return null;
        const set = try a.create(Set);
        errdefer a.destroy(set);
        var module = try cuda.Module.load(gpu.d, cuda.kernels.inkling2);
        errdefer module.unload();
        const candidate = try module.function("tf_ik_attention_decode");
        const baseline = try module.function("tf_ik_attention");
        set.* = .{ .gpu = gpu, .module = module, .cases = undefined };
        for (&set.cases, forms) |*x, form| x.* = .{
            .rig = .{ .gpu = gpu, .s = s },
            .form = form,
            .candidate = candidate,
            .baseline = baseline,
        };
        errdefer for (&set.cases) |*x| x.deinit();
        for (&set.cases) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.cases) |*x| x.deinit();
        set.module.unload();
        set.gpu.gpa.destroy(set);
    }
};
