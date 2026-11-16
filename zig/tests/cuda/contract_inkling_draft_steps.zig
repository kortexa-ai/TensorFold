//! DSpark's elementwise BF16 nodes, checked at each rounding point with independent FP64 intervals.
const std = @import("std");
const cuda = @import("cuda");
const math = @import("contract_moss_image_ref.zig");
const c = math.c;
const g = math.g;
const R = math.Range;
const Form = enum { add, swiglu, gates, logits };
const served = [_][1]c.Shape{
    .{.{ .rows = 8, .n = 4608, .k = 1 }}, .{.{ .rows = 8, .n = 9216, .k = 1 }},
    .{.{ .rows = 1, .n = 256, .k = 1 }},  .{.{ .rows = 8, .n = 100512, .k = 1 }},
};
fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

pub const Case = struct {
    rig: @import("contract_rig.zig").Rig,
    form: Form,
    function: cuda.Function,
    ends: math.Ends,
    shape: c.Shape = undefined,
    host: [4]std.ArrayList(f64) = @splat(.empty),
    inputs: [4]u64 = @splat(0),
    out: u64 = 0,
    latent: u64 = 0,
    control: bool = false,

    pub fn kernel(t: *Case) g.Kernel {
        const name = switch (t.form) {
            .add => "cuda inkling draft add",
            .swiglu => "cuda inkling draft swiglu",
            .gates => "cuda inkling draft recurrent gates",
            .logits => "cuda inkling draft folded logits",
        };
        return .{ .spec = .{ .name = if (t.control) "control inkling draft missing final element" else name, .accum = .none, .depth = math.noSums, .out = if (t.form == .logits) .f32 else .bf16, .served = &served[@backingInt(t.form)], .tiles = .{ .{ .size = 8 }, .{ .size = 256 }, .{} } }, .ctx = t, .vt = &g.hooks(Case), .control = if (t.control) .shapes else null };
    }
    pub fn deinit(t: *Case) void {
        for (&t.host) |*h| h.deinit(t.rig.gpu.gpa);
        t.ends.deinit();
        t.rig.deinit();
    }
    pub fn count(t: *Case) usize {
        return if (t.form == .gates) 512 else t.shape.rows * t.shape.n;
    }
    pub fn prepare(t: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 8 or shape.n == 0 or (t.form == .gates and (shape.rows != 1 or shape.n != 256))) return false;
        t.rig.reset();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const n = t.count();
        var rng = std.Random.DefaultPrng.init(seed);
        for (&t.host, 0..) |*h, j| {
            const size = if (t.form == .gates) (if (j < 2) @as(usize, 768) else 256) else n;
            try h.resize(a, size);
            for (h.items) |*v| v.* = math.bf((rng.random().float(f32) - 0.5) * 16);
            if (fill == .nonfinite) {
                h.items[0] = std.math.nan(f64);
                if (size > 2) {
                    h.items[1] = std.math.inf(f64);
                    h.items[2] = -std.math.inf(f64);
                }
            }
            const wide = (t.form == .gates and j == 0) or (t.form == .logits and j == 1);
            if (wide) {
                const data = try a.alloc(f32, size);
                defer a.free(data);
                for (data, h.items) |*v, x| v.* = @floatCast(x);
                t.inputs[j] = try t.rig.input(std.mem.sliceAsBytes(data));
            } else {
                const data = try a.alloc(u16, size);
                defer a.free(data);
                for (data, h.items) |*v, x| v.* = c.toBf16(@floatCast(x));
                t.inputs[j] = try t.rig.input(std.mem.sliceAsBytes(data));
            }
        }
        t.out = try t.rig.output((if (t.form == .gates) @as(usize, 256) else n) * (if (t.form == .logits) @as(usize, 4) else 2));
        if (t.form == .gates) t.latent = try t.rig.output(256 * 2);
        try t.ends.size(n);
        return true;
    }
    pub fn run(t: *Case) !void {
        const n = t.shape.rows * t.shape.n;
        var args: cuda.Args = .{};
        if (t.form == .add or t.form == .gates) try t.rig.gpu.d.check(t.rig.gpu.d.api.cuMemcpyDtoDAsync_v2(t.out, t.inputs[if (t.form == .add) 0 else 2], (if (t.form == .add) n else 256) * 2, t.rig.s.handle), "draft contract seed");
        switch (t.form) {
            .add => {
                args.add(t.out);
                args.add(t.inputs[1]);
                args.add(@as(i32, @intCast(n)));
            },
            .swiglu => {
                args.add(t.inputs[0]);
                args.add(t.inputs[1]);
                args.add(t.out);
                args.add(@as(i32, @intCast(n)));
            },
            .gates => {
                args.add(t.inputs[0]);
                args.add(t.inputs[1]);
                args.add(t.out);
                args.add(t.latent);
            },
            .logits => {
                args.add(t.inputs[0]);
                args.add(t.inputs[1]);
                args.add(t.inputs[2]);
                args.add(t.out);
                args.add(@as(i32, @intCast(n)));
            },
        }
        try cuda.launch.launch(t.function, .{ .grid = .{ .x = if (t.form == .gates) 1 else @intCast((n + 255) / 256) }, .block = .{ .x = 256 } }, t.rig.s, &args);
        if (t.control) {
            const p = t.out + (n - 1) * 2;
            try t.rig.gpu.d.check(t.rig.gpu.d.api.cuMemsetD8Async(p, 0xff, 2, t.rig.s.handle), "draft planted control");
        }
    }
    pub fn reference(t: *Case, want: []f64, mag: []f64) !void {
        const x = t.host[0].items;
        const y = t.host[1].items;
        const z = t.host[2].items;
        for (0..t.count()) |i| {
            var r: R = undefined;
            switch (t.form) {
                .add => r = R.of(x[i]).add(R.of(y[i])).fp32(0.5).bf16(),
                .swiglu => r = R.of(x[i]).mul(R.of(x[i]).up(sigmoid).fp32(5)).fp32(0.5).bf16().mul(R.of(y[i])).fp32(0.5).bf16(),
                .logits => {
                    const delta = R.of(y[i]).add(R.of(z[i])).fp32(0.5).bf16();
                    const logit = R.of(x[i]).add(delta).fp32(0.5).bf16();
                    r = logit.mul(R.of(1.0 / 30.0)).fp32(0.5).up(math.tanh).fp32(2).mul(R.of(30)).fp32(0.5);
                },
                .gates => {
                    const j = i % 256;
                    const gate = R.of(x[j]).add(R.of(y[j])).fp32(0.5).bf16().up(sigmoid).fp32(5).bf16();
                    const candidate = R.of(x[256 + j]).add(R.of(y[256 + j])).fp32(0.5).bf16().up(math.tanh).fp32(2).bf16();
                    if (i >= 256) r = R.of(x[512 + j]).add(R.of(y[512 + j])).fp32(0.5).bf16().up(math.tanh).fp32(2).bf16() else {
                        const kept = gate.mul(R.of(z[j])).fp32(0.5).bf16();
                        const inverse: R = .{ .lo = 1 - gate.hi, .hi = 1 - gate.lo };
                        const blend = inverse.fp32(0.5).bf16().mul(candidate).fp32(0.5).bf16();
                        r = kept.add(blend).fp32(0.5).bf16();
                    }
                },
            }
            t.ends.put(want, mag, i, if (r.lo == r.hi) r.lo else (r.lo + r.hi) / 2, r);
        }
    }
    pub fn bounds(t: *Case, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Case, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
};

pub const Set = struct {
    module: cuda.Module,
    cases: [5]Case,
    norms: [4]@import("contract_inkling_draft_norm.zig").Case,
    copies: [2]@import("contract_inkling_draft_copy.zig").Case,
    verify: cuda.Module,
    attentions: [2]@import("contract_inkling_draft_attention.zig").Case,
    pub fn init(gpu: @import("check.zig").Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !?*Set {
        if (try gpu.ctx.capability() != 121) return null;
        var module = try cuda.Module.load(gpu.d, cuda.kernels.inkling2_draft);
        errdefer module.unload();
        const set = try a.create(Set);
        var verify = try cuda.Module.load(gpu.d, cuda.kernels.inkling2_verify);
        errdefer verify.unload();
        set.* = .{ .module = module, .verify = verify, .cases = undefined, .norms = undefined, .copies = undefined, .attentions = undefined };
        for (&set.cases, 0..) |*x, i| {
            const form: Form = @fromBackingInt(@intCast(@min(i, 3)));
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = form, .function = try module.function(switch (form) {
                .add => "tf_ik_draft_add",
                .swiglu => "tf_ik_draft_swiglu",
                .gates => "tf_ik_draft_gates",
                .logits => "tf_ik_draft_logits",
            }), .ends = .{ .a = a }, .control = i == 4 };
            if (i == 4) {
                x.form = .add;
                x.function = try module.function("tf_ik_draft_add");
            }
            try list.append(a, x.kernel());
        }
        for (&set.norms, 0..) |*x, i| {
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = @fromBackingInt(@intCast(i)), .function = try module.function(if (i == 0) "tf_ik_draft_mix" else if (i == 3) "tf_ik_draft_norm" else "tf_ik_draft_prep"), .ends = .{ .a = a } };
            try list.append(a, x.kernel());
        }
        for (&set.copies, 0..) |*x, i| {
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = @fromBackingInt(@intCast(i)), .function = if (i == 0) try module.function("tf_ik_draft_store") else try verify.function("tf_ik_verify_conv") };
            try list.append(a, x.kernel());
        }
        for (&set.attentions, 0..) |*x, i| {
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .local = i == 1, .attention = try module.function("tf_ik_draft_attention"), .merge = try module.function("tf_ik_draft_merge") };
            try list.append(a, x.kernel());
        }
        return set;
    }
    pub fn deinit(set: *Set) void {
        for (&set.cases) |*x| x.deinit();
        for (&set.norms) |*x| x.deinit();
        for (&set.copies) |*x| x.deinit();
        for (&set.attentions) |*x| x.deinit();
        set.verify.unload();
        set.module.unload();
        set.cases[0].rig.gpu.gpa.destroy(set);
    }
};
