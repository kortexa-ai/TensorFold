//! The checkpoint's five-layer BF16 DSpark adapter and its committed target-feature cache.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("exl3_config.zig");
const dc = @import("draft_config.zig");
const wt = @import("weights.zig");
const Forward = @import("forward.zig").Forward;
const int = @import("kernels.zig").int;
const ceil = @import("kernels.zig").ceil;
const Op = enum { mix, norm, prep, store, attention, merge, add, swiglu, gates, logits };
const Layer = struct {
    context_norm: u64,
    input_norm: u64,
    post_norm: u64,
    qnorm: u64,
    knorm: u64,
    ctxnorm: u64,
    q: wt.Linear,
    k: wt.Linear,
    v: wt.Linear,
    kc: wt.Linear,
    vc: wt.Linear,
    o: wt.Linear,
    gate: wt.Linear,
    up: wt.Linear,
    down: wt.Linear,
    cache_k: u64,
    cache_v: u64,
    capacity: usize,
};

pub const Draft = struct {
    pos: usize = 0,
    owner: wt.Weights,
    module: cuda.Module,
    ops: std.EnumArray(Op, cuda.Function),
    layers: [5]Layer = undefined,
    taps: u64 = 0,
    alpha: u64 = 0,
    mask: u64 = 0,
    norm: u64 = 0,
    markov_norm: u64 = 0,
    markov_embed: u64 = 0,
    rnn: wt.Linear = undefined,
    rnn_bias: u64 = 0,
    fold: wt.Linear = undefined,
    fold_bias: u64 = 0,
    x: u64 = 0,
    normed: u64 = 0,
    context_x: u64 = 0,
    q: u64 = 0,
    k: u64 = 0,
    v: u64 = 0,
    attention: u64 = 0,
    partials: u64 = 0,
    gate: u64 = 0,
    up: u64 = 0,
    hidden: u64 = 0,
    parts: u64 = 0,
    residual: u64 = 0,
    f_rnn: u64 = 0,
    rnn_input: u64 = 0,
    projection: u64 = 0,
    latent: u64 = 0,
    base_logits: u64 = 0,
    correction: u64 = 0,
    logits: u64 = 0,
    top1: u64 = 0,
    snapshots: u64 = 0,
    snapshot_stride: usize = 0,
    verify_logits: u64 = 0,
    verify_picks: u64 = 0,

    fn take(d: *Draft, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(d.owner.d, @max(256, bytes));
        errdefer b.free();
        try d.owner.buffers.append(d.owner.gpa, b);
        d.owner.bytes += b.len;
        return b.ptr;
    }
    fn tensor(d: *Draft, f: anytype, name: []const u8, shape: []const usize) !u64 {
        return d.owner.put((try wt.get(f, name, .bf16, shape)).bytes);
    }
    fn projectionWeight(d: *Draft, f: anytype, name: []const u8, n: usize, k: usize, columns: bool) !wt.Linear {
        const s = d.owner.share;
        return wt.linear(&d.owner, f, name, n, k, if (columns) .{ 0, n } else .{ s.rank * n / 2, (s.rank + 1) * n / 2 }, if (columns) .{ s.rank * k / 2, (s.rank + 1) * k / 2 } else .{ 0, k });
    }

    pub fn init(gpa: std.mem.Allocator, io: std.Io, driver: *const cuda.Driver, dir: []const u8, share: cfg.Share, context: usize, rows: usize) !Draft {
        var module = try cuda.Module.load(driver, cuda.kernels.inkling2_draft);
        errdefer module.unload();
        var d: Draft = .{ .owner = .{ .gpa = gpa, .d = driver, .share = share }, .module = module, .ops = undefined };
        errdefer d.owner.deinit();
        inline for (comptime std.meta.tags(Op)) |op| d.ops.set(op, try module.function("tf_ik_draft_" ++ @tagName(op)));
        const config_path = try std.fs.path.join(gpa, &.{ dir, "auxiliary/config.json" });
        defer gpa.free(config_path);
        const config_bytes = try std.Io.Dir.cwd().readFileAlloc(io, config_path, gpa, .limited(1 << 20));
        defer gpa.free(config_bytes);
        try dc.admit(gpa, config_bytes);
        var file = try wt.open(gpa, io, dir, "auxiliary/mtp.safetensors");
        defer file.closeCopied(io);
        var fold = try wt.open(gpa, io, dir, "auxiliary/dspark-fold.safetensors");
        defer fold.closeCopied(io);
        d.alpha = try d.tensor(&fold, "alpha", &.{ 6, 8 });
        d.mask = try d.tensor(&file, "mask_embed", &.{4608});
        d.norm = try d.tensor(&file, "norm.weight", &.{4608});
        d.markov_norm = try d.tensor(&file, "markov_head.dflare_norm.weight", &.{4608});
        d.markov_embed = try d.tensor(&file, "markov_head.markov_w1.weight", &.{ 201024, 256 });
        d.rnn = try wt.linear(&d.owner, &file, "markov_head.rnn_proj.weight", 768, 9728, .{ 0, 768 }, .{ 0, 9728 });
        d.rnn_bias = try d.tensor(&file, "markov_head.rnn_proj.bias", &.{768});
        d.fold = try d.projectionWeight(&fold, "weight", cfg.vocab, 256, false);
        const bias = try wt.get(&fold, "bias", .bf16, &.{cfg.vocab});
        d.fold_bias = try d.owner.put(bias.bytes[share.rank * d.fold.n * 2 ..][0 .. d.fold.n * 2]);
        for (&d.layers, 0..) |*l, i| {
            var name: [160]u8 = undefined;
            inline for (.{ .{ "context_norm", "dflare_norm.weight", 4608 }, .{ "input_norm", "input_layernorm.weight", 4608 }, .{ "post_norm", "post_attention_layernorm.weight", 4608 }, .{ "qnorm", "self_attn.q_norm.weight", 128 }, .{ "knorm", "self_attn.k_norm.weight", 128 }, .{ "ctxnorm", "self_attn.k_norm_ctx.weight", 128 } }) |p| {
                @field(l, p[0]) = try d.tensor(&file, try std.fmt.bufPrint(&name, "layers.{d}.{s}", .{ i, p[1] }), &.{p[2]});
            }
            inline for (.{ .{ "q", "self_attn.q_proj.weight", 6144, 4608, false }, .{ "k", "self_attn.k_proj.weight", 1024, 4608, false }, .{ "v", "self_attn.v_proj.weight", 1024, 4608, false }, .{ "kc", "self_attn.k_proj_ctx.weight", 1024, 4608, false }, .{ "vc", "self_attn.v_proj_ctx.weight", 1024, 4608, false }, .{ "o", "self_attn.o_proj.weight", 4608, 6144, true }, .{ "gate", "mlp.gate_proj.weight", 18432, 4608, false }, .{ "up", "mlp.up_proj.weight", 18432, 4608, false }, .{ "down", "mlp.down_proj.weight", 4608, 18432, true } }) |p| {
                @field(l, p[0]) = try d.projectionWeight(&file, try std.fmt.bufPrint(&name, "layers.{d}.{s}", .{ i, p[1] }), p[2], p[3], p[4]);
            }
            l.capacity = dc.capacity(context, i);
            l.cache_k = try d.take(l.capacity * 512 * 2);
            l.cache_v = try d.take(l.capacity * 512 * 2);
        }
        d.taps = try d.take(8 * rows * 4608 * 2);
        d.x = try d.take(8 * 4608 * 2);
        d.normed = try d.take(8 * 4608 * 2);
        d.context_x = try d.take(rows * 4608 * 2);
        d.q = try d.take(8 * 3072 * 2);
        d.k = try d.take(rows * 512 * 2);
        d.v = try d.take(rows * 512 * 2);
        d.attention = try d.take(8 * 3072 * 2);
        d.partials = try d.take(8 * 24 * 32 * 130 * 4);
        d.gate = try d.take(8 * 9216 * 2);
        d.up = try d.take(8 * 9216 * 2);
        d.hidden = try d.take(8 * 9216 * 2);
        d.parts = try d.take(2 * 8 * 4608 * 4);
        d.residual = try d.take(8 * 4608 * 2);
        d.f_rnn = try d.take(4608 * 2);
        d.rnn_input = try d.take(9728 * 2);
        d.projection = try d.take(768 * 4);
        d.latent = try d.take(256 * 2);
        d.base_logits = try d.take(8 * d.fold.n * 2);
        d.correction = try d.take(d.fold.n * 4);
        d.logits = try d.take(d.fold.n * 4);
        d.top1 = try d.take(32);
        d.snapshot_stride = cfg.layers * 3 * (2 * 512 + 2 * 4608);
        d.snapshots = try d.take(9 * d.snapshot_stride * 2);
        d.verify_logits = try d.take(8 * d.fold.n * 4);
        d.verify_picks = try d.take(8 * 32);
        std.log.info("inkling DSpark: {d:.3} GiB weights, KV and scratch per rank", .{@as(f64, @floatFromInt(d.owner.bytes)) / (1 << 30)});
        return d;
    }

    pub fn deinit(d: *Draft) void {
        d.module.unload();
        d.owner.deinit();
    }
    fn go(d: *Draft, f: *Forward, comptime op: Op, grid: cuda.Dim3, threads: u32, values: anytype) !void {
        var args: cuda.Args = .{};
        inline for (values) |v| args.add(v);
        try cuda.launch.launch(d.ops.get(op), .{ .grid = grid, .block = .{ .x = threads } }, f.stream, &args);
    }
    pub fn ingest(d: *Draft, f: *Forward, pos: usize, count: usize) !void {
        if (count == 0) return error.InklingDraftRows;
        if (pos != d.pos) return error.InklingDraftPosition;
        for (d.layers, 0..) |l, i| {
            try d.go(f, .mix, .{ .x = @intCast(count) }, 256, .{ d.taps, d.alpha + i * 16, l.context_norm, d.context_x, int(f.s.rows * 4608) });
            try f.product(l.kc, d.context_x, d.k, count, .bf16);
            try f.product(l.vc, d.context_x, d.v, count, .bf16);
            try d.go(f, .prep, .{ .x = 4, .y = @intCast(count) }, 128, .{ d.k, l.ctxnorm, d.k, l.cache_k, @as(i32, 4), int(pos), int(l.capacity) });
            try d.go(f, .store, .{ .x = ceil(count * 512, 256) }, 256, .{ d.v, l.cache_v, @as(i32, 512), int(pos), int(l.capacity), int(count) });
        }
        try d.go(f, .mix, .{ .x = 1 }, 256, .{ d.taps + (count - 1) * 4608 * 2, d.alpha + 5 * 16, d.markov_norm, d.f_rnn, int(f.s.rows * 4608) });
        d.pos = pos + count;
    }
    fn reduce(d: *Draft, f: *Forward, l: wt.Linear, x: u64) !void {
        const bytes = 8 * 4608 * 4;
        const mine = d.parts + f.w.share.rank * bytes;
        try f.product(l, x, mine, 8, .f32);
        try f.gather(mine, d.parts, bytes);
        try f.k.go(.rank_sum, f.stream, .{ .x = ceil(8 * 4608, 256) }, 256, .{ d.parts, d.residual, @as(i32, 8 * 4608), @as(i32, 2) });
        try d.go(f, .add, .{ .x = ceil(8 * 4608, 256) }, 256, .{ d.x, d.residual, @as(i32, 8 * 4608) });
    }
    pub fn propose(d: *Draft, f: *Forward, anchor: u32, depth: usize, out: []u32) !void {
        try f.settleDraft();
        if (d.pos != f.s.pos) return error.InklingDraftPosition;
        if (anchor >= cfg.vocabulary or depth > 7 or out.len < depth) return error.InklingDraftRows;
        if (depth == 0) return;
        try f.embedding(&.{anchor}, d.x);
        for (1..8) |r| try f.copy(d.x + r * 4608 * 2, d.mask, 4608 * 2);
        for (d.layers, 0..) |l, i| {
            try d.go(f, .norm, .{ .x = 8 }, 256, .{ d.x, l.input_norm, d.normed, @as(i32, 4608) });
            try f.product(l.q, d.normed, d.q, 8, .bf16);
            try f.product(l.k, d.normed, d.k, 8, .bf16);
            try f.product(l.v, d.normed, d.v, 8, .bf16);
            try d.go(f, .prep, .{ .x = 24, .y = 8 }, 128, .{ d.q, l.qnorm, d.q, @as(u64, 0), @as(i32, 24), int(f.s.pos), int(l.capacity) });
            try d.go(f, .prep, .{ .x = 4, .y = 8 }, 128, .{ d.k, l.knorm, d.k, l.cache_k, @as(i32, 4), int(f.s.pos), int(l.capacity) });
            try d.go(f, .store, .{ .x = 16 }, 256, .{ d.v, l.cache_v, @as(i32, 512), int(f.s.pos), int(l.capacity), @as(i32, 8) });
            try d.go(f, .attention, .{ .x = 24, .y = 32 }, 256, .{ d.q, l.cache_k, l.cache_v, d.partials, int(f.s.pos), @as(i32, 24), @as(i32, 4), int(l.capacity), @as(i32, @intFromBool(dc.local(i))) });
            try d.go(f, .merge, .{ .x = 24, .y = 8 }, 128, .{ d.partials, d.attention, @as(i32, 24) });
            try d.reduce(f, l.o, d.attention);
            try d.go(f, .norm, .{ .x = 8 }, 256, .{ d.x, l.post_norm, d.normed, @as(i32, 4608) });
            try f.product(l.gate, d.normed, d.gate, 8, .bf16);
            try f.product(l.up, d.normed, d.up, 8, .bf16);
            try d.go(f, .swiglu, .{ .x = 288 }, 256, .{ d.gate, d.up, d.hidden, @as(i32, 8 * 9216) });
            try d.reduce(f, l.down, d.hidden);
        }
        try d.go(f, .norm, .{ .x = 8 }, 256, .{ d.x, d.norm, d.normed, @as(i32, 4608) });
        try f.product(f.w.head, d.normed, d.base_logits, 8, .bf16);
        try f.zero(d.rnn_input, 256 * 2);
        var previous = anchor;
        for (0..depth) |step| {
            try f.copy(d.rnn_input + 256 * 2, d.markov_embed + @as(usize, previous) * 256 * 2, 256 * 2);
            try f.copy(d.rnn_input + 512 * 2, d.normed + (step + 1) * 4608 * 2, 4608 * 2);
            try f.copy(d.rnn_input + (512 + 4608) * 2, d.f_rnn, 4608 * 2);
            try f.product(d.rnn, d.rnn_input, d.projection, 1, .f32);
            try d.go(f, .gates, .{ .x = 1 }, 256, .{ d.projection, d.rnn_bias, d.rnn_input, d.latent });
            try f.product(d.fold, d.latent, d.correction, 1, .f32);
            try d.go(f, .logits, .{ .x = ceil(d.fold.n, 256) }, 256, .{ d.base_logits + (step + 1) * d.fold.n * 2, d.correction, d.fold_bias, d.logits, int(d.fold.n) });
            const first = f.w.share.rank * d.fold.n;
            try f.k.go(.top1, f.stream, .{ .x = 1 }, 256, .{ d.logits, d.top1 + f.w.share.rank * 16, int(@min(d.fold.n, cfg.vocabulary - first)), int(first) });
            try f.gather(d.top1 + f.w.share.rank * 16, d.top1, 16);
            previous = try f.pickAt(d.top1);
            out[step] = previous;
        }
    }
};
