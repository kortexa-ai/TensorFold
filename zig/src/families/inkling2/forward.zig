//! Inkling's native text layer graph: attention TP, whole-expert EP, and unmodified shared EXL3 products.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const wt = @import("weights.zig");
const st = @import("state.zig");
const kern = @import("kernels.zig");
const int = kern.int;
const ceil = kern.ceil;

const Plan = struct { rows: usize, k: usize, n: usize, out: cuda.cublaslt.DataType, plan: cuda.cublaslt.Linear };
pub const Exchange = struct { lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, group: ?*cuda.tp_group.Group = null };

pub const Forward = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    w: *const wt.Weights,
    s: *st.State,
    k: *const kern.Kernels,
    lt: *const cuda.cublaslt.Library,
    plans: std.ArrayList(Plan) = .empty,
    exchange: ?Exchange = null,
    decode_reported: bool = false,
    dense_reported: bool = false,
    blas_version: usize = 0,

    pub fn deinit(f: *Forward) void {
        for (f.plans.items) |*p| p.plan.deinit();
        f.plans.deinit(f.gpa);
    }

    fn product(f: *Forward, l: wt.Linear, x: u64, out: u64, rows: usize, kind: cuda.cublaslt.DataType) !void {
        if (f.k.dense) |*dense| {
            if (f.blas_version == 0) f.blas_version = f.lt.api.cublasLtGetVersion();
            if (@import("dense.zig").serves(121, f.blas_version, rows, l.n, l.k, kind)) {
                if (!f.dense_reported) {
                    std.log.info("inkling dense: staged BF16 Q/gate and FP32 O/head, SM121 cuBLASLt 130101", .{});
                    f.dense_reported = true;
                }
                return dense.run(f.stream, x, l.ptr, out, l.n, l.k, kind);
            }
        }
        for (f.plans.items) |*p| if (p.rows == rows and p.k == l.k and p.n == l.n and p.out == kind)
            return p.plan.run(x, l.ptr, out, f.s.workspace, 64 << 20, f.stream.handle);
        var p = try cuda.cublaslt.Linear.init(f.lt, rows, l.n, l.k, kind, 64 << 20);
        errdefer p.deinit();
        try f.plans.append(f.gpa, .{ .rows = rows, .k = l.k, .n = l.n, .out = kind, .plan = p });
        try p.run(x, l.ptr, out, f.s.workspace, 64 << 20, f.stream.handle);
    }

    fn zero(f: *Forward, ptr: u64, bytes: usize) !void {
        try f.d.check(f.d.api.cuMemsetD8Async(ptr, 0, bytes, f.stream.handle), "inkling zero slots");
    }

    fn gather(f: *Forward, input: u64, output: u64, bytes: usize) !void {
        if (f.exchange) |e| {
            if (e.group) |g| return g.allGather(f.stream, input, output, bytes);
            try e.lib.check(e.lib.api.ncclAllGather(input, output, bytes, .u8, e.comm, f.stream.handle), "inkling all-gather");
        } else try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(output, input, bytes, f.stream.handle), "inkling local gather");
    }

    fn conv(f: *Forward, c: wt.Conv, input: u64, dynamic_input: u64, history: u64, output: u64, rows: usize) !void {
        if (c.dynamic) |w| try f.product(w, dynamic_input, f.s.dynamic, rows, .bf16);
        try f.k.go(.conv, f.stream, .{ .x = ceil(rows * c.width, 256) }, 256, .{ input, c.weight, if (c.dynamic != null) f.s.dynamic else @as(u64, 0), history, output, int(rows), int(c.width) });
        try f.k.go(.conv_commit, f.stream, .{ .x = ceil(c.width, 256) }, 256, .{ input, history, int(rows), int(c.width) });
    }

    fn projectExperts(f: *Forward, m: wt.Matrix, input: u64, input_slots: usize, output: u64, rows: usize, splits: usize) !void {
        const pairs = rows * cfg.topk;
        const count = f.w.share.expertCount();
        const items = cuda.grouped.maxItems(pairs, count + 1, 16);
        try f.k.go(.rotate, f.stream, .{ .x = @intCast(pairs), .y = @intCast(m.k / 128) }, 32, .{ input, f.s.picks, m.suh, f.s.rotated, int(m.k), int(input_slots), int(count) });
        try f.k.dot(f.stream, .{ .x0 = f.s.rotated, .x1 = f.s.rotated, .tp0 = m.ptrs, .tp1 = m.ptrs, .k2_0 = m.bits, .k2_1 = m.bits, .uids = f.s.plan.items, .ucount = f.s.plan.counts, .members = f.s.plan.members, .z = f.s.z, .k = int(m.k), .n = int(m.n), .p = int(pairs), .sk = int(splits), .maxm = 16, .slots = cfg.topk, .nu = int(items), .mats = 1, .ns = 0, .standard = 1, .skip = int(count) });
        try f.k.go(.unrotate, f.stream, .{ .x = @intCast(pairs), .y = @intCast(m.n / 128) }, 32, .{ f.s.z, f.s.picks, m.svh, output, int(pairs), int(m.n), int(splits), int(count) });
    }

    fn layer(f: *Forward, l: wt.Layer, index: usize, rows: usize) !void {
        const h = cfg.hidden;
        const s = f.s;
        const stream = f.stream;
        const share = f.w.share;
        const local = cfg.local(index);
        const streams: usize = if (local) 1 else 3;
        const blocks = cfg.blocksBefore(index);
        if (index == 0) {
            try f.k.go(.norm, stream, .{ .x = @intCast(rows) }, 256, .{ s.partial, l.norms, s.streams, int(h) });
        } else try f.k.go(.mix, stream, .{ .x = @intCast(rows), .y = @intCast(streams) }, 256, .{kern.Mix{ .partial = s.partial, .blocks = s.blocks, .attention = 0, .query = l.attn_query, .norm = l.norms, .out = s.streams, .rows = int(rows), .width = h, .block_stride = int(s.rows * h), .blocks_count = int(blocks), .queries = int(streams), .mode = 0 }});
        const qsource = s.streams;
        const ksource = s.streams + (if (local) @as(usize, 0) else rows * h * 2);
        const vsource = s.streams + (if (local) @as(usize, 0) else 2 * rows * h * 2);
        try f.product(l.q, qsource, s.q, rows, .bf16);
        try f.product(l.k, ksource, s.conv_temp, rows, .bf16);
        try f.conv(l.k_conv, s.conv_temp, ksource, s.cache[index].conv[0], s.k, rows);
        try f.product(l.v, vsource, s.conv_temp, rows, .bf16);
        try f.conv(l.v_conv, s.conv_temp, vsource, s.cache[index].conv[1], s.v, rows);
        try f.product(l.r, qsource, s.r, rows, .bf16);
        try f.product(l.gate, qsource, s.gate, rows, .bf16);
        try f.product(l.sink, qsource, s.sink, rows, .bf16);
        const heads = share.queryHeads();
        const keys = share.keyHeads();
        const cache = s.cache[index];
        try f.k.go(.prep, stream, .{ .x = @intCast(heads), .y = @intCast(rows) }, 128, .{ s.q, l.q_scale, s.q, @as(u64, 0), int(heads), @as(i32, 128), int(s.pos), int(cache.capacity), @as(i32, 0), @as(i32, @intFromBool(!local)) });
        try f.k.go(.prep, stream, .{ .x = @intCast(keys), .y = @intCast(rows) }, 128, .{ s.k, @as(u64, 0), s.k, cache.k, int(keys), @as(i32, 128), int(s.pos), int(cache.capacity), @as(i32, 1), @as(i32, 0) });
        try f.k.go(.prep, stream, .{ .x = @intCast(keys), .y = @intCast(rows) }, 128, .{ s.v, @as(u64, 0), s.v, cache.v, int(keys), @as(i32, 128), int(s.pos), int(cache.capacity), @as(i32, 2), @as(i32, 0) });
        try f.k.go(.prep, stream, .{ .x = @intCast(heads), .y = @intCast(rows) }, 128, .{ s.r, l.r_scale, s.r, @as(u64, 0), int(heads), @as(i32, 16), int(s.pos), int(cache.capacity), @as(i32, 3), @as(i32, @intFromBool(!local)) });
        try f.k.go(.relative, stream, .{ .x = 8, .y = @intCast(rows * heads) }, 128, .{ s.r, l.relative, s.relative, int(heads) });
        const attention: kern.Attention = .{ .q = s.q, .k = cache.k, .v = cache.v, .relative = s.relative, .gate = s.gate, .sink = s.sink, .out = s.attn_heads, .rows = int(rows), .pos = int(s.pos), .hq = int(heads), .hk = int(keys), .capacity = int(cache.capacity), .window = if (local) 1023 else 0 };
        if (rows == 1 and f.k.split_decode) {
            try f.k.go(.attention_decode, stream, .{ .x = 1, .y = @intCast(heads), .z = 4 }, 128, .{attention});
            if (!f.decode_reported) {
                std.log.info("inkling attention decode: tf_ik_attention_decode, 4 output partitions per head", .{});
                f.decode_reported = true;
            }
        } else try f.k.go(.attention, stream, .{ .x = ceil(rows, 16), .y = @intCast(heads) }, 128, .{attention});
        const part_bytes = rows * h * 4;
        const mine = s.parts + share.rank * part_bytes;
        try f.product(l.o, s.attn_heads, mine, rows, .f32);
        try f.gather(mine, s.parts, part_bytes);
        try f.k.go(.rank_sum, stream, .{ .x = ceil(rows * h, 256) }, 256, .{ s.parts, s.conv_temp, int(rows * h), int(share.world) });
        try f.conv(l.attn_conv, s.conv_temp, 0, cache.conv[2], s.attention, rows);
        try f.k.go(.mix, stream, .{ .x = @intCast(rows) }, 256, .{kern.Mix{ .partial = s.partial, .blocks = s.blocks, .attention = s.attention, .query = l.mlp_query, .norm = l.mlp_norm, .out = s.mlp_input, .rows = int(rows), .width = h, .block_stride = int(s.rows * h), .blocks_count = int(blocks), .queries = 1, .mode = if (index % cfg.block_size == 0) 1 else 2 }});
        try f.product(l.router, s.mlp_input, s.router, rows, .f32);
        try f.k.go(.route, stream, .{ .x = @intCast(rows) }, 512, .{ s.router, l.router_bias, s.picks, s.gamma, int(share.expertStart()), int(share.expertCount()) });
        try f.k.go(.quant, stream, .{ .x = @intCast(rows) }, 256, .{ s.mlp_input, s.quant_up, int(h), @as(i32, @intFromBool(index < 40)) });
        const pairs = rows * cfg.topk;
        try f.k.plan.route(stream, s.picks, pairs, share.expertCount() + 1, 16, s.plan);
        try f.zero(s.gu, pairs * h * 2);
        try f.zero(s.down, pairs * h * 2);
        try f.projectExperts(l.w13, s.quant_up, cfg.topk, s.gu, rows, 4);
        try f.k.go(.swiglu, stream, .{ .x = ceil(pairs * cfg.intermediate, 256) }, 256, .{ s.gu, s.gamma, s.hidden, int(pairs * cfg.intermediate), @as(i32, cfg.intermediate) });
        try f.k.go(.quant, stream, .{ .x = @intCast(pairs) }, 256, .{ s.hidden, s.quant_down, @as(i32, cfg.intermediate), @as(i32, @intFromBool(index < 40)) });
        try f.projectExperts(l.w2, s.quant_down, 0, s.down, rows, 1);
        try f.gather(s.down, s.gathered, pairs * h * 2);
        try f.k.go(.moe_sum, stream, .{ .x = ceil(rows * h, 256) }, 256, .{ s.gathered, s.conv_temp, int(rows), @as(i32, h), int(share.world) });
        try f.conv(l.mlp_conv, s.conv_temp, 0, cache.conv[3], s.mlp, rows);
        try f.k.go(.finish, stream, .{ .x = ceil(rows * h, 256) }, 256, .{ s.partial, s.blocks + blocks * s.rows * h * 2, s.attention, s.mlp, int(rows * h), @as(i32, @intFromBool(index % cfg.block_size == 0)) });
    }

    pub fn run(f: *Forward, ids: []const u32, final: bool) !void {
        const rows = ids.len;
        const s = f.s;
        if (rows == 0 or rows > s.rows or rows + s.pos > s.context) return error.InklingContext;
        for (ids) |id| if (id >= cfg.vocabulary) return error.InklingToken;
        try cuda.copy_source.enqueue(f.d, s.ids, std.mem.sliceAsBytes(ids), f.stream);
        try f.k.go(.embed, f.stream, .{ .x = ceil(rows * cfg.hidden, 256) }, 256, .{ s.ids, f.w.embed, s.raw_embed, int(rows), @as(i32, cfg.hidden) });
        try f.k.go(.norm, f.stream, .{ .x = @intCast(rows) }, 256, .{ s.raw_embed, f.w.embed_norm, s.partial, @as(i32, cfg.hidden) });
        for (f.w.layer, 0..) |l, i| try f.layer(l, i, rows);
        s.pos += rows;
        if (!final) return;
        const last = (rows - 1) * cfg.hidden * 2;
        try f.k.go(.mix, f.stream, .{ .x = 1 }, 256, .{kern.Mix{ .partial = s.partial + last, .blocks = s.blocks + last, .attention = 0, .query = f.w.output_query, .norm = f.w.norm, .out = s.normed, .rows = 1, .width = cfg.hidden, .block_stride = int(s.rows * cfg.hidden), .blocks_count = int(cfg.blocksBefore(f.w.layer.len)), .queries = 1, .mode = 0 }});
        try f.product(f.w.head, s.normed, s.logits, 1, .f32);
        const n = f.w.head.n;
        const first = f.w.share.rank * n;
        try f.k.go(.logits, f.stream, .{ .x = ceil(n, 256) }, 256, .{ s.logits, int(n) });
        try f.k.go(.top1, f.stream, .{ .x = 1 }, 256, .{ s.logits, s.top1 + f.w.share.rank * 16, int(@min(n, cfg.vocabulary - first)), int(first) });
        try f.gather(s.top1 + f.w.share.rank * 16, s.top1, 16);
    }

    pub fn pick(f: *Forward) !u32 {
        try f.stream.synchronize();
        if (f.exchange) |e| if (e.group) |g| try g.check();
        var picks: [2]kern.Pick = undefined;
        const all = picks[0..f.w.share.world];
        try cuda.memory.download(f.d, f.s.top1, std.mem.sliceAsBytes(all));
        var best: kern.Pick = .{ .value = -std.math.inf(f32), .token = std.math.maxInt(u32), .nonfinite = 0, .pad = 0 };
        for (all) |p| {
            if (p.nonfinite != 0) return error.NonFiniteLogits;
            if (p.value > best.value or (p.value == best.value and p.token < best.token)) best = p;
        }
        return best.token;
    }
};
