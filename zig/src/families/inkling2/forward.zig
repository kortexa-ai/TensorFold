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
    drafter: ?*@import("draft.zig").Draft = null,
    verifying: bool = false,
    verify_start: usize = 0,
    history_offset: usize = 0,
    verify_mod: ?cuda.Module = null,
    verify_bf16: cuda.Function = undefined,
    verify_f32: cuda.Function = undefined,
    verify_conv: cuda.Function = undefined,
    verify_check: bool = false,
    audit: ?@import("verify_audit.zig").Audit = null,

    pub fn deinit(f: *Forward) void {
        for (f.plans.items) |*p| p.plan.deinit();
        f.plans.deinit(f.gpa);
        if (f.verify_mod) |*m| m.unload();
        if (f.audit) |*a| a.deinit();
    }

    pub fn product(f: *Forward, l: wt.Linear, x: u64, out: u64, rows: usize, kind: cuda.cublaslt.DataType) !void {
        if (f.verifying and rows > 1) {
            if (@import("dense.zig").serves(121, f.lt.api.cublasLtGetVersion(), 1, l.n, l.k, kind)) {
                var args: cuda.Args = .{};
                inline for (.{ x, l.ptr, out, int(l.k), int(l.n), int(rows) }) |v| args.add(v);
                return cuda.launch.launch(if (kind == .bf16) f.verify_bf16 else f.verify_f32, .{ .grid = .{ .x = @intCast(l.n / 8 * rows) }, .block = .{ .x = 32 } }, f.stream, &args);
            }
            const bytes: usize = if (kind == .bf16) 2 else 4;
            for (0..rows) |r| try f.product(l, x + r * l.k * 2, out + r * l.n * bytes, 1, kind);
            return;
        }
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

    pub fn zero(f: *Forward, ptr: u64, bytes: usize) !void {
        try f.d.check(f.d.api.cuMemsetD8Async(ptr, 0, bytes, f.stream.handle), "inkling zero slots");
    }

    pub fn gather(f: *Forward, input: u64, output: u64, bytes: usize) !void {
        if (f.exchange) |e| {
            if (e.group) |g| return g.allGather(f.stream, input, output, bytes);
            try e.lib.check(e.lib.api.ncclAllGather(input, output, bytes, .u8, e.comm, f.stream.handle), "inkling all-gather");
        } else try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(output, input, bytes, f.stream.handle), "inkling local gather");
    }

    pub fn copy(f: *Forward, dst: u64, src: u64, bytes: usize) !void {
        try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, f.stream.handle), "inkling copy");
    }

    pub fn embedding(f: *Forward, ids: []const u32, out: u64) !void {
        const row_bytes = cfg.hidden * 2;
        const first = f.w.embed_first orelse {
            try cuda.copy_source.enqueue(f.d, f.s.ids, std.mem.sliceAsBytes(ids), f.stream);
            return f.k.go(.embed, f.stream, .{ .x = ceil(ids.len * cfg.hidden, 256) }, 256, .{ f.s.ids, f.w.embed, out, int(ids.len), @as(i32, cfg.hidden) });
        };
        const bytes = ids.len * row_bytes;
        try f.zero(f.s.raw_embed, bytes);
        for (ids, 0..) |id, r| {
            if (try @import("embedding.zig").owner(id) == f.w.share.rank)
                try f.copy(f.s.raw_embed + r * row_bytes, f.w.embed + (id - first) * row_bytes, row_bytes);
        }
        try f.gather(f.s.raw_embed, f.s.parts, bytes);
        for (ids, 0..) |id, r| {
            const rank = try @import("embedding.zig").owner(id);
            try f.copy(out + r * row_bytes, f.s.parts + rank * bytes + r * row_bytes, row_bytes);
        }
    }

    pub fn prepareVerify(f: *Forward) !void {
        const m = try cuda.Module.load(f.d, cuda.kernels.inkling2_verify);
        f.verify_mod = m;
        f.verify_bf16 = try m.function("tf_ik_verify_bf16");
        f.verify_f32 = try m.function("tf_ik_verify_f32");
        f.verify_conv = try m.function("tf_ik_verify_conv");
        if (f.verify_check) f.audit = try @import("verify_audit.zig").Audit.init(f.d, f.s.cache.len, f.w.share.keyHeads() * 128);
    }

    fn conv(f: *Forward, c: wt.Conv, input: u64, dynamic_input: u64, history: u64, output: u64, rows: usize) !void {
        if (c.dynamic) |w| try f.product(w, dynamic_input, f.s.dynamic, rows, .bf16);
        try f.k.go(.conv, f.stream, .{ .x = ceil(rows * c.width, 256) }, 256, .{ input, c.weight, if (c.dynamic != null) f.s.dynamic else @as(u64, 0), history, output, int(rows), int(c.width) });
        if (f.verifying) {
            const d = f.drafter orelse return error.InklingNoDrafts;
            var args: cuda.Args = .{};
            inline for (.{ input, history, d.snapshots + f.history_offset * 2, int(rows), int(c.width), int(d.snapshot_stride) }) |v| args.add(v);
            try cuda.launch.launch(f.verify_conv, .{ .grid = .{ .x = ceil(c.width, 256), .y = @intCast(rows + 1) }, .block = .{ .x = 256 } }, f.stream, &args);
            f.history_offset += 3 * c.width;
        }
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
        if (f.verifying and rows > 1) {
            for (0..rows) |r| {
                var a = attention;
                a.rows = 1;
                a.pos += int(r);
                a.q += r * heads * 128 * 2;
                a.relative += r * heads * 1024 * 2;
                a.gate += r * heads * 128 * 2;
                a.sink += r * heads * 2;
                a.out += r * heads * 128 * 2;
                try f.k.go(.attention_decode, stream, .{ .x = 1, .y = @intCast(heads), .z = 4 }, 128, .{a});
            }
        } else if (rows == 1 and f.k.split_decode) {
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
        if (!f.verifying) try f.settleDraft();
        const rows = ids.len;
        const s = f.s;
        if (rows == 0 or rows > s.rows or rows + s.pos > s.context) return error.InklingContext;
        for (ids) |id| if (id >= cfg.vocabulary) return error.InklingToken;
        try f.embedding(ids, s.raw_embed);
        try f.k.go(.norm, f.stream, .{ .x = @intCast(rows) }, 256, .{ s.raw_embed, f.w.embed_norm, s.partial, @as(i32, cfg.hidden) });
        f.history_offset = 0;
        for (f.w.layer, 0..) |l, i| {
            if (f.drafter) |d| for (@import("draft_config.zig").targets, 0..) |tap, j| {
                if (i == tap) try f.copy(d.taps + j * s.rows * cfg.hidden * 2, s.partial, rows * cfg.hidden * 2);
            };
            try f.layer(l, i, rows);
        }
        const start = s.pos;
        s.pos += rows;
        if (!final and f.drafter == null) return;
        const all = f.verifying or f.drafter != null;
        const last = if (all) 0 else (rows - 1) * cfg.hidden * 2;
        const norm_rows = if (all) rows else 1;
        try f.k.go(.mix, f.stream, .{ .x = @intCast(norm_rows) }, 256, .{kern.Mix{ .partial = s.partial + last, .blocks = s.blocks + last, .attention = 0, .query = f.w.output_query, .norm = f.w.norm, .out = s.normed, .rows = int(norm_rows), .width = cfg.hidden, .block_stride = int(s.rows * cfg.hidden), .blocks_count = int(cfg.blocksBefore(f.w.layer.len)), .queries = 1, .mode = 0 }});
        if (f.drafter) |d| {
            try f.copy(d.taps + 7 * s.rows * cfg.hidden * 2, s.normed, rows * cfg.hidden * 2);
            if (!f.verifying) try d.ingest(f, start, rows);
        }
        if (!final) return;
        const d = f.drafter;
        const logits = if (f.verifying) d.?.verify_logits else s.logits;
        const picks = if (f.verifying) d.?.verify_picks else s.top1;
        const n = f.w.head.n;
        const head_rows = if (f.verifying) rows else 1;
        const head_x = s.normed + (if (all and !f.verifying) (rows - 1) * cfg.hidden * 2 else 0);
        try f.product(f.w.head, head_x, logits, head_rows, .f32);
        const first = f.w.share.rank * n;
        try f.k.go(.logits, f.stream, .{ .x = ceil(head_rows * n, 256) }, 256, .{ logits, int(head_rows * n) });
        const mine = picks + f.w.share.rank * head_rows * 16;
        for (0..head_rows) |r| try f.k.go(.top1, f.stream, .{ .x = 1 }, 256, .{ logits + r * n * 4, mine + r * 16, int(@min(n, cfg.vocabulary - first)), int(first) });
        try f.gather(mine, picks, head_rows * 16);
    }

    pub fn verify(f: *Forward, ids: []const u32, out: []u32) !void {
        if (ids.len > 8 or out.len < ids.len or f.drafter == null) return error.InklingDraftRows;
        try f.settleDraft();
        f.verify_start = f.s.pos;
        f.verifying = true;
        defer f.verifying = false;
        try f.run(ids, true);
        for (0..ids.len) |r| {
            try f.copy(f.s.top1, f.drafter.?.verify_picks + r * 16, 16);
            try f.copy(f.s.top1 + 16, f.drafter.?.verify_picks + (ids.len + r) * 16, 16);
            out[r] = try f.pick();
        }
        if (f.verify_check) {
            try f.audit.?.snapshot(f, ids.len);
            try f.restoreHistory(0);
            f.s.pos = f.verify_start;
            const d = f.drafter;
            f.drafter = null;
            f.verifying = false;
            defer f.drafter = d;
            for (ids, 0..) |id, r| {
                try f.run(&.{id}, true);
                const token = try f.pick();
                if (token != out[r]) {
                    std.log.err("inkling verify row {d} at {d}: batched token {d}, plain {d}", .{ r, f.verify_start, out[r], token });
                    return error.InklingVerifyMismatch;
                }
                try f.audit.?.checkRow(f, d.?, r);
            }
        }
    }

    fn restoreHistory(f: *Forward, count: usize) !void {
        const d = f.drafter orelse return error.InklingNoDrafts;
        var offset: usize = 0;
        for (f.s.cache) |c| for (c.conv, 0..) |history, j| {
            const size = 3 * (if (j < 2) f.w.share.keyHeads() * 128 else cfg.hidden);
            try f.copy(history, d.snapshots + (count * d.snapshot_stride + offset) * 2, size * 2);
            offset += size;
        };
    }

    pub fn keep(f: *Forward, count: usize) !void {
        if (f.drafter == null) return error.InklingNoDrafts;
        if (count == 0 or count > f.s.pos - f.verify_start) return error.InklingDraftRows;
        try f.restoreHistory(count);
        f.s.pos = f.verify_start + count;
    }

    pub fn settleDraft(f: *Forward) !void {
        const d = f.drafter orelse return;
        const count = try @import("draft_commit.zig").pending(f.s.pos, d.pos, f.verify_start);
        if (count > 0) try d.ingest(f, d.pos, count);
    }

    pub fn clear(f: *Forward) !void {
        try f.s.clear();
        if (f.drafter) |d| d.pos = 0;
        if (f.audit) |*a| a.checked = 0;
    }

    pub fn pick(f: *Forward) !u32 {
        return f.pickAt(f.s.top1);
    }

    pub fn pickAt(f: *Forward, ptr: u64) !u32 {
        try f.stream.synchronize();
        if (f.exchange) |e| if (e.group) |g| try g.check();
        var picks: [2]kern.Pick = undefined;
        const all = picks[0..f.w.share.world];
        try cuda.memory.download(f.d, ptr, std.mem.sliceAsBytes(all));
        var best: kern.Pick = .{ .value = -std.math.inf(f32), .token = std.math.maxInt(u32), .nonfinite = 0, .pad = 0 };
        for (all) |p| {
            if (p.nonfinite != 0) return error.NonFiniteLogits;
            if (p.value > best.value or (p.value == best.value and p.token < best.token)) best = p;
        }
        return best.token;
    }
};
