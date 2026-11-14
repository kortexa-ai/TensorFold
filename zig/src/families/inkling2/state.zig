//! Owned scratch, exact BF16 KV caches and raw convolution history; all storage precedes inference.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const weights = @import("weights.zig");

pub const prompt_rows = 256;
pub const Cache = struct { k: u64, v: u64, capacity: usize, conv: [4]u64 };

pub const State = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
    cache: []Cache = &.{},
    rows: usize,
    context: usize,
    pos: usize = 0,
    ids: u64 = 0,
    partial: u64 = 0,
    blocks: u64 = 0,
    streams: u64 = 0,
    raw_embed: u64 = 0,
    q: u64 = 0,
    k: u64 = 0,
    v: u64 = 0,
    r: u64 = 0,
    relative: u64 = 0,
    gate: u64 = 0,
    sink: u64 = 0,
    dynamic: u64 = 0,
    conv_temp: u64 = 0,
    attn_heads: u64 = 0,
    attention: u64 = 0,
    mlp: u64 = 0,
    mlp_input: u64 = 0,
    normed: u64 = 0,
    router: u64 = 0,
    picks: u64 = 0,
    gamma: u64 = 0,
    quant_up: u64 = 0,
    quant_down: u64 = 0,
    rotated: u64 = 0,
    z: u64 = 0,
    gu: u64 = 0,
    hidden: u64 = 0,
    down: u64 = 0,
    gathered: u64 = 0,
    parts: u64 = 0,
    logits: u64 = 0,
    top1: u64 = 0,
    workspace: u64 = 0,
    plan: cuda.grouped.Plan = undefined,

    fn take(s: *State, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(s.d, std.mem.alignForward(usize, @max(bytes, 256), 256));
        errdefer b.free();
        try s.owned.append(s.gpa, b);
        return b.ptr;
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, context: usize) !State {
        if (context == 0 or context > cfg.maximum_context) return error.InklingContext;
        var s: State = .{ .gpa = gpa, .d = d, .context = context, .rows = @min(context, prompt_rows) };
        errdefer s.deinit();
        const r = s.rows;
        const h = cfg.hidden;
        const pairs = r * cfg.topk;
        const heads = w.share.queryHeads();
        const keys = w.share.keyHeads();
        s.cache = try gpa.alloc(Cache, w.layer.len);
        for (s.cache, 0..) |*c, i| {
            const capacity = if (cfg.local(i)) @min(context, cfg.local_window + r) else context;
            c.* = .{ .capacity = capacity, .k = try s.take(capacity * keys * 128 * 2), .v = try s.take(capacity * keys * 128 * 2), .conv = undefined };
            for (&c.conv, 0..) |*p, j| p.* = try s.take(3 * (if (j < 2) keys * 128 else h) * 2);
        }
        s.ids = try s.take(r * 4);
        s.partial = try s.take(r * h * 2);
        s.blocks = try s.take(8 * r * h * 2);
        s.streams = try s.take(3 * r * h * 2);
        s.raw_embed = try s.take(r * h * 2);
        s.q = try s.take(r * heads * 128 * 2);
        s.k = try s.take(r * keys * 128 * 2);
        s.v = try s.take(r * keys * 128 * 2);
        s.r = try s.take(r * heads * 16 * 2);
        s.relative = try s.take(r * heads * 1024 * 2);
        s.gate = try s.take(r * heads * 128 * 2);
        s.sink = try s.take(r * heads * 2);
        s.dynamic = try s.take(r * keys * 4 * 2);
        s.conv_temp = try s.take(r * h * 2);
        s.attn_heads = try s.take(r * heads * 128 * 2);
        s.attention = try s.take(r * h * 2);
        s.mlp = try s.take(r * h * 2);
        s.mlp_input = try s.take(r * h * 2);
        s.normed = try s.take(r * h * 2);
        s.router = try s.take(r * cfg.experts * 4);
        s.picks = try s.take(pairs * 4);
        s.gamma = try s.take(pairs * 4);
        s.quant_up = try s.take(r * h * 2);
        s.quant_down = try s.take(pairs * cfg.intermediate * 2);
        s.rotated = try s.take(pairs * h * 2);
        s.z = try s.take(4 * pairs * h * 4);
        s.gu = try s.take(pairs * h * 2);
        s.hidden = try s.take(pairs * cfg.intermediate * 2);
        s.down = try s.take(pairs * h * 2);
        s.gathered = try s.take(w.share.world * pairs * h * 2);
        s.parts = try s.take(w.share.world * r * h * 4);
        s.logits = try s.take(w.head.n * 4);
        s.top1 = try s.take(w.share.world * 16);
        s.workspace = try s.take(64 << 20);
        const nb = (pairs + 1023) / 1024;
        s.plan = .{ .members = try s.take(pairs * 4), .items = try s.take(cuda.grouped.maxItems(pairs, w.share.expertCount(), 16) * 3 * 4), .counts = try s.take((w.share.expertCount() + 2) * 4), .rank = try s.take(pairs * 4), .hist = try s.take(nb * w.share.expertCount() * 4) };
        try s.clear();
        return s;
    }

    pub fn clear(s: *State) !void {
        // Positional KV is overwritten before read; only recurrent convolution history needs clearing.
        for (s.cache) |c| for (c.conv) |p| {
            // All buffers are individually owned; identify this history's exact allocation.
            for (s.owned.items) |b| if (b.ptr == p) {
                try b.fill8(0, null);
                break;
            };
        };
        s.pos = 0;
    }

    pub fn deinit(s: *State) void {
        for (s.owned.items) |*b| b.free();
        s.owned.deinit(s.gpa);
        s.gpa.free(s.cache);
    }
};
