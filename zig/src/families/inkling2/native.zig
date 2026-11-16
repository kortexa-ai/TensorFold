//! Inkling's first native text profile on two GB10s, using the shared lane engine and paired transport.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const core = @import("core");
const environment = @import("sys").environment;
const cfg = @import("config.zig");
const wt = @import("weights.zig");
const state = @import("state.zig");
const kernels = @import("kernels.zig");
const forward = @import("forward.zig");
const backend = @import("backend.zig");
const link = @import("link.zig");

pub const model_type = "inkling2";
pub const greedy_only = true;
pub const formats: []const []const u8 = &.{"exl3"};
pub const default_context: i64 = cfg.target_context;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = state.prompt_rows;
pub const snap_lookahead: u32 = 0;
pub const explains_open = true;
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 1, lanes_fixed: bool = false, room: ?Room = null };
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    target: *anyopaque,
};

const Owned = struct {
    gpa: std.mem.Allocator,
    w: wt.Weights,
    s: state.State,
    k: kernels.Kernels,
    stream: cuda.Stream,
    lt: cuda.cublaslt.Library,
    f: forward.Forward,
    b: backend.Backend,
    lib: ?cuda.nccl.Library = null,
    comm: cuda.nccl.Comm = null,
    group: ?cuda.tp_group.Group = null,
    peer: ?link.Link = null,
    draft: ?@import("draft.zig").Draft = null,
};

pub fn place() !cfg.Share {
    return cfg.Share.init(try environment.read(usize, "TF_INKLING_RANK", 0), try environment.read(usize, "TF_INKLING_WORLD", 1));
}

pub fn admit(sm: u32, world: usize) !void {
    if (sm != 121) return error.InklingNeedsGb10;
    if (world != 2) return error.InklingNeedsPair;
}

fn manifest(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![]u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, "checkpoint.json" });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
}

pub fn tensorWeightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, _: bool) !u64 {
    const bytes = try manifest(gpa, io, dir);
    defer gpa.free(bytes);
    try cfg.admit(gpa, bytes);
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer doc.deinit();
    const n = doc.value.object.get("total_bytes") orelse return error.InklingCheckpoint;
    if (n != .integer or n.integer <= 0) return error.InklingCheckpoint;
    return @intCast(n.integer);
}

pub fn weightShare(_: std.mem.Allocator, _: std.Io, _: []const u8, total: u64, _: bool) u64 {
    const share = place() catch return total;
    // Conservative admission: count every global and router whole in addition to the rank's model fraction.
    return total / share.world + 4 * (1 << 30);
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, captured: ?[]const u8, o: Options) !Loaded {
    if (captured != null) return error.InklingCaptured;
    if (o.lanes_fixed and o.lanes > 1) return error.InklingSingleRequest;
    const share = try place();
    try admit(try ctx.capability(), share.world);
    try cuda.tp_watch.start(io, share.rank, share.world);
    return openPair(gpa, io, ctx, dir, o, share) catch |err| {
        cuda.tp_watch.fail(@errorName(err));
        return err;
    };
}

fn openPair(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options, share: cfg.Share) !Loaded {
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .w = undefined, .s = undefined, .k = undefined, .stream = undefined, .lt = undefined, .f = undefined, .b = undefined };
    own.lib = try cuda.nccl.Library.open();
    errdefer own.lib.?.close();
    try own.lib.?.requireSupportedVersion();
    var id: cuda.nccl.UniqueId = undefined;
    if (share.rank == 0) {
        try own.lib.?.check(own.lib.?.api.ncclGetUniqueId(&id), "inkling communicator id");
        cuda.tp_watch.sendId(&id.internal);
    } else id.internal = try cuda.tp_watch.takeId(io);
    try own.lib.?.check(own.lib.?.api.ncclCommInitRank(&own.comm, 2, id, @intCast(share.rank)), "inkling communicator");
    errdefer _ = own.lib.?.api.ncclCommAbort(own.comm);
    own.stream = try cuda.Stream.init(ctx.d, false);
    errdefer own.stream.deinit();
    own.k = try kernels.Kernels.load(ctx.d);
    errdefer own.k.deinit();
    own.lt = try cuda.cublaslt.Library.open();
    errdefer own.lt.close();
    var cache = core.checkpoint.cache.Load.begin(io, true);
    defer cache.deinit();
    own.w = try wt.load(gpa, io, ctx.d, dir, share, cfg.layers);
    errdefer own.w.deinit();
    const kv = try share.kvBytes(o.context);
    if (o.room) |r| if ((r.read(r.ctx) orelse 0) < kv + (1 << 30)) return error.InklingKvRoom;
    own.s = try state.State.init(gpa, ctx.d, &own.w, o.context);
    errdefer own.s.deinit();
    own.group = .{ .lib = own.lib.?, .comm = own.comm, .rank = share.rank, .world = 2 };
    try own.group.?.openRoce(gpa, io, ctx.d, own.stream, 1 << 20);
    errdefer if (own.group.?.fast) |f| f.deinit();
    own.peer = try link.Link.init(ctx.d, &own.lib.?, own.comm, own.stream, own.s.rows);
    errdefer own.peer.?.deinit();
    own.f = .{ .gpa = gpa, .d = ctx.d, .stream = own.stream, .w = &own.w, .s = &own.s, .k = &own.k, .lt = &own.lt, .exchange = .{ .lib = &own.lib.?, .comm = own.comm, .group = &own.group.? } };
    if (o.drafts) {
        own.draft = try @import("draft.zig").Draft.init(gpa, io, ctx.d, dir, share, o.context, own.s.rows);
        own.f.drafter = &own.draft.?;
        own.f.verify_check = try environment.read(bool, "TF_INKLING_VERIFY_CHECK", false);
        try own.f.prepareVerify();
    }
    errdefer if (own.draft) |*d| d.deinit();
    own.b = .{ .f = &own.f, .peer = &own.peer.? };
    const bytes = try manifest(gpa, io, dir);
    defer gpa.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    try cache.finishAfter(.{ io, digest }, cuda.tp_watch.ready);
    if (share.rank != 0) {
        own.peer.?.follow(&own.f, gpa) catch |err| {
            cuda.tp_watch.fail(@errorName(err));
            return err;
        };
        cuda.tp_watch.ended();
        release(own);
        std.c._exit(0);
    }
    return .{ .backend = own.b.backend(), .facts = backend.Backend.facts(o.drafts), .rows = if (o.drafts) 8 else 1, .stream_bytes = kv, .ctx = own, .deinit = release, .target = own };
}

fn release(p: *anyopaque) void {
    const o: *Owned = @ptrCast(@alignCast(p));
    if (o.w.share.rank == 0) if (o.peer) |*l| {
        l.send(.end, &.{}, false) catch {};
        o.stream.synchronize() catch {};
    };
    cuda.tp_watch.ended();
    o.f.deinit();
    if (o.peer) |*l| l.deinit();
    if (o.group) |*g| if (g.fast) |f| f.deinit();
    if (o.draft) |*d| d.deinit();
    o.s.deinit();
    o.w.deinit();
    o.lt.close();
    o.k.deinit();
    o.stream.deinit();
    if (o.lib) |*l| {
        _ = l.api.ncclCommDestroy(o.comm);
        l.close();
    }
    o.gpa.destroy(o);
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.InklingNeedsGb10, error.InklingNeedsPair => "Inkling 2's native text profile requires two GB10 DGX Sparks",
        error.InklingRanks => "TF_INKLING_RANK must be 0 or 1 of TF_INKLING_WORLD=2",
        error.InklingContext => "Inkling's prompt and reply must fit the configured context, at most 1048576 tokens",
        error.InklingCheckpoint, error.InklingIncomplete => "Inkling requires the complete published kortexa Inkling EXL3 pack",
        error.InklingGeometry, error.InklingExl3, error.InklingMissingTensor, error.InklingTensorShape => "Inkling checkpoint geometry or stored tensor layout is unsupported",
        error.InklingKvRoom => "Insufficient memory remains for Inkling's exact BF16 cache and scratch; lower --context",
        error.InklingCaptured => "Inkling uses native operators; captured-kernel directories are unsupported",
        error.InklingSingleRequest, error.InklingBusy => "Inkling's initial native text profile holds one request; use --parallel 1",
        error.InklingTextOnly => "Inkling's initial native profile supports text only",
        error.InklingGreedyOnly => "Inkling's initial native profile supports greedy decoding; set temperature to 0",
        error.InklingNoDrafts => "Inkling's native profile has no qualified draft head or copy windows",
        error.InklingNoStructures => "Inkling's initial native profile has no structured-output path",
        error.NonFiniteLogits => "Inkling produced non-finite logits; generation stopped",
        else => null,
    };
}

test "unqualified device/rank layouts are refused before loading" {
    try admit(121, 2);
    try std.testing.expectError(error.InklingNeedsGb10, admit(120, 2));
    try std.testing.expectError(error.InklingNeedsPair, admit(121, 1));
}
