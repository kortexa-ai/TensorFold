//! Initial greedy text profile on the shared lane engine. One request owns this pair's fixed cache arena.
const std = @import("std");
const lanes = @import("lanes");
const forward = @import("forward.zig");
const link = @import("link.zig");
const be = lanes.backend;

pub const Backend = struct {
    f: *forward.Forward,
    peer: ?*link.Link = null,
    active: ?*lanes.Stream = null,
    filling: usize = 0,
    token: u32 = 0,
    held: [7]u32 = undefined,
    held_n: usize = 0,
    rounds: usize = 0,
    verified: usize = 0,

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &vtable };
    }
    pub fn facts(drafting: bool) lanes.Model {
        return .{ .exact_width = if (drafting) 8 else 1, .gpu_tokens = false, .mtp = drafting, .speculate = drafting, .speculate_early = false, .drafts = if (drafting) 7 else 0, .hidden_rows = false, .batch_rows = 1, .max_streams = 1, .copy_drafts = false };
    }
    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = step, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true };
    fn self(p: *anyopaque) *Backend {
        return @ptrCast(@alignCast(p));
    }

    fn begin(b: *Backend, s: *lanes.Stream) !void {
        if (b.active != null and b.active != s) return error.InklingBusy;
        if (s.images.len > 0) return error.InklingTextOnly;
        if (s.sampling) |sp| if (sp.temperature > 0) return error.InklingGreedyOnly;
        if (s.prompt().len == 0 or s.prompt().len + s.max_new > b.f.s.context) return error.InklingContext;
        if (b.peer) |p| try p.send(.begin, &.{}, false);
        try b.f.clear();
        b.active = s;
        b.filling = 0;
        b.held_n = 0;
        b.rounds = 0;
        b.verified = 0;
        s.cached = 0;
        if (s.reuse.saved != null) s.reuse_failed = true;
    }

    fn run(b: *Backend, ids: []const u32, final: bool) !void {
        if (b.peer) |p| try p.send(.rows, ids, final);
        try b.f.run(ids, final);
        if (final) b.token = try b.f.pick();
    }

    fn step(p: *anyopaque, s: *lanes.Stream, max_rows: u32) anyerror!bool {
        const b = self(p);
        if (b.active != s) try b.begin(s);
        if (s.isCancelled()) return error.Cancelled;
        const prompt = s.prompt();
        const count = if (max_rows == 0) b.f.s.rows else @min(max_rows, b.f.s.rows);
        const end = @min(prompt.len, b.filling + count);
        const done = end == prompt.len;
        try b.run(prompt[b.filling..end], done);
        b.filling = end;
        s.fill_at.store(@intCast(end), .monotonic);
        return done;
    }

    fn prefill(p: *anyopaque, s: *lanes.Stream) anyerror!void {
        while (!try step(p, s, 0)) {}
    }
    fn first(p: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = self(p);
        if (b.active != s or position != b.f.s.pos) return error.InklingPosition;
        return b.token;
    }
    fn queue(p: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const b = self(p);
        if (b.active != s or position != b.f.s.pos + 1) return error.InklingPosition;
        const id: u32 = switch (feed) {
            .value => |v| v,
            .handle => |v| @intCast(v),
        };
        try b.run(&.{id}, true);
        return b.token;
    }
    fn read(_: *anyopaque, value: u64) anyerror!u32 {
        return @intCast(value);
    }
    fn verify(p: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(p);
        if (windows.len != 1 or out.len != 1) return error.InklingSingleRequest;
        const w = windows[0];
        if (w.masks.len != 0) return error.InklingNoStructures;
        if (w.parents != null or w.rows() > 8 or w.held > b.held_n) return error.InklingDraftRows;
        if (b.active != w.stream) return error.InklingPosition;
        for (w.positions, 0..) |position, i| if (position != b.f.s.pos + 1 + i) return error.InklingPosition;
        if (b.f.drafter == null) {
            if (w.rows() != 1) return error.InklingNoDrafts;
            out[0].sampled[0] = @intCast(try queue(p, w.stream, .{ .value = w.pending }, w.positions[0]));
            return;
        }
        var ids: [8]u32 = undefined;
        ids[0] = w.pending;
        const drafts = if (w.held > 0) b.held[0..w.held] else w.tokens;
        @memcpy(ids[1..][0..drafts.len], drafts);
        if (b.peer) |peer| try peer.send(.verify, ids[0..w.rows()], true);
        try b.f.verify(ids[0..w.rows()], out[0].sampled);
        b.rounds += 1;
        b.verified += w.rows();
        @memcpy(out[0].drafts[0..drafts.len], drafts);
    }
    fn keep(p: *anyopaque, _: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(p);
        if (paths.len != 1 or paths[0].len == 0) return error.InklingDraftRows;
        const path = paths[0];
        for (path, 0..) |row, i| if (row != i) return error.InklingDraftRows;
        if (b.f.drafter == null) {
            if (path.len != 1) return error.InklingNoDrafts;
            return;
        }
        if (b.peer) |peer| try peer.send(.keep, &.{@intCast(path.len)}, false);
        try b.f.keep(path.len);
    }
    fn draft(p: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = self(p);
        for (requests) |r| {
            if (r.lanes != null or r.depth > 7) return error.InklingDraftRows;
            b.held_n = 0;
            if (r.depth == 0) continue;
            const d = b.f.drafter orelse return error.InklingNoDrafts;
            if (r.stream != b.active or r.position != b.f.s.pos + 1) return error.InklingPosition;
            const anchor = if (r.follow.len > 0) r.follow[r.follow.len - 1] else switch (r.first orelse return error.InklingDraftRows) {
                .value => |v| v,
                .handle => |v| @as(u32, @intCast(v)),
            };
            if (b.peer) |peer| try peer.send(.draft, &.{ anchor, r.depth }, false);
            try d.propose(b.f, anchor, r.depth, &b.held);
            b.held_n = r.depth;
        }
    }
    fn release(p: *anyopaque, s: *lanes.Stream) void {
        const b = self(p);
        if (b.active == s) {
            if (b.f.drafter != null) std.log.info("inkling DSpark: {d} verification windows, {d} input rows", .{ b.rounds, b.verified });
            if (b.f.audit) |a| std.log.info("inkling verification audit: {d} rows checked for logits, hidden, KV and history bytes", .{a.checked});
            b.active = null;
        }
    }
};
