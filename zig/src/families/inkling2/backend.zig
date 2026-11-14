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

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &vtable };
    }
    pub fn facts() lanes.Model {
        return .{ .exact_width = 1, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = false, .batch_rows = 1, .max_streams = 1, .copy_drafts = false };
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
        try b.f.s.clear();
        b.active = s;
        b.filling = 0;
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
        if (windows.len != 1 or windows[0].rows() != 1 or windows[0].held != 0) return error.InklingNoDrafts;
        const w = windows[0];
        if (w.masks.len != 0) return error.InklingNoStructures;
        out[0].sampled[0] = @intCast(try queue(p, w.stream, .{ .value = w.pending }, w.positions[0]));
    }
    fn keep(_: *anyopaque, _: []const be.Window, paths: []const []const u32) anyerror!void {
        for (paths) |path| if (path.len != 1 or path[0] != 0) return error.InklingNoDrafts;
    }
    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.InklingNoDrafts;
    }
    fn release(p: *anyopaque, s: *lanes.Stream) void {
        const b = self(p);
        if (b.active == s) b.active = null;
    }
};
