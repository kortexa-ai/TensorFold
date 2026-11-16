//! Optional teacher-forced comparison of target logits, hidden rows, exact KV and convolution prefixes.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const Forward = @import("forward.zig").Forward;
const Draft = @import("draft.zig").Draft;

pub const Audit = struct {
    kv: cuda.DeviceBuffer,
    width: usize,
    checked: usize = 0,

    pub fn init(d: *const cuda.Driver, layers: usize, width: usize) !Audit {
        return .{ .kv = try cuda.DeviceBuffer.alloc(d, layers * 8 * 2 * width * 2), .width = width };
    }
    pub fn deinit(a: *Audit) void {
        a.kv.free();
    }
    fn at(a: *Audit, layer: usize, row: usize, kind: usize) u64 {
        return a.kv.ptr + ((layer * 8 + row) * 2 + kind) * a.width * 2;
    }
    pub fn snapshot(a: *Audit, f: *Forward, rows: usize) !void {
        for (f.s.cache, 0..) |cache, layer| for (0..rows) |r| {
            const offset = ((f.verify_start + r) % cache.capacity) * a.width * 2;
            try f.copy(a.at(layer, r, 0), cache.k + offset, a.width * 2);
            try f.copy(a.at(layer, r, 1), cache.v + offset, a.width * 2);
        };
    }
    fn equal(f: *Forward, label: []const u8, layer: usize, row: usize, actual: u64, expected: u64, bytes: usize) !void {
        var got: [32768]u8 = undefined;
        var want: [32768]u8 = undefined;
        var offset: usize = 0;
        while (offset < bytes) {
            const n = @min(got.len, bytes - offset);
            try cuda.memory.download(f.d, actual + offset, got[0..n]);
            try cuda.memory.download(f.d, expected + offset, want[0..n]);
            if (std.mem.indexOfDiff(u8, got[0..n], want[0..n])) |diff| {
                std.log.err("inkling verify {s}: layer {d}, row {d}, byte {d} differs from plain", .{ label, layer, row, offset + diff });
                return error.InklingVerifyMismatch;
            }
            offset += n;
        }
    }
    pub fn checkRow(a: *Audit, f: *Forward, d: *const Draft, r: usize) !void {
        try equal(f, "logits", cfg.layers, r, f.s.logits, d.verify_logits + r * f.w.head.n * 4, f.w.head.n * 4);
        try equal(f, "final hidden", cfg.layers, r, f.s.normed, d.taps + (7 * f.s.rows + r) * cfg.hidden * 2, cfg.hidden * 2);
        var history_offset: usize = 0;
        for (f.s.cache, 0..) |cache, layer| {
            const offset = ((f.verify_start + r) % cache.capacity) * a.width * 2;
            try equal(f, "K", layer, r, cache.k + offset, a.at(layer, r, 0), a.width * 2);
            try equal(f, "V", layer, r, cache.v + offset, a.at(layer, r, 1), a.width * 2);
            for (cache.conv, 0..) |history, j| {
                const size = 3 * (if (j < 2) a.width else cfg.hidden);
                try equal(f, "history", layer, r, history, d.snapshots + ((r + 1) * d.snapshot_stride + history_offset) * 2, size * 2);
                history_offset += size;
            }
        }
        a.checked += 1;
    }
};
