//! The bundled DSpark checkpoint geometry, TP2 ownership and exact BF16 cache budget.
const std = @import("std");
const cfg = @import("config.zig");

pub const hidden = cfg.hidden;
pub const intermediate = 18432;
pub const layers = 5;
pub const block = 8;
pub const proposals = block - 1;
pub const rank = 256;
pub const targets = [_]u32{ 5, 11, 17, 23, 29, 35, 41, 48 };
pub const eps: f32 = 1e-6;
pub const theta: f32 = 1000000;

pub fn local(layer: usize) bool {
    return layer != 2;
}

pub fn capacity(context: usize, layer: usize) usize {
    return if (local(layer)) @min(context + block, cfg.local_window + 256) else context + block;
}

pub fn kvBytes(share: cfg.Share, context: usize) !usize {
    if (context == 0 or context > cfg.maximum_context) return error.InklingContext;
    var count: usize = 0;
    for (0..layers) |i| count += capacity(context, i) * share.keyHeads() * 128 * 4;
    return count;
}

pub fn admit(gpa: std.mem.Allocator, bytes: []const u8) !void {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.InklingDraftConfig;
    const v = doc.value.object.get("mtp_config") orelse return error.InklingDraftConfig;
    if (v != .object) return error.InklingDraftConfig;
    const o = v.object;
    inline for (.{ .{ "hidden_size", hidden }, .{ "num_hidden_layers", layers }, .{ "num_attention_heads", 48 }, .{ "num_key_value_heads", 8 }, .{ "head_dim", 128 }, .{ "intermediate_size", intermediate }, .{ "block_size", block }, .{ "markov_rank", rank }, .{ "sliding_window_size", 1024 } }) |pair| {
        const x = o.get(pair[0]) orelse return error.InklingDraftConfig;
        if (x != .integer or x.integer != pair[1]) return error.InklingDraftConfig;
    }
    const kind = o.get("markov_head_type") orelse return error.InklingDraftConfig;
    if (kind != .string or !std.mem.eql(u8, kind.string, "rnn_hidden")) return error.InklingDraftConfig;
    const taps = o.get("target_layer_ids") orelse return error.InklingDraftConfig;
    if (taps != .array or taps.array.items.len != targets.len) return error.InklingDraftConfig;
    for (taps.array.items, targets) |x, want| if (x != .integer or x.integer != want) return error.InklingDraftConfig;
    const windows = o.get("local_layer_ids") orelse return error.InklingDraftConfig;
    if (windows != .array or windows.array.items.len != 4) return error.InklingDraftConfig;
    for (windows.array.items, [_]usize{ 0, 1, 3, 4 }) |x, want| if (x != .integer or x.integer != want) return error.InklingDraftConfig;
}

test "draft KV includes only its one global history and bounded local rings" {
    const share = try cfg.Share.init(0, 2);
    try std.testing.expectEqual(@as(usize, (524288 + 8 + 4 * 1280) * 4 * 128 * 4), try kvBytes(share, 524288));
    try std.testing.expectError(error.InklingContext, kvBytes(share, 0));
    try std.testing.expectError(error.InklingDraftConfig, admit(std.testing.allocator, "{}"));
}
