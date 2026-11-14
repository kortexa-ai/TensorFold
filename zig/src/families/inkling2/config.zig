//! Inkling 2 Small's text geometry and the published mixed-width EXL3 pack.
const std = @import("std");

pub const hidden = 4608;
pub const intermediate = 2304;
pub const layers = 48;
pub const experts = 384;
pub const topk = 8;
pub const vocab = 201024;
pub const vocabulary = 200058;
pub const block_size = 6;
pub const local_window = 1024;
pub const maximum_context = 1048576;
pub const target_context = 524288;
pub const format = "kortexa-inkling2-exl3-v1";

pub fn local(layer: usize) bool {
    return layer % block_size != block_size - 1;
}

pub fn blocksBefore(layer: usize) usize {
    return if (layer == 0) 0 else (layer + block_size - 1) / block_size;
}

pub const Share = struct {
    rank: usize,
    world: usize,

    pub fn init(rank: usize, world: usize) !Share {
        if ((world != 1 and world != 2) or rank >= world) return error.InklingRanks;
        return .{ .rank = rank, .world = world };
    }

    pub fn expertStart(s: Share) usize {
        return s.rank * s.expertCount();
    }

    pub fn expertCount(s: Share) usize {
        return experts / s.world;
    }

    pub fn queryHeads(s: Share) usize {
        return 48 / s.world;
    }

    pub fn keyHeads(s: Share) usize {
        return 8 / s.world;
    }

    pub fn kvBytes(s: Share, context: usize) !usize {
        if (context == 0 or context > maximum_context) return error.InklingContext;
        // BF16 keys and values; local rings also hold a prompt pass before its attention finishes.
        return (8 * context + 40 * @min(context, local_window + @as(usize, @min(context, 256)))) * s.keyHeads() * 128 * 2 * 2;
    }
};

/// The custom pack retains native activation formats and the exact original text geometry.
pub fn admit(gpa: std.mem.Allocator, bytes: []const u8) !void {
    const p = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer p.deinit();
    const o = if (p.value == .object) p.value.object else return error.InklingCheckpoint;
    const f = o.get("format") orelse return error.InklingCheckpoint;
    if (f != .string or !std.mem.eql(u8, f.string, format)) return error.InklingCheckpoint;
    const status = o.get("status") orelse return error.InklingCheckpoint;
    if (status != .string or !std.mem.eql(u8, status.string, "complete")) return error.InklingIncomplete;
    const c = o.get("text_config") orelse return error.InklingCheckpoint;
    if (c != .object) return error.InklingGeometry;
    inline for (.{
        .{ "hidden_size", hidden },             .{ "num_hidden_layers", layers },               .{ "n_routed_experts", experts },
        .{ "intermediate_size", intermediate }, .{ "num_attention_heads", 48 },                 .{ "num_key_value_heads", 8 },
        .{ "head_dim", 128 },                   .{ "d_rel", 16 },                               .{ "rel_extent", 1024 },
        .{ "num_experts_per_tok", topk },       .{ "n_layers_per_attn_res_block", block_size }, .{ "vocab_size", vocab },
    }) |pair| {
        const v = c.object.get(pair[0]) orelse return error.InklingGeometry;
        if (v != .integer or v.integer != pair[1]) return error.InklingGeometry;
    }
}

test "Inkling TP2 owns whole experts and key-head groups, with exact BF16 KV accounting" {
    const s = try Share.init(1, 2);
    try std.testing.expectEqual(@as(usize, 192), s.expertStart());
    try std.testing.expectEqual(@as(usize, 192), s.expertCount());
    try std.testing.expectEqual(@as(usize, 24), s.queryHeads());
    try std.testing.expectEqual(@as(usize, 4), s.keyHeads());
    try std.testing.expectEqual(@as(usize, 8 * (1 << 30) + 40 * 1280 * 4 * 128 * 4), try s.kvBytes(target_context));
    try std.testing.expectError(error.InklingRanks, Share.init(0, 3));
    try std.testing.expectError(error.InklingRanks, Share.init(2, 2));
    try std.testing.expectError(error.InklingContext, s.kvBytes(maximum_context + 1));
}

test "Inkling depth blocks begin after each boundary layer, with eight global layers" {
    var global: usize = 0;
    for (0..layers) |i| global += @intFromBool(!local(i));
    try std.testing.expectEqual(@as(usize, 8), global);
    try std.testing.expectEqual(@as(usize, 0), blocksBefore(0));
    try std.testing.expectEqual(@as(usize, 1), blocksBefore(6));
    try std.testing.expectEqual(@as(usize, 2), blocksBefore(7));
    try std.testing.expectEqual(@as(usize, 8), blocksBefore(47));
}
