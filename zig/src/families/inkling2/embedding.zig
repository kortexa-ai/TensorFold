//! Vocab-row TP2 ownership; transport selects original BF16 bytes without any arithmetic.
const std = @import("std");
const cfg = @import("config.zig");

pub fn owner(token: usize) !usize {
    if (token >= cfg.vocabulary) return error.InklingToken;
    return token / (cfg.vocab / 2);
}

test "embedding ownership includes both shard edges and excludes padded vocabulary" {
    try std.testing.expectEqual(@as(usize, 0), try owner(0));
    try std.testing.expectEqual(@as(usize, 0), try owner(cfg.vocab / 2 - 1));
    try std.testing.expectEqual(@as(usize, 1), try owner(cfg.vocab / 2));
    try std.testing.expectEqual(@as(usize, 1), try owner(cfg.vocabulary - 1));
    try std.testing.expectError(error.InklingToken, owner(cfg.vocabulary));
    try std.testing.expectError(error.InklingToken, owner(std.math.maxInt(usize)));
}
