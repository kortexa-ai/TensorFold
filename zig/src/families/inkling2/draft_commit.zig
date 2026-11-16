//! Accepted target features may await absorption even when the core needed no rollback callback.
const std = @import("std");
const block = @import("draft_config.zig").block;

pub fn pending(target: usize, draft: usize, start: usize) !usize {
    if (target == draft) return 0;
    if (draft > target or draft != start or target - draft > block) return error.InklingDraftPosition;
    return target - draft;
}

test "fully accepted windows need absorption without a keep call, and partial prefixes absorb once" {
    try std.testing.expectEqual(@as(usize, 4), try pending(104, 100, 100));
    try std.testing.expectEqual(@as(usize, 2), try pending(102, 100, 100));
    try std.testing.expectEqual(@as(usize, 0), try pending(104, 104, 100));
    try std.testing.expectEqual(@as(usize, 0), try pending(0, 0, 0));
    try std.testing.expectError(error.InklingDraftPosition, pending(104, 99, 100));
    try std.testing.expectError(error.InklingDraftPosition, pending(99, 100, 100));
    try std.testing.expectError(error.InklingDraftPosition, pending(109, 100, 100));
}
