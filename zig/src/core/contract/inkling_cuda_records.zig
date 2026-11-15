//! Inkling's unchanged attention and its byte-preserving output partition, nvcc 13.0.88.
const CudaRecord = @import("registry.zig").CudaRecord;
const none = @import("registry.zig").none[0];

pub const records = [_]CudaRecord{
    .{ .launch = .{ .name = "inkling attention", .symbol = "tf_ik_attention", .image = "inkling2", .threads = 128, .min_sm = 120 }, .usage = .{ none, none, none, .{ .regs = 114, .static = 47428, .bound = 128 }, .{ .regs = 114, .static = 47428, .bound = 128 } } },
    .{ .launch = .{ .name = "inkling decode output partition", .symbol = "tf_ik_attention_decode", .image = "inkling2", .threads = 128, .min_sm = 120 }, .usage = .{ none, none, none, .{ .regs = 110, .static = 47428, .bound = 128 }, .{ .regs = 110, .static = 47428, .bound = 128 } } },
};
