//! Inkling's unchanged attention and its byte-preserving output partition, nvcc 13.0.88.
const CudaRecord = @import("registry.zig").CudaRecord;
const none = @import("registry.zig").none[0];

pub const records = [_]CudaRecord{
    .{ .launch = .{ .name = "inkling dense bf16", .symbol = "drama3_staged_gemv", .image = "drama3_staged_gemv", .threads = 32, .min_sm = 120 }, .usage = .{ none, none, none, .{ .regs = 118, .static = 12288, .bound = 32 }, .{ .regs = 118, .static = 12288, .bound = 32 } } },
    .{ .launch = .{ .name = "inkling dense f32", .symbol = "tf_ik_dense_gemv_f32", .image = "inkling2_dense", .threads = 32, .min_sm = 120 }, .usage = .{ none, none, none, .{ .regs = 118, .static = 12288, .bound = 32 }, .{ .regs = 118, .static = 12288, .bound = 32 } } },
    .{ .launch = .{ .name = "inkling attention", .symbol = "tf_ik_attention", .image = "inkling2", .threads = 128, .min_sm = 120 }, .usage = .{ none, none, none, .{ .regs = 114, .static = 46404, .bound = 128 }, .{ .regs = 114, .static = 46404, .bound = 128 } } },
    .{ .launch = .{ .name = "inkling decode output partition", .symbol = "tf_ik_attention_decode", .image = "inkling2", .threads = 128, .min_sm = 120 }, .usage = .{ none, none, none, .{ .regs = 110, .static = 46404, .bound = 128 }, .{ .regs = 110, .static = 46404, .bound = 128 } } },
    .{ .launch = .{ .name = "inkling gqa G2 S4", .symbol = "tf_ik_gqa_2_4", .image = "inkling2_gqa", .threads = 128, .min_sm = 120 }, .usage = .{ none, none, none, .{ .regs = 95, .static = 46408, .bound = 128 }, .{ .regs = 95, .static = 46408, .bound = 128 } } },
    .{ .launch = .{ .name = "inkling gqa G2 S8", .symbol = "tf_ik_gqa_2_8", .image = "inkling2_gqa", .threads = 128, .min_sm = 120 }, .usage = .{ none, none, none, .{ .regs = 95, .static = 46408, .bound = 128 }, .{ .regs = 95, .static = 46408, .bound = 128 } } },
    .{ .launch = .{ .name = "inkling gqa G3 S8", .symbol = "tf_ik_gqa_3_8", .image = "inkling2_gqa", .threads = 128, .min_sm = 120 }, .usage = .{ none, none, none, .{ .regs = 95, .static = 46408, .bound = 128 }, .{ .regs = 95, .static = 46408, .bound = 128 } } },
    .{ .launch = .{ .name = "inkling gqa G6 S16", .symbol = "tf_ik_gqa_6_16", .image = "inkling2_gqa", .threads = 128, .min_sm = 120 }, .usage = .{ none, none, none, .{ .regs = 95, .static = 46408, .bound = 128 }, .{ .regs = 95, .static = 46408, .bound = 128 } } },
    .{ .launch = .{ .name = "inkling gqa tile address G1 S4", .symbol = "tf_ik_gqa_address_1_4", .image = "inkling2_gqa", .threads = 128, .min_sm = 120 }, .usage = .{ none, none, none, .{ .regs = 86, .static = 46408, .bound = 128 }, .{ .regs = 86, .static = 46408, .bound = 128 } } },
    .{ .launch = .{ .name = "inkling gqa tile address G3 S8", .symbol = "tf_ik_gqa_address_3_8", .image = "inkling2_gqa", .threads = 128, .min_sm = 120 }, .usage = .{ none, none, none, .{ .regs = 84, .static = 46408, .bound = 128 }, .{ .regs = 84, .static = 46408, .bound = 128 } } },
};
