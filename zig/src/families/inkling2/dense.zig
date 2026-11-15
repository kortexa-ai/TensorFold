//! Candidate one-row GB10 projections with cuBLASLt's summation order and original output precision.
const std = @import("std");
const cuda = @import("cuda");

pub fn serves(sm: u32, version: usize, rows: usize, n: usize, k: usize, out: cuda.cublaslt.DataType) bool {
    if (sm != 121 or version != 130101 or rows != 1) return false;
    return switch (out) {
        .bf16 => n == 3072 and k == 4608,
        .f32 => (n == 4608 and k == 3072) or (n == 100512 and k == 4608),
        else => false,
    };
}

pub const Dense = struct {
    bf16_mod: cuda.Module,
    f32_mod: cuda.Module,
    bf16: cuda.Function,
    f32: cuda.Function,

    pub fn load(d: *const cuda.Driver) !?Dense {
        if (try d.currentCapability() != 121) return null;
        var bf16_mod = try cuda.Module.load(d, cuda.kernels.drama3_staged_gemv);
        errdefer bf16_mod.unload();
        var f32_mod = try cuda.Module.load(d, cuda.kernels.inkling2_dense);
        errdefer f32_mod.unload();
        return .{ .bf16_mod = bf16_mod, .f32_mod = f32_mod, .bf16 = try bf16_mod.function("drama3_staged_gemv"), .f32 = try f32_mod.function("tf_ik_dense_gemv_f32") };
    }

    pub fn deinit(self: *Dense) void {
        self.f32_mod.unload();
        self.bf16_mod.unload();
    }

    pub fn run(self: *const Dense, s: cuda.Stream, x: u64, w: u64, out: u64, n: usize, k: usize, kind: cuda.cublaslt.DataType) !void {
        var args: cuda.Args = .{};
        inline for (.{ x, w, out, @as(i32, @intCast(k)) }) |v| args.add(v);
        try cuda.launch.launch(if (kind == .bf16) self.bf16 else self.f32, .{ .grid = .{ .x = @intCast(n / 8) }, .block = .{ .x = 32 } }, s, &args);
    }
};

test "Inkling dense selection excludes prompt rows, small projections and unqualified runtimes" {
    try std.testing.expect(serves(121, 130101, 1, 3072, 4608, .bf16));
    try std.testing.expect(serves(121, 130101, 1, 4608, 3072, .f32));
    try std.testing.expect(serves(121, 130101, 1, 100512, 4608, .f32));
    try std.testing.expect(!serves(120, 130101, 1, 3072, 4608, .bf16));
    try std.testing.expect(!serves(121, 130100, 1, 3072, 4608, .bf16));
    try std.testing.expect(!serves(121, 130101, 2, 3072, 4608, .bf16));
    try std.testing.expect(!serves(121, 130101, 1, 512, 4608, .bf16));
    try std.testing.expect(!serves(121, 130101, 1, 384, 4608, .f32));
}
