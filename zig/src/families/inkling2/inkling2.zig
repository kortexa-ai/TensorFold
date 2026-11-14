//! Native CUDA bring-up for Inkling 2 Small, using the existing lane/server and EXL3 owners.
pub const config = @import("config.zig");
pub const weights = @import("weights.zig");
pub const state = @import("state.zig");
pub const kernels = @import("kernels.zig");
pub const forward = @import("forward.zig");
pub const native = @import("native.zig");

test {
    _ = config;
    _ = kernels;
    _ = native;
    _ = @import("link.zig");
}
