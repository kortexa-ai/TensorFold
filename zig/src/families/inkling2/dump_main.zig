//! Native development oracle: a bounded prefix of the real checkpoint, with raw residual and logit dumps.
const std = @import("std");
const cuda = @import("cuda");
const ik = @import("inkling2");

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 5) {
        std.debug.print("usage: tf-inkling-cuda MODEL IDS.json LAYERS OUTPUT_DIR\n", .{});
        return 2;
    }
    const count = try std.fmt.parseInt(usize, args[3], 10);
    if (count < 1 or count > ik.config.layers) return error.InklingGeometry;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(1 << 22));
    const doc = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    const ids = try a.alloc(u32, doc.value.array.items.len);
    for (ids, doc.value.array.items) |*id, v| id.* = @intCast(v.integer);
    if (ids.len == 0 or ids.len > ik.state.prompt_rows) return error.InklingProbeRows;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var name: [192]u8 = undefined;
    const gpu_name = try ctx.name(&name);
    if (std.mem.indexOf(u8, gpu_name, "RTX PRO 6000") == null) return error.InklingProbeGpu;
    const old_limit = try cuda.setAllocationLimit(16 << 30);
    defer _ = cuda.setAllocationLimit(old_limit) catch {};
    var stream = try cuda.Stream.init(&driver, false);
    defer stream.deinit();
    var kernels = try ik.kernels.Kernels.load(&driver);
    defer kernels.deinit();
    var lt = try cuda.cublaslt.Library.open();
    defer lt.close();
    var w = try ik.weights.load(gpa, io, &driver, args[1], try ik.config.Share.init(0, 1), count);
    defer w.deinit();
    var s = try ik.state.State.init(gpa, &driver, &w, @max(ids.len, 32));
    defer s.deinit();
    var f: ik.forward.Forward = .{ .gpa = gpa, .d = &driver, .stream = stream, .w = &w, .s = &s, .k = &kernels, .lt = &lt };
    defer f.deinit();
    try f.run(ids, true);
    const token = try f.pick();
    try std.Io.Dir.cwd().createDirPath(io, args[4]);
    var out = try std.Io.Dir.cwd().openDir(io, args[4], .{});
    defer out.close(io);
    const arrays = .{
        .{ "partial.bf16", s.partial, ids.len * ik.config.hidden * 2 },
        .{ "attention.bf16", s.attention, ids.len * ik.config.hidden * 2 },
        .{ "mlp.bf16", s.mlp, ids.len * ik.config.hidden * 2 },
        .{ "router.f32", s.router, ids.len * ik.config.experts * 4 },
        .{ "logits.f32", s.logits, w.head.n * 4 },
    };
    inline for (arrays) |entry| {
        const bytes = try a.alloc(u8, entry[2]);
        try cuda.memory.download(&driver, entry[1], bytes);
        try out.writeFile(io, .{ .sub_path = entry[0], .data = bytes });
    }
    std.debug.print("inkling native dump: {d} layers, {d} rows, token {d}, peak {d} bytes\n", .{ count, ids.len, token, cuda.usage(false).peak });
    return 0;
}
