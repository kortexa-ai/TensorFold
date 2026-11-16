//! The native lane host owns scheduling; these small NCCL records only mirror its forwards on the worker.
const std = @import("std");
const cuda = @import("cuda");
const forward = @import("forward.zig");

pub const Kind = enum(u32) { begin = 1, rows, end, verify, keep, draft };
const Header = extern struct { kind: u32, rows: u32 = 0, final: u32 = 0, reserved: u32 = 0 };

pub const Link = struct {
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    header: cuda.DeviceBuffer,
    ids: cuda.DeviceBuffer,

    pub fn init(d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, rows: usize) !Link {
        var h = try cuda.DeviceBuffer.alloc(d, @sizeOf(Header));
        errdefer h.free();
        return .{ .d = d, .lib = lib, .comm = comm, .stream = stream, .header = h, .ids = try cuda.DeviceBuffer.alloc(d, rows * 4) };
    }
    pub fn deinit(l: *Link) void {
        l.header.free();
        l.ids.free();
    }

    fn broadcast(l: *Link, ptr: u64, bytes: usize) !void {
        try l.lib.check(l.lib.api.ncclBroadcast(ptr, ptr, bytes, .u8, 0, l.comm, l.stream.handle), "inkling command");
    }

    pub fn send(l: *Link, kind: Kind, ids: []const u32, final: bool) !void {
        if (ids.len * 4 > l.ids.len) return error.InklingCommandRows;
        const header: Header = .{ .kind = @backingInt(kind), .rows = @intCast(ids.len), .final = @intFromBool(final) };
        try cuda.copy_source.enqueue(l.d, l.header.ptr, std.mem.asBytes(&header), l.stream);
        try l.broadcast(l.header.ptr, @sizeOf(Header));
        if (ids.len > 0) {
            try cuda.copy_source.enqueue(l.d, l.ids.ptr, std.mem.sliceAsBytes(ids), l.stream);
            try l.broadcast(l.ids.ptr, ids.len * 4);
        }
    }

    pub fn follow(l: *Link, f: *forward.Forward, gpa: std.mem.Allocator) !void {
        const ids = try gpa.alloc(u32, l.ids.len / 4);
        defer gpa.free(ids);
        while (true) {
            try l.broadcast(l.header.ptr, @sizeOf(Header));
            try l.stream.synchronize();
            if (f.exchange) |e| if (e.group) |g| try g.check();
            var header: Header = undefined;
            try l.header.download(0, std.mem.asBytes(&header));
            const kind = std.enums.fromInt(Kind, header.kind) orelse return error.InklingCommand;
            switch (kind) {
                .begin => {
                    if (header.rows != 0) return error.InklingCommand;
                    try f.clear();
                },
                .rows, .verify, .keep, .draft => {
                    if (header.rows == 0 or header.rows > ids.len or header.final > 1) return error.InklingCommand;
                    const values = ids[0..header.rows];
                    try l.broadcast(l.ids.ptr, values.len * 4);
                    try l.stream.synchronize();
                    try l.ids.download(0, std.mem.sliceAsBytes(values));
                    switch (kind) {
                        .rows => try f.run(values, header.final != 0),
                        .verify => {
                            var sampled: [8]u32 = undefined;
                            try f.verify(values, &sampled);
                        },
                        .keep => {
                            if (values.len != 1) return error.InklingCommand;
                            try f.keep(values[0]);
                        },
                        .draft => {
                            if (values.len != 2) return error.InklingCommand;
                            var proposals: [7]u32 = undefined;
                            try (f.drafter orelse return error.InklingNoDrafts).propose(f, values[0], values[1], &proposals);
                        },
                        else => unreachable,
                    }
                },
                .end => return,
            }
        }
    }
};

test "the paired forward header is fixed-size and contains no process-local pointers" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Header));
}
