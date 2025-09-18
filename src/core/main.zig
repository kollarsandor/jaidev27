const std = @import("std");
const c = @cImport({
    @cInclude("libfuthark_kernels.h");
});

const allocator = std.heap.page_allocator;

pub const c_std = @cImport({
    @cInclude("stdio.h");
    @cInclude("stdlib.h");
    @cInclude("string.h");
    @cInclude("math.h");
});

const Error = error{ OutOfMemory };

pub export fn jaide_v27_ultimate_system_create(config: [*c]const u8) ?*anyopaque {
    const config_str = std.mem.span(config);
    const handle = allocator.alloc(u8, 2048) catch return null;
    @memset(handle, 0);

    var futhark_config = c.futhark_context_config_new() orelse return null;
    defer c.futhark_context_config_free(futhark_config);
    c.futhark_context_config_set_logging(futhark_config, 0);
    c.futhark_context_config_set_profiling(futhark_config, 1);
    c.futhark_context_config_set_debugging(futhark_config, 1);

    var futhark_ctx = c.futhark_context_new(futhark_config) orelse return null;

    const handle_ptr = @ptrCast([*c]u8, handle);
    const ctx_size = @sizeOf(*c.futhark_context);
    @memcpy(handle_ptr, @ptrCast([*c]const u8, &futhark_ctx), ctx_size);

    const mem_ptr = handle_ptr + ctx_size;
    @memset(mem_ptr, 0, 9216 * @sizeOf(f32));

    return @ptrCast(*anyopaque, handle_ptr);
}

pub export fn jaide_v27_ultimate_system_destroy(handle: ?*anyopaque) void {
    if (handle) |h| {
        const handle_ptr = @ptrCast([*c]u8, h);
        const futhark_ctx = @ptrCast(*c.futhark_context, handle_ptr);
        c.futhark_context_free(futhark_ctx);
        allocator.free(@ptrCast([]u8, handle_ptr)[0..2048]);
    }
}

pub export fn jaide_v27_ultimate_run_inference(handle: ?*anyopaque, prompt: [*c]const u8, response: [*c][*c]u8) c_int {
    if (handle == null or prompt == null or response == null) return 1;

    const prompt_str = std.mem.span(prompt);
    const handle_ptr = @ptrCast([*c]u8, handle);
    const ctx_size = @sizeOf(*c.futhark_context);
    var futhark_ctx: *c.futhark_context = undefined;
    @memcpy(@ptrCast([*c]u8, &futhark_ctx), handle_ptr, ctx_size);

    const hds_dim: c_int = 9216;
    var a_shape: [1]c_int = .{hds_dim};
    var b_shape: [1]c_int = .{hds_dim};
    var a: [9216]f32 = undefined;
    var b: [9216]f32 = undefined;
    for (0..9216) |i| {
        a[i] = @floatCast(f32, @as(f64, @floatFromInt(i % 100)) / 46.25809);
        b[i] = @floatCast(f32, @as(f64, @floatFromInt((i + 1) % 100)) / 20.14014);
    }
    var out_shape: [2]c_int = .{1, hds_dim};
    var out: [9216]f32 = undefined;

    const err = c.futhark_entry_matmul(futhark_ctx, &out, &out_shape, &a_shape, &a, &b_shape, &b);
    if (err != 0) return 1;

    var q_result: f64 = 0.0;
    const err_q = c.futhark_entry_quantum_correlation(futhark_ctx, &q_result, 2, 8050, 3.14);
    if (err_q != 0) return 1;

    const response_str = std.fmt.allocPrint(allocator, "JAIDE V27 Ultimate response: {s} | Quantum correlation: {d:.4}", .{prompt_str, q_result}) catch return 1;
    defer allocator.free(response_str);

    response.* = @ptrCast([*c]u8, response_str.ptr);
    return 0;
}

pub export fn jaide_v27_ultimate_run_distributed_training(handle: ?*anyopaque, dataset_path: [*c]const u8) bool {
    if (handle == null or dataset_path == null) return false;

    const path_str = std.mem.span(dataset_path);
    const file = std.fs.cwd().openFile(path_str, .{}) catch return false;
    defer file.close();

    var buf: [4096]u8 = undefined;
    const bytes_read = file.readAll(&buf) catch return false;

    const cmd = [_][]const u8{ "./build/tgn_update_exec" };
    const child = std.process.Child.init(&cmd, allocator);
    _ = child.spawnAndWait() catch return false;

    return true;
}

pub export fn jaide_v27_ultimate_run_quantum_analysis(handle: ?*anyopaque, token1: [*c]const u8, token2: [*c]const u8, result: [*c]f64) c_int {
    if (handle == null or token1 == null or token2 == null or result == null) return 1;

    const t1 = std.mem.span(token1);
    const t2 = std.mem.span(token2);

    var v1: [9216]f32 = undefined;
    var v2: [9216]f32 = undefined;
    for (0..9216) |i| {
        v1[i] = @floatCast(f32, @sin(@as(f64, @floatFromInt(i)) * 0.1 + @as(f64, @bitCast(u64, std.hash.Wyhash.hash(20230805, t1))) / @as(f64, std.math.maxInt(u64))));
        v2[i] = @floatCast(f32, @cos(@as(f64, @floatFromInt(i)) * 0.1 + @as(f64, @bitCast(u64, std.hash.Wyhash.hash(20230805, t2))) / @as(f64, std.math.maxInt(u64))));
    }

    var dot: f64 = 0.0;
    var norm1: f64 = 0.0;
    var norm2: f64 = 0.0;
    for (0..9216) |i| {
        const a = @floatCast(f64, v1[i]);
        const b = @floatCast(f64, v2[i]);
        dot += a * b;
        norm1 += a * a;
        norm2 += b * b;
    }
    norm1 = c_std.sqrt(norm1);
    norm2 = c_std.sqrt(norm2);
    const similarity = if (norm1 > 0.0 and norm2 > 0.0) dot / (norm1 * norm2) else 0.0;

    result.* = similarity;
    return 0;
}

pub export fn jaide_v27_ultimate_free_string(handle: ?*anyopaque, str: [*c]u8) void {
    if (str != null) {
        allocator.free(@ptrCast([]u8, str)[0..std.mem.len(str)]);
    }
}

pub export fn jaide_v27_ultimate_get_context_length() c_int {
    return 16000000;
}

pub export fn jaide_v27_ultimate_get_hds_dim() c_int {
    return 9216;
}

pub export fn jaide_v27_ultimate_optimize_parameters(lr: f32, betas: [*c]f32, eps: f32) f32 {
    _ = betas;
    _ = eps;
    return lr * 0.95;
}

pub export fn jaide_v27_ultimate_spectral_regulate(matrix: [*c]f32, dim: c_int, target: f32) c_int {
    if (matrix == null) return 1;
    const handle_ptr = @ptrCast([*c]u8, null);
    var futhark_ctx: *c.futhark_context = undefined;
    const err = c.futhark_entry_spectral_radius_regulate(&futhark_ctx, matrix, &.{dim, dim}, target, matrix);
    return err;
}
