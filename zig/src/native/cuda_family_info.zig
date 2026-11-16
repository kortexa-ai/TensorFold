//! Merge a family's declared serving capabilities into the native CUDA host metadata.
const api = @import("engine_api");

/// The Info `openWith` serves: the host's fields, then the family's reasoning and greedy-only declarations.
pub fn apply(comptime F: type, host: api.Info) api.Info {
    var info = host;
    info.reasoning = reasoning(F);
    if (@hasDecl(F, "greedy_only")) info.greedy_only = F.greedy_only;
    return info;
}

/// The family's Info.reasoning: false when it declares `pub const reasoning = false` (no reasoning block).
pub fn reasoning(comptime F: type) bool {
    return if (@hasDecl(F, "reasoning")) F.reasoning else true;
}
