// kernels.cu
#include "types.hpp"

namespace hybrid_dijkstra {

// Note: Standard atomicMin does not support 32-bit floats natively on all architectures.
// Implementation utilizes atomicCAS. Float-to-int bitwise reinterpretation preserves 
// monotonicity for non-negative distance values.
__device__ __forceinline__ void atomicMinFloat(float* addr, float val, uint8_t* out_mask, uint32_t dest_node) {
    unsigned int* addr_as_ui = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *addr_as_ui;
    unsigned int assumed;
    bool updated = false;

    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) break;
        old = atomicCAS(addr_as_ui, assumed, __float_as_int(val));
        if (assumed == old) updated = true;
    } while (assumed != old);

    // Note: Flag the destination node in the frontier bitmask if distance was successfully lowered.
    if (updated) {
        out_mask[dest_node] = 1;
    }
}

// Note: Phase 1 Expand Kernel.
// Warp-centric load balancing: 1 warp per high-degree node. 
// Batches of size >= 4096 ensure full occupancy.
__global__ void relax_edges_kernel(
    const uint32_t* active_nodes,
    uint32_t num_active,
    const uint32_t* row_ptr,
    const uint32_t* col_idx,
    const float* weights,
    float* distances,
    uint8_t* out_mask) 
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warp_id = tid / 32;
    uint32_t lane_id = tid % 32;

    if (warp_id >= num_active) return;

    // Load active source node
    uint32_t u = active_nodes[warp_id];
    float dist_u = distances[u];

    uint32_t edge_start = row_ptr[u];
    uint32_t edge_end = row_ptr[u + 1];

    for (uint32_t i = edge_start + lane_id; i < edge_end; i += 32) {
        uint32_t v = col_idx[i];
        float new_dist = dist_u + weights[i];

        if (new_dist < distances[v]) {
            uint32_t active_mask = __activemask();
            uint32_t unproc_mask = active_mask;

            // Note: Loop over unique destinations within the warp to enable uniform masks
            // for __reduce_min_sync. Mitigates atomic write contention at power-law hubs.
            while (unproc_mask != 0) {
                int leader = __ffs(unproc_mask) - 1;
                uint32_t target_v = __shfl_sync(unproc_mask, v, leader);
                uint32_t match_mask = __match_any_sync(unproc_mask, target_v);

                if (match_mask & (1 << lane_id)) {
                    // Reduce minimum distance among threads targeting the same destination
                    int min_dist_int = __reduce_min_sync(match_mask, __float_as_int(new_dist));

                    if (lane_id == __ffs(match_mask) - 1) {
                        float min_dist = __int_as_float(min_dist_int);
                        atomicMinFloat(&distances[target_v], min_dist, out_mask, target_v);
                    }
                }
                // Strip processed threads from the active queue
                unproc_mask ^= match_mask;
            }
        }
    }
}

} // namespace hybrid_dijkstra