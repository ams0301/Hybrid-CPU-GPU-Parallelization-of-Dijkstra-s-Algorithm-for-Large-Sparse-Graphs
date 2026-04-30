// scheduler.cpp
#include "types.hpp"
#include <queue>
#include <vector>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <cub/cub.cuh>

namespace hybrid_dijkstra {

// Note: External linkage to Phase 1 CUDA kernel.
extern __global__ void relax_edges_kernel(
    const uint32_t* active_nodes, uint32_t num_active,
    const uint32_t* row_ptr, const uint32_t* col_idx,
    const float* weights, float* distances, uint8_t* out_mask);

struct NodeDist {
    float dist;
    uint32_t id;
    bool operator>(const NodeDist& other) const { return dist > other.dist; }
};

// Note: Selection functor for Phase 2 CUB compaction.
struct MaskFilter {
    const uint8_t* mask;
    __host__ __device__ __forceinline__ bool operator()(const uint32_t& node_idx) const {
        return mask[node_idx] == 1;
    }
};

void execute_hybrid_dijkstra(CSRGraph& graph, uint32_t source_node) {
    // 1. Threshold Computation: \theta = median(degree) + 2\sigma
    std::vector<uint32_t> degrees(graph.num_nodes);
    double sum_deg = 0.0;
    for (uint32_t i = 0; i < graph.num_nodes; ++i) {
        degrees[i] = graph.row_ptr[i + 1] - graph.row_ptr[i];
        sum_deg += degrees[i];
        graph.distances[i] = INF_DIST;
    }
    
    std::vector<uint32_t> sorted_deg = degrees;
    std::nth_element(sorted_deg.begin(), sorted_deg.begin() + graph.num_nodes / 2, sorted_deg.end());
    uint32_t median_deg = sorted_deg[graph.num_nodes / 2];
    
    double mean_deg = sum_deg / graph.num_nodes;
    double variance = 0.0;
    for (uint32_t d : degrees) {
        variance += (d - mean_deg) * (d - mean_deg);
    }
    variance /= graph.num_nodes;
    float sigma = std::sqrt(variance);
    
    uint32_t threshold = median_deg + static_cast<uint32_t>(2.0f * sigma);

    // 2. Resource Allocation: Double-buffering & Async Streams
    cudaStream_t streams[2];
    uint32_t* d_active_nodes[2];
    uint8_t* d_out_mask[2];
    uint32_t* d_compacted_out[2];
    uint32_t* d_num_selected[2];
    void* d_temp_storage[2] = {nullptr, nullptr};
    size_t temp_storage_bytes[2] = {0, 0};

    std::vector<uint32_t> h_batch[2];
    uint32_t* h_compacted_out[2];
    uint32_t h_num_selected[2] = {0, 0};
    bool gpu_busy[2] = {false, false};

    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMalloc(&d_active_nodes[i], GPU_BATCH_THRESHOLD * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_out_mask[i], graph.num_nodes * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_compacted_out[i], graph.num_nodes * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_num_selected[i], sizeof(uint32_t)));
        
        // Note: Pinned memory ensures optimal PCIe throughput for async copybacks.
        CUDA_CHECK(cudaMallocHost(&h_compacted_out[i], graph.num_nodes * sizeof(uint32_t)));
        h_batch[i].reserve(GPU_BATCH_THRESHOLD);
        
        cub::CountingInputIterator<uint32_t> iter(0);
        cub::DeviceSelect::If(nullptr, temp_storage_bytes[i], iter, d_compacted_out[i], d_num_selected[i], graph.num_nodes, MaskFilter{d_out_mask[i]}, streams[i]);
        CUDA_CHECK(cudaMalloc(&d_temp_storage[i], temp_storage_bytes[i]));
    }

    // 3. Initialization
    std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>> pq;
    graph.distances[source_node] = 0.0f;
    pq.push({0.0f, source_node});
    
    int active_buf = 0;

    // Note: Lambda encapsulation for stream dispatch and buffer swap.
    auto dispatch_gpu_batch = [&]() {
        if (h_batch[active_buf].empty()) return;
        
        // Sync current stream to retrieve previous results before overwriting.
        CUDA_CHECK(cudaStreamSynchronize(streams[active_buf]));
        
        if (gpu_busy[active_buf]) {
            for (uint32_t i = 0; i < h_num_selected[active_buf]; ++i) {
                uint32_t v = h_compacted_out[active_buf][i];
                pq.push({graph.distances[v], v});
            }
        }
        
        uint32_t batch_size = h_batch[active_buf].size();
        CUDA_CHECK(cudaMemcpyAsync(d_active_nodes[active_buf], h_batch[active_buf].data(), batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[active_buf]));
        CUDA_CHECK(cudaMemsetAsync(d_out_mask[active_buf], 0, graph.num_nodes * sizeof(uint8_t), streams[active_buf]));
        
        // Phase 1: Expand
        // Warp-centric load balancing assumes 1 warp (32 threads) per node
        uint32_t blocks = (batch_size * 32 + 1023) / 1024; 
        relax_edges_kernel<<<blocks, 1024, 0, streams[active_buf]>>>(
            d_active_nodes[active_buf], batch_size, 
            graph.row_ptr, graph.col_idx, graph.weights, 
            graph.distances, d_out_mask[active_buf]);
        
        // Phase 2: Compact using CUB::DeviceSelect::If
        cub::CountingInputIterator<uint32_t> iter(0);
        cub::DeviceSelect::If(
            d_temp_storage[active_buf], temp_storage_bytes[active_buf], 
            iter, d_compacted_out[active_buf], d_num_selected[active_buf], 
            graph.num_nodes, MaskFilter{d_out_mask[active_buf]}, streams[active_buf]);
        
        // Async Readback
        CUDA_CHECK(cudaMemcpyAsync(h_num_selected + active_buf, d_num_selected[active_buf], sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[active_buf]));
        CUDA_CHECK(cudaMemcpyAsync(h_compacted_out[active_buf], d_compacted_out[active_buf], graph.num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[active_buf]));
        
        gpu_busy[active_buf] = true;
        h_batch[active_buf].clear();
        active_buf = 1 - active_buf; // Toggle buffer
    };

    // 4. Hybrid Execution Loop
    bool processing = true;
    while (processing) {
        if (!pq.empty()) {
            auto [dist_u, u] = pq.top();
            pq.pop();
            
            // Lazy deletion check. 
            // Note: std::atomic_ref required due to concurrent GPU unified memory writes.
            std::atomic_ref<float> a_dist(graph.distances[u]);
            if (dist_u > a_dist.load(std::memory_order_relaxed)) continue;

            if (degrees[u] >= threshold) {
                h_batch[active_buf].push_back(u);
                if (h_batch[active_buf].size() >= GPU_BATCH_THRESHOLD) {
                    dispatch_gpu_batch();
                }
            } else {
                // CPU sequential relaxation
                for (uint32_t i = graph.row_ptr[u]; i < graph.row_ptr[u + 1]; ++i) {
                    uint32_t v = graph.col_idx[i];
                    float new_dist = dist_u + graph.weights[i];
                    
                    std::atomic_ref<float> dist_v(graph.distances[v]);
                    float current_dist = dist_v.load(std::memory_order_relaxed);
                    while (new_dist < current_dist) {
                        if (dist_v.compare_exchange_weak(current_dist, new_dist, std::memory_order_relaxed)) {
                            pq.push({new_dist, v});
                            break;
                        }
                    }
                }
            }
        } else {
            // Priority queue drained. Flush any pending high-degree nodes.
            if (!h_batch[active_buf].empty()) {
                dispatch_gpu_batch();
            }
            
            // Poll streams to ingest completed GPU work
            bool work_found = false;
            for (int b = 0; b < 2; ++b) {
                if (gpu_busy[b]) {
                    CUDA_CHECK(cudaStreamSynchronize(streams[b]));
                    for (uint32_t i = 0; i < h_num_selected[b]; ++i) {
                        uint32_t v = h_compacted_out[b][i];
                        pq.push({graph.distances[v], v});
                    }
                    gpu_busy[b] = false;
                    work_found = true;
                }
            }
            if (!work_found && pq.empty()) {
                processing = false; // Convergence
            }
        }
    }

    // 5. Cleanup
    for (int i = 0; i < 2; ++i) {
        cudaFree(d_active_nodes[i]);
        cudaFree(d_out_mask[i]);
        cudaFree(d_compacted_out[i]);
        cudaFree(d_num_selected[i]);
        cudaFree(d_temp_storage[i]);
        cudaFreeHost(h_compacted_out[i]);
        cudaStreamDestroy(streams[i]);
    }
}

} // namespace hybrid_dijkstra