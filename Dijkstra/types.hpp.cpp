// types.hpp
#pragma once

#include <cstdint>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Note: Unrecoverable error macro for CUDA API calls. 
// Halts execution to prevent undefined behavior in async streams.
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA Error at %s:%d - %s\n",              \
                         __FILE__, __LINE__, cudaGetErrorString(err));      \
            std::abort();                                                   \
        }                                                                   \
    } while (0)

namespace hybrid_dijkstra {

// Core configuration constants
constexpr uint32_t GPU_BATCH_THRESHOLD = 4096;
constexpr float INF_DIST = std::numeric_limits<float>::infinity();

// Note: Represents a static sparse graph in Compressed Sparse Row format.
// Utilizes Unified Memory for seamless CPU-GPU page migration.
struct CSRGraph {
    uint32_t num_nodes;
    uint32_t num_edges;

    // Topology arrays
    uint32_t* row_ptr; // Size: num_nodes + 1
    uint32_t* col_idx; // Size: num_edges
    float* weights;    // Size: num_edges

    // Distance array shared via Unified Memory
    float* distances;  // Size: num_nodes

    CSRGraph(uint32_t nodes, uint32_t edges) 
        : num_nodes(nodes), num_edges(edges), 
          row_ptr(nullptr), col_idx(nullptr), 
          weights(nullptr), distances(nullptr) {
        
        CUDA_CHECK(cudaMallocManaged(&row_ptr, (num_nodes + 1) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocManaged(&col_idx, num_edges * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocManaged(&weights, num_edges * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&distances, num_nodes * sizeof(float)));
    }

    ~CSRGraph() {
        if (row_ptr) cudaFree(row_ptr);
        if (col_idx) cudaFree(col_idx);
        if (weights) cudaFree(weights);
        if (distances) cudaFree(distances);
    }

    // Disable copy semantics to prevent double-free of managed memory
    CSRGraph(const CSRGraph&) = delete;
    CSRGraph& operator=(const CSRGraph&) = delete;

    // Enable move semantics for efficient ownership transfer
    CSRGraph(CSRGraph&& other) noexcept 
        : num_nodes(other.num_nodes), num_edges(other.num_edges),
          row_ptr(other.row_ptr), col_idx(other.col_idx), 
          weights(other.weights), distances(other.distances) {
        
        other.row_ptr = nullptr;
        other.col_idx = nullptr;
        other.weights = nullptr;
        other.distances = nullptr;
    }
    
    CSRGraph& operator=(CSRGraph&& other) noexcept = delete;
};

} // namespace hybrid_dijkstra