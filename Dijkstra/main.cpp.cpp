// main.cpp
#include "types.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

namespace hybrid_dijkstra {
// Note: External linkage to the scheduler control loop.
extern void execute_hybrid_dijkstra(CSRGraph& graph, uint32_t source_node);
}

struct Edge {
    uint32_t u, v;
    float w;
};

void print_usage() {
    std::fprintf(stderr, "Usage: ./hybrid_dijkstra --graph <path> --source <id> [--sigma_thresh <float>]\n");
    std::fprintf(stderr, "Options:\n");
    std::fprintf(stderr, "  --graph         Path to edge list file (Format: u v weight per line)\n");
    std::fprintf(stderr, "  --source        Source node ID (0-indexed)\n");
    std::fprintf(stderr, "  --sigma_thresh  Threshold tuning parameter for GPU offload (Default: 2.0)\n");
}

int main(int argc, char** argv) {
    std::string graph_path = "";
    uint32_t source_node = 0;
    float sigma_thresh = 2.0f; // Note: Parsed for threshold tuning logic expansion.

    // 1. Command-line argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--graph" && i + 1 < argc) {
            graph_path = argv[++i];
        } else if (arg == "--source" && i + 1 < argc) {
            source_node = std::stoul(argv[++i]);
        } else if (arg == "--sigma_thresh" && i + 1 < argc) {
            sigma_thresh = std::stof(argv[++i]);
        }
    }

    if (graph_path.empty()) {
        print_usage();
        return EXIT_FAILURE;
    }

    // 2. Dataset loading
    std::ifstream infile(graph_path);
    if (!infile.is_open()) {
        std::fprintf(stderr, "IO Error: Cannot open graph file %s\n", graph_path.c_str());
        return EXIT_FAILURE;
    }

    std::vector<Edge> edges;
    uint32_t max_node = 0;
    uint32_t u, v;
    float w;

    while (infile >> u >> v >> w) {
        edges.push_back({u, v, w});
        if (u > max_node) max_node = u;
        if (v > max_node) max_node = v;
    }
    
    uint32_t num_nodes = max_node + 1;
    uint32_t num_edges = edges.size();

    if (source_node >= num_nodes) {
        std::fprintf(stderr, "Domain Error: Source node %u exceeds graph bounds (%u nodes)\n", source_node, num_nodes);
        return EXIT_FAILURE;
    }

    // 3. CSR Conversion
    // Note: Allocates Unified Memory arrays internally via constructor.
    hybrid_dijkstra::CSRGraph graph(num_nodes, num_edges);

    std::vector<uint32_t> degree(num_nodes, 0);
    for (const auto& e : edges) {
        degree[e.u]++;
    }

    graph.row_ptr[0] = 0;
    for (uint32_t i = 0; i < num_nodes; ++i) {
        graph.row_ptr[i + 1] = graph.row_ptr[i] + degree[i];
    }

    std::vector<uint32_t> current_offset(num_nodes, 0);
    for (const auto& e : edges) {
        uint32_t offset = graph.row_ptr[e.u] + current_offset[e.u]++;
        graph.col_idx[offset] = e.v;
        graph.weights[offset] = e.w;
    }

    std::fprintf(stdout, "[INFO] Graph initialized: V = %u, E = %u. Target Source: %u.\n", num_nodes, num_edges, source_node);

    // 4. Execution & Profiling
    auto start_time = std::chrono::high_resolution_clock::now();
    
    hybrid_dijkstra::execute_hybrid_dijkstra(graph, source_node);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    // 5. Verification output
    std::fprintf(stdout, "[INFO] Hybrid Dijkstra execution completed in %.3f ms.\n", duration.count());
    
    // Output a few sample distances for sanity checks
    std::fprintf(stdout, "[DATA] Distance to node 0: %.4f\n", graph.distances[0]);
    if (num_nodes > 1) {
        std::fprintf(stdout, "[DATA] Distance to node %u: %.4f\n", num_nodes - 1, graph.distances[num_nodes - 1]);
    }

    return EXIT_SUCCESS;
}