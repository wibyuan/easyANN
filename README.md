# ANN - Approximate Nearest Neighbor Search

A comparative study of multiple Approximate Nearest Neighbor (ANN) search algorithms, implementing and benchmarking various graph-based approaches including HNSW, FlatNav, and ONNG.

## Overview

This project implements 30+ variants of ANN algorithms to find the K nearest neighbors in high-dimensional vector spaces. It includes a complete evaluation framework for measuring recall and query performance (QPS).

## Algorithms Implemented

| Algorithm | Description | Location |
|-----------|-------------|----------|
| Brute Force | Exact search baseline (O(n)) | `method/bruteforce/` |
| HNSW | Hierarchical Navigable Small World (multi-layer) | `method/hnsw1/` |
| HNSW + Adaptive Search | HNSW with NGT's adaptive search strategy | `method/hnsw2/` |
| FlatNav | Single-layer HNSW graph | `method/flatnav*/` |
| ONNG | FlatNav + ONNG edge pruning | `method/onng*/` |

### Implementation Notes

**FlatNav**: A single-layer variant of HNSW where `random_level()` always returns 0. This simplifies the graph structure while maintaining the core navigation properties.

**ONNG**: Built on top of FlatNav (single-layer graph), with the addition of an **edge pruning (shortcut reduction) strategy** adapted from [NGT](https://github.com/yahoojapan/NGT)'s `adjustPathsEffectively`.

**HNSW2**: Uses NGT's **adaptive search strategy** with a `gamma` parameter to control early termination based on the ratio between current candidate distance and the worst distance in the result set.

**Important**: The edge pruning primarily affects graph storage size and cache efficiency. It does **not significantly reduce the number of distance computations** during search. The `onng1_test2_sq16_2` directory contains a pure single-layer graph **without** edge pruning for comparison.

### Optimizations

- **SIMD**: Vectorized distance computation (`flatnav1_SIMD/`)
- **Multi-threading**: Parallel search (`flatnav1_threads*/`)
- **Graph Reordering**: G-order, RCM optimization (`flatnav1_threads1_gorder/`, `flatnav1_threads1_rcm/`)
- **Quantization**: SQ8, SQ16 compression (`onng1_test2_sq*/`)

## Datasets

| Dataset | Base Vectors | Dimensions | Queries |
|---------|--------------|------------|---------|
| SIFT | 1,000,000 | 128 | 10,000 |
| GLOVE | 1,183,514 | 100 | 9,000 |
| DEBUG | 1,000 | 16 | 100 |

## Performance Results

Example results on SIFT dataset (1M vectors, 128-dim):

| Method | Search Latency (ms) | Recall@10 |
|--------|---------------------|-----------|
| Brute Force | 128.45 | 99.94% |
| HNSW | 1.27 | 99.57% |
| ONNG | 0.33 | 99.33% |

## Project Structure

```
.
├── method/           # Algorithm implementations
│   ├── bruteforce/   # Baseline brute force search
│   ├── hnsw*/        # HNSW variants
│   ├── flatnav*/     # FlatNav variants
│   ├── onng*/        # ONNG variants
│   └── flatnavlib/   # Third-party libraries (Cereal, RapidJSON)
├── checker/          # Evaluation tools
│   ├── evaluate.cpp  # Main evaluation program
│   ├── run_eval.bat  # Windows evaluation script
│   └── run_eval.sh   # Linux evaluation script
└── data/             # Datasets (binary format)
    ├── sift/
    ├── glove/
    └── debug/
```

## Quick Start

### Prerequisites

- C++17 compatible compiler (g++ recommended)
- Python 3.8+ (for data processing scripts)

### Build and Run

```bash
# Navigate to checker directory
cd checker

# Run evaluation for a specific method
# Windows:
.\run_eval.bat hnsw1

# Linux:
./run_eval.sh hnsw1
```

### Data Format

- **Binary format** (`.bin`): Optimized for fast loading
- **Text format** (`.txt`): Human-readable, one vector per line

## Implementing Your Own Algorithm

1. Create a new directory under `method/`
2. Implement `MySolution.h` and `MySolution.cpp` with the required interface:

```cpp
class MySolution {
public:
    void build(const std::vector<std::vector<float>>& base);
    std::vector<int> search(const std::vector<float>& query, int k);
};
```

3. Run evaluation: `.\run_eval.bat your_method_name`

## Dependencies

This project uses the following open-source libraries:

- [Cereal](https://github.com/USCiLab/cereal) - Serialization (BSD-3-Clause)
- [RapidJSON](https://github.com/Tencent/rapidjson) - JSON parsing (MIT)
- [hnswlib](https://github.com/nmslib/hnswlib) - HNSW implementation reference (Apache-2.0)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SIFT dataset: [ANN Benchmarks](http://ann-benchmarks.com/)
- GloVe dataset: [Stanford NLP](https://nlp.stanford.edu/projects/glove/)
