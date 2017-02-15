# VectorizedSTL
###An AVX Vectorized STL

VectorizedSTL is a header only library that provides containers and algorithms that make use of modern AVX2. The library is currently built only for AVX2 and only supports Intel Haswell CPUs or newer.

## VectorTable
An unordered_map replacement optimized for small tables (less than 1,000 key/value pairs). Even at 1,000,000 elements VectorTable still provides a 50% or more speed up compared to unordered_map.
