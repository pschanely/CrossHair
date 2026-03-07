# CrossHair Issue #115: Trace Callback Performance Optimization

## Summary

This implementation addresses [Issue #115](https://github.com/pschanely/CrossHair/issues/115) by optimizing the trace callback performance in CrossHair. The issue noted that CrossHair spends a lot of execution time in pure-Python `sys.settrace` handlers, and moving more logic to C could significantly speed up CrossHair.

## Changes Made

### 1. C Extension Optimizations (`crosshair/_tracers.c`)

#### Opcode-to-Handler Caching
- **Problem**: For every traced opcode, the C code would iterate through all handler tables and call into Python, even when there's only one handler.
- **Solution**: Added a simple cache that remembers the last opcode and its handler for single-handler scenarios.
- **Implementation**: 
  - Added `last_opcode` and `last_handler` fields to the CTracer struct
  - When there's only one handler table, check the cache first before iterating
  - Cache is invalidated when modules are pushed/popped

#### Streamlined Post-Op Callback Processing
- **Problem**: Post-op callback processing had nested conditionals and multiple exit points.
- **Solution**: Refactored into a separate inline function `process_postop_callbacks()` with early returns for the common no-callback case.

#### Cache Statistics
- Added `cache_hits` and `cache_misses` counters for debugging
- Added `get_cache_stats()` method to expose statistics to Python
- Helps verify that optimizations are working effectively

#### Handler Table Iteration Order
- **Optimization**: Changed from forward iteration to reverse iteration
- **Rationale**: More recently added handlers are likely to be more active; reverse iteration improves cache locality

### 2. Header File Updates (`crosshair/_tracers.h`)

Added new fields to the `CTracer` struct:
```c
int last_opcode;           /* Cache: last opcode processed */
PyObject* last_handler;    /* Cache: handler for last_opcode */
long cache_hits;           /* Statistics: cache hits */
long cache_misses;         /* Statistics: cache misses */
```

### 3. Python-Level Optimizations (`crosshair/tracers.py`)

#### Fast-Path for Primitive Types
- **Problem**: The `trace_op` method performs expensive attribute lookups for every target, even for primitive types that will never need tracing.
- **Solution**: Added an early exit check for common primitive types:
```python
fast_path_types = (type(None), int, float, str, bool, list, dict, tuple, set)
if target_type in fast_path_types:
    return None
```

#### Optimized Exception Handling
- **Problem**: Stack reads were wrapped in try/except blocks that always ran.
- **Solution**: Combined operations that can fail into single try/except blocks to reduce overhead.

#### Inline Attribute Lookups
- **Problem**: Multiple `__getattribute__` calls for the same object.
- **Solution**: Inlined the lookups to reduce function call overhead.

### 4. Testing

Created two test files:

#### `tracers_performance_test.py`
- Unit tests for the new fast-path behavior
- Tests for cache statistics functionality
- Ensures optimizations don't break existing functionality

#### `tracers_performance_benchmark.py`
- Benchmarks for various tracing scenarios
- Measures cache hit rates
- Demonstrates the impact of optimizations

### 5. Documentation

#### Updated `doc/source/changelog.rst`
- Added entry for the next version documenting the performance improvements

#### Added inline documentation
- C code includes detailed comments explaining the optimizations
- Python docstrings explain the fast-path behavior

## Performance Impact

### Expected Improvements

1. **Reduced Python-to-C Transitions**: The opcode-to-handler cache avoids Python function calls for the common single-handler case.

2. **Fewer Attribute Lookups**: The primitive type fast-path in Python eliminates expensive `__getattribute__` calls for common types.

3. **Better Cache Locality**: Reverse iteration of handler tables improves CPU cache utilization.

4. **Streamlined Processing**: The refactored post-op callback processing has fewer branches and better branch prediction.

### Benchmarking

Run the benchmark script to measure improvements:
```bash
python crosshair/tracers_performance_benchmark.py
```

The benchmark tests:
- Simple function calls
- Nested function calls
- Method calls
- Loops with function calls

## Technical Details

### Cache Invalidation Strategy

The opcode-to-handler cache is invalidated when:
1. A new module is pushed (`CTracer_push_module`)
2. A module is popped (`CTracer_pop_module`)

This ensures that the cache never contains stale data while minimizing invalidation overhead.

### Thread Safety Considerations

The cache is per-CTracer instance, and CTracer instances are not shared between threads. The existing thread safety model of CrossHair is maintained.

### Backward Compatibility

All changes are backward compatible:
- The `get_cache_stats()` method is new and doesn't affect existing code
- The fast-path optimizations are transparent to callers
- All existing tests should continue to pass

## Future Work

Potential additional optimizations:

1. **More Sophisticated Caching**: Could implement an LRU cache for multiple handlers
2. **Fast-Path for Common Patterns**: Identify and optimize common opcode sequences
3. **Batch Processing**: Group multiple opcodes before calling into Python
4. **Profile-Guided Optimization**: Use cache stats to identify the most common paths and optimize them further

## Files Changed

1. `crosshair/_tracers.c` - Core C extension optimizations
2. `crosshair/_tracers.h` - Header file with new struct fields
3. `crosshair/tracers.py` - Python-level fast-path optimizations
4. `doc/source/changelog.rst` - Documentation update
5. `crosshair/tracers_performance_test.py` - New test file
6. `crosshair/tracers_performance_benchmark.py` - New benchmark file

## References

- Issue: https://github.com/pschanely/CrossHair/issues/115
- Discussion: https://github.com/pschanely/CrossHair/discussions/105#discussioncomment-662381
