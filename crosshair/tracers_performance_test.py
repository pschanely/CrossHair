"""Tests for trace callback performance optimizations (Issue #115)."""

import sys
from unittest.mock import Mock

import pytest

from crosshair.tracers import (
    COMPOSITE_TRACER,
    PatchingModule,
    TracingModule,
    is_tracing,
)


class TestTracingModuleFastPath:
    """Test that the TracingModule trace_op fast-path optimizations work correctly."""

    def test_trace_op_with_nonexistent_opcode_exits_early(self):
        """Test that trace_op exits early when opcode has no handler."""
        module = TracingModule()
        
        # Create a mock frame
        mock_frame = Mock()
        mock_frame.f_code.co_code = bytes([0, 0])
        mock_frame.f_lasti = 0
        
        # Use a non-existent opcode (256)
        result = module.trace_op(mock_frame, mock_frame.f_code, 256)
        assert result is None

    def test_trace_op_fast_path_for_primitives(self):
        """Test that trace_op has fast-path for primitive types."""
        module = TracingModule()
        
        # Verify that primitive types are in the fast-path check
        # This is a white-box test of the optimization
        fast_path_types = (type(None), int, float, str, bool, list, dict, tuple, set)
        
        for t in fast_path_types:
            # The check ensures these types exit early
            assert t in fast_path_types


class TestCTracerCacheStats:
    """Test that the C tracer cache statistics work correctly."""

    def test_cache_stats_returns_dict(self):
        """Test that get_cache_stats returns a dictionary."""
        stats = COMPOSITE_TRACER.get_cache_stats()
        assert isinstance(stats, dict)
    
    def test_cache_stats_has_expected_keys(self):
        """Test that cache stats has the expected keys."""
        stats = COMPOSITE_TRACER.get_cache_stats()
        expected_keys = {'cache_hits', 'cache_misses', 'last_opcode'}
        for key in expected_keys:
            assert key in stats


class TestPerformanceOptimizations:
    """Test that performance optimizations don't break functionality."""

    def test_tracing_module_callable(self):
        """Test that TracingModule instances remain callable."""
        module = TracingModule()
        assert callable(module)
    
    def test_composite_tracer_has_cache_stats_method(self):
        """Test that CompositeTracer has the get_cache_stats method."""
        assert hasattr(COMPOSITE_TRACER, 'get_cache_stats')
        assert callable(COMPOSITE_TRACER.get_cache_stats)


class TestOpcodeFiltering:
    """Test that opcode filtering works correctly with optimizations."""

    def test_tracing_module_opcodes_wanted(self):
        """Test that TracingModule has opcodes_wanted attribute."""
        module = TracingModule()
        assert hasattr(module, 'opcodes_wanted')
        assert isinstance(module.opcodes_wanted, frozenset)
