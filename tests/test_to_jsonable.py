"""Test JSON-safe serialization."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, date
import numpy as np

from dc_solver.reporting.run_info import to_jsonable


def test_to_jsonable_primitives():
    """Test that primitives pass through unchanged."""
    assert to_jsonable(None) is None
    assert to_jsonable(True) is True
    assert to_jsonable(False) is False
    assert to_jsonable(42) == 42
    assert to_jsonable(3.14) == 3.14
    assert to_jsonable("hello") == "hello"


def test_to_jsonable_numpy_array():
    """Test numpy array conversion to list."""
    arr = np.array([1.0, 2.0, 3.0])
    result = to_jsonable(arr)
    assert result == [1.0, 2.0, 3.0]
    assert isinstance(result, list)


def test_to_jsonable_numpy_scalar():
    """Test numpy scalar conversion."""
    val = np.float64(3.14159)
    result = to_jsonable(val)
    assert abs(result - 3.14159) < 1e-6
    assert isinstance(result, float)

    val_int = np.int64(42)
    result_int = to_jsonable(val_int)
    assert result_int == 42
    assert isinstance(result_int, int)


def test_to_jsonable_path():
    """Test Path conversion to string."""
    p = Path("/home/user/test.txt")
    result = to_jsonable(p)
    assert result == "/home/user/test.txt"
    assert isinstance(result, str)


def test_to_jsonable_datetime():
    """Test datetime conversion to ISO string."""
    dt = datetime(2025, 1, 15, 10, 30, 45)
    result = to_jsonable(dt)
    assert result == "2025-01-15T10:30:45"
    assert isinstance(result, str)

    d = date(2025, 1, 15)
    result_d = to_jsonable(d)
    assert result_d == "2025-01-15"
    assert isinstance(result_d, str)


def test_to_jsonable_collections():
    """Test set, tuple, list conversions."""
    s = {1, 2, 3}
    result_s = to_jsonable(s)
    assert sorted(result_s) == [1, 2, 3]
    assert isinstance(result_s, list)

    t = (1, 2, 3)
    result_t = to_jsonable(t)
    assert result_t == [1, 2, 3]
    assert isinstance(result_t, list)

    lst = [1, 2, 3]
    result_lst = to_jsonable(lst)
    assert result_lst == [1, 2, 3]
    assert isinstance(result_lst, list)


def test_to_jsonable_dict():
    """Test dict conversion (recursive, with str keys)."""
    d = {
        "int_key": 42,
        "np_key": np.float64(3.14),
        "arr_key": np.array([1, 2, 3]),
        123: "numeric key",  # non-string key
    }
    result = to_jsonable(d)
    assert result["int_key"] == 42
    assert abs(result["np_key"] - 3.14) < 1e-6
    assert result["arr_key"] == [1, 2, 3]
    assert result["123"] == "numeric key"


def test_to_jsonable_nested():
    """Test nested structures."""
    data = {
        "name": "test",
        "values": np.array([1.0, 2.0, 3.0]),
        "meta": {
            "dt": datetime(2025, 1, 1, 0, 0, 0),
            "path": Path("/tmp/test"),
            "nested_arr": [np.int64(10), np.int64(20)],
        },
    }
    result = to_jsonable(data)

    # Should be JSON-serializable now
    json_str = json.dumps(result)
    assert json_str is not None

    # Check values
    assert result["name"] == "test"
    assert result["values"] == [1.0, 2.0, 3.0]
    assert result["meta"]["dt"] == "2025-01-01T00:00:00"
    assert result["meta"]["path"] == "/tmp/test"
    assert result["meta"]["nested_arr"] == [10, 20]


def test_to_jsonable_full_roundtrip():
    """Test full roundtrip to JSON and back."""
    data = {
        "job": "test_job",
        "ndof": np.int64(300),
        "flops_est": np.float64(1e9),
        "amps_g_used": np.array([0.1, 0.2, 0.3]),
        "output_dir": Path("/tmp/output"),
        "timestamp": datetime(2025, 1, 1, 12, 0, 0),
        "meta": {
            "tags": {"important", "test"},
            "counts": (1, 2, 3),
        },
    }

    result = to_jsonable(data)

    # Should serialize to JSON without errors
    json_str = json.dumps(result, indent=2)
    assert json_str is not None

    # Should deserialize back
    reloaded = json.loads(json_str)
    assert reloaded["job"] == "test_job"
    assert reloaded["ndof"] == 300
    assert abs(reloaded["flops_est"] - 1e9) < 1e-3
    assert reloaded["amps_g_used"] == [0.1, 0.2, 0.3]
    assert reloaded["output_dir"] == "/tmp/output"
    assert reloaded["timestamp"] == "2025-01-01T12:00:00"


if __name__ == "__main__":
    test_to_jsonable_primitives()
    test_to_jsonable_numpy_array()
    test_to_jsonable_numpy_scalar()
    test_to_jsonable_path()
    test_to_jsonable_datetime()
    test_to_jsonable_collections()
    test_to_jsonable_dict()
    test_to_jsonable_nested()
    test_to_jsonable_full_roundtrip()
    print("All tests passed!")
