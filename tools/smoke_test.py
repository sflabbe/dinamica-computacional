#!/usr/bin/env python
"""Smoke test: verify JobRunner infrastructure and JSON-safe serialization.

This is a minimal sanity check that:
1. JobRunner creates all expected files
2. to_jsonable() handles numpy arrays and other non-serializable types
3. File tracking works
4. JSON roundtrip succeeds

Run:
    python tools/smoke_test.py
"""

from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path

# Allow running as standalone script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from datetime import datetime

from dc_solver.job import JobRunner
from dc_solver.reporting.run_info import to_jsonable


def test_to_jsonable():
    """Test JSON-safe serialization."""
    print("Testing to_jsonable()...")

    data = {
        "name": "test",
        "ndof": np.int64(300),
        "flops": np.float64(1.23e9),
        "array": np.array([1.0, 2.0, 3.0]),
        "path": Path("/tmp/test"),
        "dt": datetime(2025, 1, 1, 12, 0, 0),
        "meta": {
            "nested_array": np.array([10, 20]),
            "tuple": (1, 2, 3),
            "set": {4, 5, 6},
        },
    }

    # Convert to JSON-safe
    safe_data = to_jsonable(data)

    # Should be serializable now
    json_str = json.dumps(safe_data, indent=2)
    assert json_str is not None

    # Reload and check
    reloaded = json.loads(json_str)
    assert reloaded["name"] == "test"
    assert reloaded["ndof"] == 300
    assert abs(reloaded["flops"] - 1.23e9) < 1e-3
    assert reloaded["array"] == [1.0, 2.0, 3.0]
    assert reloaded["path"] == "/tmp/test"
    assert reloaded["dt"] == "2025-01-01T12:00:00"
    assert reloaded["meta"]["nested_array"] == [10, 20]
    assert sorted(reloaded["meta"]["set"]) == [4, 5, 6]

    print("  ✓ to_jsonable() works correctly")


def test_job_runner():
    """Test JobRunner infrastructure."""
    print("\nTesting JobRunner...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_job"

        meta = {
            "test_array": np.array([1.0, 2.0]),
            "test_path": Path("/tmp/foo"),
        }

        with JobRunner(
            job_name="smoke_test",
            output_dir=output_dir,
            meta=meta,
            print_header=False,  # Suppress console output for test
        ) as job:
            job.set_analysis_params(ndof=10, n_steps=100, integrator="explicit")
            job.log("Test message")

            # Create a dummy output file
            test_file = output_dir / "test_output.txt"
            test_file.write_text("test content")

            job.mark_success()

        # Verify expected files exist
        expected_files = [
            "smoke_test.msg",
            "smoke_test.sta",
            "smoke_test.dat",
            "journal.log",
            "smoke_test_runinfo.json",
            "smoke_test_runinfo.txt",
            "test_output.txt",
        ]

        for filename in expected_files:
            filepath = output_dir / filename
            assert filepath.exists(), f"Missing file: {filename}"

        print(f"  ✓ All expected files created ({len(expected_files)} files)")

        # Verify runinfo.json is valid and JSON-safe
        runinfo_path = output_dir / "smoke_test_runinfo.json"
        with runinfo_path.open() as f:
            runinfo = json.load(f)

        assert runinfo["job"] == "smoke_test"
        assert runinfo["success"] is True
        assert "wall_s" in runinfo
        assert "flops_est" in runinfo
        assert "new_files" in runinfo
        assert "test_output.txt" in runinfo["new_files"]

        print("  ✓ runinfo.json is valid and complete")

        # Check that .dat contains JOB TOTALS
        dat_path = output_dir / "smoke_test.dat"
        dat_content = dat_path.read_text()
        assert "JOB TOTALS" in dat_content, "Missing JOB TOTALS section in .dat"
        assert "flops_est" in dat_content, "Missing flops_est in .dat"

        print("  ✓ .dat file contains JOB TOTALS")

        # Check journal.log
        journal_path = output_dir / "journal.log"
        journal_content = journal_path.read_text()
        assert "Test message" in journal_content
        assert "JOB START" in journal_content
        assert "JOB END" in journal_content

        print("  ✓ journal.log contains expected entries")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("SMOKE TEST: JobRunner Infrastructure")
    print("=" * 60)

    try:
        test_to_jsonable()
        test_job_runner()

        print("\n" + "=" * 60)
        print("✓ ALL SMOKE TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ SMOKE TEST FAILED")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
