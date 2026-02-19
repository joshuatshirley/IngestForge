# Concurrency Integration Tests

This document describes the concurrent pipeline access tests implemented in `test_concurrency.py`.

## Overview

The concurrency test suite validates that IngestForge's StateManager correctly handles concurrent access patterns using file locking. Tests use Python's `multiprocessing` module to launch real concurrent workers that access shared state files.

## Test Organization

### TestConcurrentStateAccess (3 tests)
Tests for StateManager locking under concurrent access:
- `test_concurrent_state_writes`: 5 workers × 10 updates = 50 concurrent writes
- `test_concurrent_read_while_writing`: Mixed read/write operations (5 readers + 1 writer)
- `test_state_lock_cleanup`: Verify lock files are properly cleaned up

### TestConcurrentPipelineProcessing (3 tests)
Tests for concurrent document processing through Pipeline:
- `test_concurrent_process_file_calls`: 5 workers processing 5 different files concurrently
- `test_parallel_process_pending`: Built-in `process_pending(parallel=True)` with 3 workers
- `test_concurrent_same_file_processing`: Edge case - 3 workers processing identical file

### TestLockTimeoutHandling (2 tests)
Tests for lock timeout scenarios and recovery:
- `test_lock_timeout_fallback`: Verify fallback to non-locked write on timeout
- `test_graceful_degradation_under_contention`: 10 workers with high contention (5ms delay)

### TestStateConsistency (3 tests)
Tests for state consistency under concurrent access:
- `test_no_duplicate_document_ids`: Verify unique document IDs under concurrency
- `test_state_statistics_accuracy`: Verify accurate statistics after concurrent updates
- `test_atomic_state_updates`: Verify state file is never corrupted (no partial writes)

### TestResourceCleanup (2 tests)
Tests for proper cleanup of locks and temporary files:
- `test_lock_file_cleanup_after_processing`: Verify lock cleanup after successful processing
- `test_cleanup_on_worker_crash`: Verify cleanup when worker processes crash

## Running the Tests

### Run all concurrency tests
```bash
pytest tests/integration/test_concurrency.py -v
```

### Run specific test class
```bash
pytest tests/integration/test_concurrency.py::TestConcurrentStateAccess -v
```

### Run with markers
```bash
# Run all slow integration tests
pytest -m "integration and slow"

# Skip slow tests
pytest -m "not slow"
```

### Run with verbose output
```bash
pytest tests/integration/test_concurrency.py -vv -s
```

## Test Fixtures

### Core Fixtures
- **temp_project_dir**: Isolated temporary project directory with full structure
- **sample_files**: 5 distinct text files (Python, ML, DB, Web, Cloud topics)
- **concurrent_pipeline**: Pipeline configured for concurrent testing
- **state_file**: Path to pipeline_state.json

### Configuration
All tests use:
- JSONL storage backend (simple, file-based)
- Disabled enrichment (faster processing)
- Small chunks (100 words, 20 overlap)
- 10s lock timeout (JPL #1: Fixed bounds)

## 
### JPL #1: Fixed Timeout Bounds
- All lock operations use explicit 5s or 10s timeouts
- No unbounded waits

### JPL #2: Proper Resource Cleanup
- All fixtures use `tmp_path` for automatic cleanup
- Lock files verified to be removed after operations
- Test for cleanup on worker crash

### JPL #3: No Silent Failures
- All errors explicitly checked and asserted
- Worker errors collected and verified
- Fallback behavior tested (not just ignored)

### JPL #5: Assertions for Invariants
- State consistency checked after every operation
- Document counts verified
- JSON validity confirmed

### JPL #6: Data Structure Bounds
- Max workers = min(cpu_count, 10)
- Fixed number of updates per worker
- Bounded document counts

### JPL #9: Preprocessor Limits
- Max 10 concurrent workers (stress test)
- Typical tests use 3-5 workers

## Expected Results

### Success Metrics
- **test_concurrent_state_writes**: 100% success rate (50/50 updates)
- **test_graceful_degradation_under_contention**: >80% success rate under stress
- **test_atomic_state_updates**: 0 corrupted state snapshots
- **All tests**: No lock timeout exceptions leak to user

### Timing
- Individual tests: 5-30 seconds
- Full suite: 2-5 minutes (depends on CPU count)

## Troubleshooting

### Tests timeout
- Increase lock_timeout in worker functions
- Reduce num_workers or updates_per_worker
- Check for deadlocks in StateManager

### "filelock not installed" skip
```bash
pip install filelock
```

### High failure rate in stress test
- Acceptable if >80% success rate
- May indicate lock contention issues
- Check FileLock implementation on platform

### Windows-specific issues
- File locks may behave differently than POSIX
- Tests account for platform differences
- Check atomic file rename behavior

## Test Data

### Sample Files (5 × ~150 words each)
1. **doc1.txt**: Python Programming
2. **doc2.txt**: Machine Learning  
3. **doc3.txt**: Database Systems
4. **doc4.txt**: Web Development
5. **doc5.txt**: Cloud Computing

Each file generates ~3-5 chunks at 100-word target size.

## Coverage

### What is tested
- Concurrent state writes
- Concurrent reads during writes
- Lock acquisition and release
- Timeout handling
- Fallback behavior
- State file atomicity
- Resource cleanup
- Document ID uniqueness
- Statistics accuracy
- Worker crash recovery

### What is NOT tested
- Network-based storage backends (Chroma, Qdrant)
- Distributed locking across machines
- File system corruption
- OS-level file lock failures
- Memory-based storage (no file locks needed)

## Future Enhancements

1. **Distributed Testing**: Test locking across networked file systems (NFS, SMB)
2. **Performance Profiling**: Measure lock contention impact on throughput
3. **Chaos Engineering**: Randomly kill workers mid-operation
4. **Platform Matrix**: Test on Linux, macOS, Windows concurrently
5. **Scale Testing**: Test with 100+ concurrent workers

## References

- StateManager implementation: `ingestforge/core/state.py`
- FileLock documentation: https://py-filelock.readthedocs.io/
- NASA JPL Power of Ten: [COMMANDMENTS_CHECKLIST.md](../../COMMANDMENTS_CHECKLIST.md)
