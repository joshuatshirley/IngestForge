# Quick Start: Concurrency Tests

## Run All Tests
```bash
pytest tests/integration/test_concurrency.py -v
```

## Run Specific Test Class
```bash
# State access tests (fastest)
pytest tests/integration/test_concurrency.py::TestConcurrentStateAccess -v

# Pipeline processing tests
pytest tests/integration/test_concurrency.py::TestConcurrentPipelineProcessing -v

# Timeout handling tests
pytest tests/integration/test_concurrency.py::TestLockTimeoutHandling -v

# Consistency tests
pytest tests/integration/test_concurrency.py::TestStateConsistency -v

# Cleanup tests
pytest tests/integration/test_concurrency.py::TestResourceCleanup -v
```

## Run Single Test
```bash
pytest tests/integration/test_concurrency.py::TestConcurrentStateAccess::test_concurrent_state_writes -v
```

## Skip Slow Tests
```bash
pytest tests/integration -m "not slow"
```

## Run Only Integration Tests
```bash
pytest -m integration
```

## Verbose Output
```bash
pytest tests/integration/test_concurrency.py -vv -s
```

## With Coverage
```bash
pytest tests/integration/test_concurrency.py --cov=ingestforge.core.state --cov-report=term-missing
```

## Parallel Execution (use xdist)
```bash
pytest tests/integration/test_concurrency.py -n auto
```
Note: Not recommended for concurrency tests (defeats the purpose)

## Expected Runtime
- Single test: 5-30 seconds
- Full suite: 2-5 minutes

## Prerequisites
```bash
pip install filelock  # Required for lock timeout tests
```

## Test Output
```
PASSED tests/integration/test_concurrency.py::TestConcurrentStateAccess::test_concurrent_state_writes
PASSED tests/integration/test_concurrency.py::TestConcurrentStateAccess::test_concurrent_read_while_writing
PASSED tests/integration/test_concurrency.py::TestConcurrentStateAccess::test_state_lock_cleanup
PASSED tests/integration/test_concurrency.py::TestConcurrentPipelineProcessing::test_concurrent_process_file_calls
PASSED tests/integration/test_concurrency.py::TestConcurrentPipelineProcessing::test_parallel_process_pending
PASSED tests/integration/test_concurrency.py::TestConcurrentPipelineProcessing::test_concurrent_same_file_processing
PASSED tests/integration/test_concurrency.py::TestLockTimeoutHandling::test_lock_timeout_fallback
PASSED tests/integration/test_concurrency.py::TestLockTimeoutHandling::test_graceful_degradation_under_contention
PASSED tests/integration/test_concurrency.py::TestStateConsistency::test_no_duplicate_document_ids
PASSED tests/integration/test_concurrency.py::TestStateConsistency::test_state_statistics_accuracy
PASSED tests/integration/test_concurrency.py::TestStateConsistency::test_atomic_state_updates
PASSED tests/integration/test_concurrency.py::TestResourceCleanup::test_lock_file_cleanup_after_processing
PASSED tests/integration/test_concurrency.py::TestResourceCleanup::test_cleanup_on_worker_crash
```

## Troubleshooting

### Tests are slow
- Normal! Concurrency tests involve real multiprocessing
- Reduce worker counts in test if needed
- Skip with: `pytest -m "not slow"`

### "filelock not installed"
```bash
pip install filelock
```

### Tests timeout
- Increase timeout in pytest.ini
- Check system CPU load
- Reduce num_workers in test code

### Windows file locking issues
- Tests account for Windows behavior
- May see brief delays on cleanup
- This is expected

## More Information
- Full docs: `CONCURRENCY_TESTS_README.md`
- Implementation: `TEST_IMPLEMENTATION_SUMMARY.md`
- Source: `ingestforge/core/state.py`
