import pytest
import os
from tdda.referencetest import ReferenceTestCase, tag

# Check if we're in write mode
write_mode = os.environ.get('TDDA_WRITE_ALL', '0') == '1'

# Create a pytest fixture that provides a ReferenceTestCase instance
@pytest.fixture
def ref_case():
    case = ReferenceTestCase()
    if write_mode:
        case.write_all = True
    return case

# Register the tdda assertion methods as pytest fixtures
@pytest.fixture
def assertDataFrameCorrect(ref_case):
    return ref_case.assertDataFrameCorrect

@pytest.fixture
def assertStringCorrect(ref_case):
    return ref_case.assertStringCorrect

@pytest.fixture
def assertFileCorrect(ref_case):
    return ref_case.assertFileCorrect

@pytest.fixture
def assertCSVFileCorrect(ref_case):
    return ref_case.assertCSVFileCorrect

# Helper function to generate reference data
def pytest_configure(config):
    """
    Register a marker for tdda tests
    """
    config.addinivalue_line("markers", "tdda: mark test as using tdda reference testing")
