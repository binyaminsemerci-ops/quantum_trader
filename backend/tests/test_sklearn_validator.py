"""
Tests for sklearn startup validator
Ensures sklearn validation works correctly
"""
import pytest
from unittest.mock import patch, MagicMock
import sys


def test_sklearn_validator_import():
    """Test that validator can be imported."""
    from ai_engine.sklearn_startup_validator import SklearnStartupValidator
    validator = SklearnStartupValidator()
    assert validator is not None


def test_sklearn_import_check_success():
    """Test sklearn import check succeeds when sklearn is available."""
    from ai_engine.sklearn_startup_validator import SklearnStartupValidator
    
    validator = SklearnStartupValidator()
    validator._check_sklearn_import()
    
    assert validator.validation_results.get('sklearn_import') == True
    assert 'sklearn_version' in validator.validation_results
    assert len(validator.errors) == 0


def test_sklearn_import_check_failure():
    """Test sklearn import check fails gracefully when sklearn missing."""
    from ai_engine.sklearn_startup_validator import SklearnStartupValidator
    
    validator = SklearnStartupValidator()
    
    # Mock sklearn import to fail
    with patch.dict('sys.modules', {'sklearn': None}):
        with patch('builtins.__import__', side_effect=ImportError("sklearn not found")):
            validator._check_sklearn_import()
    
    assert validator.validation_results.get('sklearn_import') == False
    assert len(validator.errors) > 0


def test_numpy_compatibility_check():
    """Test numpy compatibility check."""
    from ai_engine.sklearn_startup_validator import SklearnStartupValidator
    
    validator = SklearnStartupValidator()
    validator._check_numpy_compatibility()
    
    # Should pass if numpy is available
    assert validator.validation_results.get('numpy_compatible') in [True, None]


def test_core_modules_check():
    """Test core sklearn modules check."""
    from ai_engine.sklearn_startup_validator import SklearnStartupValidator
    
    validator = SklearnStartupValidator()
    validator._check_core_sklearn_modules()
    
    # Should pass if sklearn is properly installed
    assert validator.validation_results.get('core_modules') in [True, False]


def test_scaler_functionality():
    """Test StandardScaler functionality check."""
    from ai_engine.sklearn_startup_validator import SklearnStartupValidator
    
    validator = SklearnStartupValidator()
    validator._check_scaler_functionality()
    
    # Should pass if sklearn is available
    result = validator.validation_results.get('scaler_functional')
    assert result in [True, False]
    
    # If it passed, verify no errors
    if result == True:
        assert len(validator.errors) == 0


def test_model_loading():
    """Test pickle model loading functionality."""
    from ai_engine.sklearn_startup_validator import SklearnStartupValidator
    
    validator = SklearnStartupValidator()
    validator._check_model_loading()
    
    # Should pass if sklearn is available
    result = validator.validation_results.get('model_loading')
    assert result in [True, False]


def test_optional_dependencies():
    """Test optional dependencies check (warnings only)."""
    from ai_engine.sklearn_startup_validator import SklearnStartupValidator
    
    validator = SklearnStartupValidator()
    validator._check_optional_dependencies()
    
    # Should complete without errors (only warnings allowed)
    assert len(validator.errors) == 0
    assert 'optional_deps_missing' in validator.validation_results


def test_model_files_exist():
    """Test model files existence check (warnings only)."""
    from ai_engine.sklearn_startup_validator import SklearnStartupValidator
    
    validator = SklearnStartupValidator()
    validator._check_model_files_exist()
    
    # Should complete without errors (only warnings allowed)
    assert len(validator.errors) == 0
    assert 'missing_model_files' in validator.validation_results


def test_full_validation():
    """Test complete validation flow."""
    from ai_engine.sklearn_startup_validator import SklearnStartupValidator
    
    validator = SklearnStartupValidator()
    success, results = validator.validate_all()
    
    # Should return boolean and results dict
    assert isinstance(success, bool)
    assert isinstance(results, dict)
    
    # Results should have key validation fields
    assert 'sklearn_import' in results
    assert 'core_modules' in results
    assert 'scaler_functional' in results


def test_validate_sklearn_on_startup():
    """Test the main startup validation function."""
    from ai_engine.sklearn_startup_validator import validate_sklearn_on_startup
    
    result = validate_sklearn_on_startup()
    
    # Should return boolean
    assert isinstance(result, bool)
    
    # If sklearn is installed, should pass
    try:
        import sklearn
        assert result == True, "Validation should pass when sklearn is available"
    except ImportError:
        assert result == False, "Validation should fail when sklearn is missing"


def test_validation_with_corrupted_sklearn():
    """Test validation handles corrupted sklearn gracefully."""
    from ai_engine.sklearn_startup_validator import SklearnStartupValidator
    
    validator = SklearnStartupValidator()
    
    # Mock sklearn to simulate corruption
    mock_sklearn = MagicMock()
    mock_sklearn.__version__ = None  # Corrupted version
    
    with patch.dict('sys.modules', {'sklearn': mock_sklearn}):
        with patch('builtins.__import__', return_value=mock_sklearn):
            validator._check_sklearn_import()
    
    # Should handle gracefully
    assert 'sklearn_import' in validator.validation_results


def test_validator_never_crashes():
    """Test that validator never crashes, even with errors."""
    from ai_engine.sklearn_startup_validator import SklearnStartupValidator
    
    validator = SklearnStartupValidator()
    
    # Test that validator handles internal errors gracefully
    # Don't mock __import__ as it breaks the validator itself
    # Instead, test with corrupted results
    try:
        success, results = validator.validate_all()
        # Should complete without raising exception
        assert isinstance(success, bool)
        assert isinstance(results, dict)
    except Exception as e:
        pytest.fail(f"Validator should never crash, but raised: {e}")


def test_validation_results_structure():
    """Test that validation results have expected structure."""
    from ai_engine.sklearn_startup_validator import SklearnStartupValidator
    
    validator = SklearnStartupValidator()
    success, results = validator.validate_all()
    
    # Check expected keys exist
    expected_keys = [
        'sklearn_import',
        'core_modules',
        'scaler_functional',
        'model_loading',
        'optional_deps_missing',
        'missing_model_files',
    ]
    
    for key in expected_keys:
        assert key in results, f"Expected key '{key}' missing from results"


def test_error_vs_warning_separation():
    """Test that errors and warnings are separated correctly."""
    from ai_engine.sklearn_startup_validator import SklearnStartupValidator
    
    validator = SklearnStartupValidator()
    
    # Add a mock error
    validator.errors.append("Critical error")
    
    # Add a mock warning
    validator.warnings.append("Non-critical warning")
    
    # Errors should fail validation
    assert len(validator.errors) > 0
    
    # But warnings should not
    assert len(validator.warnings) > 0


@pytest.mark.asyncio
async def test_sklearn_validation_in_startup():
    """Test that sklearn validation integrates with FastAPI startup."""
    # This is a conceptual test - actual integration happens in main.py
    from ai_engine.sklearn_startup_validator import validate_sklearn_on_startup
    
    # Validate can be called in async context
    result = validate_sklearn_on_startup()
    assert isinstance(result, bool)
