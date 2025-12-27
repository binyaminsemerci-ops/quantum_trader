"""
File I/O Bulletproofing Module

Handles all file operations (model loading, config reading, log writing) with comprehensive error handling.
Never crashes the application due to file I/O errors.

Features:
- Safe model loading with fallbacks
- Config reading with defaults
- Disk space monitoring
- Path validation
- Atomic file writes
- Backup/restore functionality
"""

import os
import json
import pickle
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Callable
from datetime import datetime
from dataclasses import dataclass
import tempfile

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class FileIOStats:
    """Track file I/O operations"""
    total_reads: int = 0
    successful_reads: int = 0
    failed_reads: int = 0
    
    total_writes: int = 0
    successful_writes: int = 0
    failed_writes: int = 0
    
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    
    def success_rate(self, operation: str = 'read') -> float:
        """Calculate success rate for reads or writes"""
        if operation == 'read':
            if self.total_reads == 0:
                return 1.0
            return self.successful_reads / self.total_reads
        else:  # write
            if self.total_writes == 0:
                return 1.0
            return self.successful_writes / self.total_writes


class BulletproofFileIO:
    """
    Bulletproof file I/O handler
    
    Never crashes due to:
    - Missing files
    - Permission errors
    - Disk full
    - Corrupted files
    - Invalid JSON/pickle
    - Path traversal
    """
    
    def __init__(
        self,
        base_path: Optional[Path] = None,
        backup_enabled: bool = True,
        min_disk_space_mb: int = 100
    ):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.backup_enabled = backup_enabled
        self.min_disk_space_mb = min_disk_space_mb
        self.stats = FileIOStats()
        
        logger.info(f"FileIO bulletproofing initialized at {self.base_path}")
    
    def _validate_path(self, path: Path) -> bool:
        """
        Validate path is safe (no path traversal attacks)
        
        Returns:
            True if path is safe, False otherwise
        """
        try:
            # Resolve to absolute path
            abs_path = path.resolve()
            abs_base = self.base_path.resolve()
            
            # Check if path is within base_path (prevent path traversal)
            if not str(abs_path).startswith(str(abs_base)):
                logger.warning(f"Path traversal detected: {path} outside {self.base_path}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            return False
    
    def _check_disk_space(self, path: Path) -> bool:
        """
        Check if enough disk space is available
        
        Returns:
            True if enough space, False otherwise
        """
        try:
            stat = shutil.disk_usage(path.parent if path.is_file() else path)
            free_mb = stat.free / (1024 * 1024)
            
            if free_mb < self.min_disk_space_mb:
                logger.error(
                    f"Low disk space: {free_mb:.2f}MB free "
                    f"(minimum: {self.min_disk_space_mb}MB)"
                )
                return False
            
            return True
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            return False
    
    def _create_backup(self, path: Path) -> Optional[Path]:
        """
        Create backup of existing file
        
        Returns:
            Path to backup file, or None if backup failed
        """
        if not self.backup_enabled or not path.exists():
            return None
        
        try:
            backup_path = path.with_suffix(path.suffix + '.backup')
            shutil.copy2(path, backup_path)
            logger.debug(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.warning(f"Backup creation failed: {e}")
            return None
    
    def _restore_backup(self, path: Path) -> bool:
        """
        Restore file from backup
        
        Returns:
            True if restore successful, False otherwise
        """
        backup_path = path.with_suffix(path.suffix + '.backup')
        
        if not backup_path.exists():
            logger.warning(f"No backup found: {backup_path}")
            return False
        
        try:
            shutil.copy2(backup_path, path)
            logger.info(f"Restored from backup: {path}")
            return True
        except Exception as e:
            logger.error(f"Backup restore failed: {e}")
            return False
    
    def load_pickle(
        self,
        path: Path,
        fallback: Optional[T] = None,
        validator: Optional[Callable[[Any], bool]] = None
    ) -> Optional[T]:
        """
        Load pickle file with comprehensive error handling
        
        Args:
            path: Path to pickle file
            fallback: Value to return if load fails
            validator: Optional function to validate loaded object
            
        Returns:
            Loaded object, fallback value, or None
        """
        self.stats.total_reads += 1
        
        try:
            # Validate path
            if not self._validate_path(path):
                self.stats.failed_reads += 1
                return fallback
            
            # Check file exists
            if not path.exists():
                logger.warning(f"Pickle file not found: {path}")
                self.stats.failed_reads += 1
                return fallback
            
            # Check file size (basic corruption check)
            if path.stat().st_size == 0:
                logger.error(f"Pickle file is empty: {path}")
                self.stats.failed_reads += 1
                return fallback
            
            # Load pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate if validator provided
            if validator and not validator(data):
                logger.error(f"Pickle validation failed: {path}")
                self.stats.failed_reads += 1
                return fallback
            
            self.stats.successful_reads += 1
            logger.debug(f"Loaded pickle: {path}")
            return data
            
        except pickle.UnpicklingError as e:
            logger.error(f"Pickle corrupted: {path} - {e}")
            self.stats.failed_reads += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.now()
            return fallback
        
        except (IOError, OSError, PermissionError) as e:
            logger.error(f"Pickle I/O error: {path} - {e}")
            self.stats.failed_reads += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.now()
            return fallback
        
        except Exception as e:
            logger.error(f"Unexpected pickle load error: {path} - {e}")
            self.stats.failed_reads += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.now()
            return fallback
    
    def save_pickle(
        self,
        data: Any,
        path: Path,
        atomic: bool = True
    ) -> bool:
        """
        Save data to pickle file with atomic writes
        
        Args:
            data: Data to save
            path: Path to save to
            atomic: Use atomic write (write to temp, then rename)
            
        Returns:
            True if save successful, False otherwise
        """
        self.stats.total_writes += 1
        
        try:
            # Validate path
            if not self._validate_path(path):
                self.stats.failed_writes += 1
                return False
            
            # Check disk space
            if not self._check_disk_space(path):
                self.stats.failed_writes += 1
                return False
            
            # Create backup
            backup_path = self._create_backup(path)
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if atomic:
                # Atomic write: write to temp file, then rename
                with tempfile.NamedTemporaryFile(
                    mode='wb',
                    dir=path.parent,
                    delete=False
                ) as tmp_file:
                    pickle.dump(data, tmp_file)
                    tmp_path = Path(tmp_file.name)
                
                # Rename temp file to final path (atomic on most systems)
                tmp_path.replace(path)
            else:
                # Direct write
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
            
            self.stats.successful_writes += 1
            logger.debug(f"Saved pickle: {path}")
            return True
            
        except (IOError, OSError, PermissionError) as e:
            logger.error(f"Pickle save error: {path} - {e}")
            self.stats.failed_writes += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.now()
            
            # Try to restore backup
            if backup_path:
                self._restore_backup(path)
            
            return False
        
        except Exception as e:
            logger.error(f"Unexpected pickle save error: {path} - {e}")
            self.stats.failed_writes += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.now()
            
            if backup_path:
                self._restore_backup(path)
            
            return False
    
    def load_json(
        self,
        path: Path,
        fallback: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load JSON file with comprehensive error handling
        
        Args:
            path: Path to JSON file
            fallback: Value to return if load fails
            
        Returns:
            Loaded JSON data, fallback value, or None
        """
        self.stats.total_reads += 1
        
        try:
            if not self._validate_path(path):
                self.stats.failed_reads += 1
                return fallback
            
            if not path.exists():
                logger.warning(f"JSON file not found: {path}")
                self.stats.failed_reads += 1
                return fallback
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.stats.successful_reads += 1
            logger.debug(f"Loaded JSON: {path}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON malformed: {path} - {e}")
            self.stats.failed_reads += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.now()
            return fallback
        
        except (IOError, OSError, PermissionError) as e:
            logger.error(f"JSON I/O error: {path} - {e}")
            self.stats.failed_reads += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.now()
            return fallback
        
        except Exception as e:
            logger.error(f"Unexpected JSON load error: {path} - {e}")
            self.stats.failed_reads += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.now()
            return fallback
    
    def save_json(
        self,
        data: Dict[str, Any],
        path: Path,
        atomic: bool = True,
        pretty: bool = True
    ) -> bool:
        """
        Save data to JSON file with atomic writes
        
        Args:
            data: Data to save (must be JSON-serializable)
            path: Path to save to
            atomic: Use atomic write
            pretty: Pretty-print JSON
            
        Returns:
            True if save successful, False otherwise
        """
        self.stats.total_writes += 1
        
        try:
            if not self._validate_path(path):
                self.stats.failed_writes += 1
                return False
            
            if not self._check_disk_space(path):
                self.stats.failed_writes += 1
                return False
            
            backup_path = self._create_backup(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if atomic:
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    dir=path.parent,
                    delete=False,
                    encoding='utf-8'
                ) as tmp_file:
                    json.dump(data, tmp_file, indent=2 if pretty else None)
                    tmp_path = Path(tmp_file.name)
                
                tmp_path.replace(path)
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2 if pretty else None)
            
            self.stats.successful_writes += 1
            logger.debug(f"Saved JSON: {path}")
            return True
            
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization error: {path} - {e}")
            self.stats.failed_writes += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.now()
            return False
        
        except (IOError, OSError, PermissionError) as e:
            logger.error(f"JSON save error: {path} - {e}")
            self.stats.failed_writes += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.now()
            
            if backup_path:
                self._restore_backup(path)
            
            return False
        
        except Exception as e:
            logger.error(f"Unexpected JSON save error: {path} - {e}")
            self.stats.failed_writes += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.now()
            
            if backup_path:
                self._restore_backup(path)
            
            return False
    
    def load_text(
        self,
        path: Path,
        fallback: Optional[str] = None,
        encoding: str = 'utf-8'
    ) -> Optional[str]:
        """
        Load text file with error handling
        
        Args:
            path: Path to text file
            fallback: Value to return if load fails
            encoding: Text encoding
            
        Returns:
            File content as string, fallback value, or None
        """
        self.stats.total_reads += 1
        
        try:
            if not self._validate_path(path):
                self.stats.failed_reads += 1
                return fallback
            
            if not path.exists():
                logger.warning(f"Text file not found: {path}")
                self.stats.failed_reads += 1
                return fallback
            
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()
            
            self.stats.successful_reads += 1
            return content
            
        except (UnicodeDecodeError, UnicodeError) as e:
            logger.error(f"Text encoding error: {path} - {e}")
            self.stats.failed_reads += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.now()
            return fallback
        
        except (IOError, OSError, PermissionError) as e:
            logger.error(f"Text I/O error: {path} - {e}")
            self.stats.failed_reads += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.now()
            return fallback
        
        except Exception as e:
            logger.error(f"Unexpected text load error: {path} - {e}")
            self.stats.failed_reads += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.now()
            return fallback
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file I/O statistics"""
        return {
            'reads': {
                'total': self.stats.total_reads,
                'successful': self.stats.successful_reads,
                'failed': self.stats.failed_reads,
                'success_rate': self.stats.success_rate('read')
            },
            'writes': {
                'total': self.stats.total_writes,
                'successful': self.stats.successful_writes,
                'failed': self.stats.failed_writes,
                'success_rate': self.stats.success_rate('write')
            },
            'last_error': self.stats.last_error,
            'last_error_time': self.stats.last_error_time.isoformat() if self.stats.last_error_time else None
        }


# Singleton instance
_file_io_handler: Optional[BulletproofFileIO] = None


def get_file_io_handler(base_path: Optional[Path] = None) -> BulletproofFileIO:
    """Get singleton file I/O handler"""
    global _file_io_handler
    
    if _file_io_handler is None:
        _file_io_handler = BulletproofFileIO(base_path=base_path)
    
    return _file_io_handler


def load_model_safe(model_path: Path, fallback: Any = None) -> Any:
    """
    Convenience function to safely load ML model
    
    Args:
        model_path: Path to model file (.pkl)
        fallback: Value to return if load fails
        
    Returns:
        Loaded model or fallback value
    """
    handler = get_file_io_handler()
    return handler.load_pickle(model_path, fallback=fallback)


def load_config_safe(config_path: Path, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to safely load config
    
    Args:
        config_path: Path to config file (.json)
        fallback: Value to return if load fails
        
    Returns:
        Loaded config or fallback value
    """
    handler = get_file_io_handler()
    return handler.load_json(config_path, fallback=fallback or {})
