"""
Calibration Config Deployer

Handles safe deployment of calibration configurations:
- Atomic config file updates
- Versioning and archiving
- Rollback support
- AI Engine signaling
"""
import os
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from microservices.learning.calibration_types import (
    Calibration,
    CalibrationDeploymentError
)

logger = logging.getLogger(__name__)


class CalibrationConfigDeployer:
    """
    Deploys calibration configurations safely with versioning and rollback.
    
    Safety guarantees:
    - Atomic file writes (tmp → rename)
    - Full version history in archive
    - Instant rollback capability
    - Validation before deployment
    """
    
    def __init__(
        self,
        config_path: str = "/home/qt/quantum_trader/config/calibration.json",
        archive_dir: str = "/home/qt/quantum_trader/config/calibration_archive"
    ):
        self.config_path = Path(config_path)
        self.archive_dir = Path(archive_dir)
        
        # Ensure directories exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[CalibrationDeployer] Config: {self.config_path}")
        logger.info(f"[CalibrationDeployer] Archive: {self.archive_dir}")
    
    def create_snapshot(self, version: str) -> Optional[Path]:
        """
        Create backup snapshot of current configuration.
        
        Args:
            version: Version identifier for snapshot
        
        Returns:
            Path to snapshot file, or None if no current config exists
        """
        if not self.config_path.exists():
            logger.warning("No current config to snapshot")
            return None
        
        # Generate snapshot filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot_name = f"{version}_backup_{timestamp}.json"
        snapshot_path = self.archive_dir / snapshot_name
        
        try:
            # Copy current config to archive
            shutil.copy2(self.config_path, snapshot_path)
            logger.info(f"✅ Snapshot created: {snapshot_path.name}")
            return snapshot_path
            
        except Exception as e:
            logger.error(f"❌ Failed to create snapshot: {e}")
            raise CalibrationDeploymentError(f"Snapshot failed: {e}")
    
    def deploy_calibration(
        self,
        calibration: Calibration,
        dry_run: bool = False
    ) -> bool:
        """
        Deploy calibration configuration atomically.
        
        Process:
        1. Create snapshot of current config
        2. Validate new calibration
        3. Write to temporary file
        4. Atomic rename to production path
        5. Archive the deployed config
        
        Args:
            calibration: Calibration to deploy
            dry_run: If True, validate but don't actually deploy
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"{'[DRY-RUN] ' if dry_run else ''}Deploying calibration: {calibration.version}")
        
        # Step 1: Snapshot current config (if exists)
        if self.config_path.exists() and not dry_run:
            snapshot = self.create_snapshot(calibration.version)
            if snapshot:
                logger.info(f"  Snapshot: {snapshot.name}")
        
        # Step 2: Validate calibration
        try:
            config_dict = calibration.to_dict()
            
            # Ensure critical fields exist
            required_fields = [
                'version', 'created_at', 'confidence_calibration',
                'ensemble_weights', 'metadata'
            ]
            
            for field in required_fields:
                if field not in config_dict:
                    raise CalibrationDeploymentError(f"Missing required field: {field}")
            
            # Validate JSON serializability
            json_str = json.dumps(config_dict, indent=2)
            
            if dry_run:
                logger.info(f"  ✅ Validation passed (dry-run mode)")
                logger.info(f"  Config size: {len(json_str)} bytes")
                return True
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            raise CalibrationDeploymentError(f"Validation error: {e}")
        
        # Step 3: Write to temporary file
        temp_path = self.config_path.parent / f"{self.config_path.name}.tmp"
        
        try:
            with open(temp_path, 'w') as f:
                f.write(json_str)
            
            logger.info(f"  Wrote temp file: {temp_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to write temp file: {e}")
            raise CalibrationDeploymentError(f"Write error: {e}")
        
        # Step 4: Atomic rename
        try:
            # On Unix, rename is atomic
            temp_path.rename(self.config_path)
            logger.info(f"  ✅ Deployed: {self.config_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to deploy: {e}")
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            raise CalibrationDeploymentError(f"Deploy error: {e}")
        
        # Step 5: Archive deployed config
        try:
            archive_name = f"{calibration.version}_deployed.json"
            archive_path = self.archive_dir / archive_name
            
            shutil.copy2(self.config_path, archive_path)
            logger.info(f"  Archived: {archive_name}")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to archive: {e}")
            # Not critical, deployment already succeeded
        
        # Step 6: Signal AI Engine to reload (optional, depends on implementation)
        self._signal_config_reload()
        
        logger.info(f"✅ Calibration {calibration.version} deployed successfully")
        
        return True
    
    def rollback(self, to_version: Optional[str] = None) -> bool:
        """
        Rollback to previous calibration version.
        
        Args:
            to_version: Specific version to rollback to. If None, uses most recent backup.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"🔄 Initiating rollback{f' to {to_version}' if to_version else ''}...")
        
        # Find version to rollback to
        if to_version:
            # Look for specific version
            candidates = list(self.archive_dir.glob(f"{to_version}_*.json"))
            
            if not candidates:
                logger.error(f"❌ Version {to_version} not found in archive")
                return False
            
            # Use most recent if multiple matches
            backup_path = max(candidates, key=lambda p: p.stat().st_mtime)
            
        else:
            # Use most recent backup
            backups = sorted(
                self.archive_dir.glob("*_backup_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            if not backups:
                logger.error("❌ No backups found in archive")
                return False
            
            backup_path = backups[0]
        
        logger.info(f"  Rolling back to: {backup_path.name}")
        
        try:
            # Load backup
            with open(backup_path, 'r') as f:
                config_dict = json.load(f)
            
            # Validate it's a valid calibration
            Calibration.from_dict(config_dict)
            
            # Create snapshot of current config before rollback
            if self.config_path.exists():
                self.create_snapshot("rollback_from")
            
            # Copy backup to current config (via atomic rename)
            temp_path = self.config_path.parent / f"{self.config_path.name}.tmp"
            
            shutil.copy2(backup_path, temp_path)
            temp_path.rename(self.config_path)
            
            # Signal reload
            self._signal_config_reload()
            
            logger.info(f"✅ Rollback successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all calibration versions in archive.
        
        Returns:
            List of version info dictionaries
        """
        versions = []
        
        for path in sorted(self.archive_dir.glob("*.json"), reverse=True):
            try:
                stat = path.stat()
                
                # Parse version from filename
                # Format: {version}_backup_{timestamp}.json or {version}_deployed.json
                filename = path.stem  # Remove .json
                
                if "_backup_" in filename:
                    version = filename.split("_backup_")[0]
                    version_type = "backup"
                elif "_deployed" in filename:
                    version = filename.replace("_deployed", "")
                    version_type = "deployed"
                else:
                    version = filename
                    version_type = "unknown"
                
                versions.append({
                    "version": version,
                    "type": version_type,
                    "filename": path.name,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": str(path)
                })
                
            except Exception as e:
                logger.warning(f"Failed to parse {path.name}: {e}")
                continue
        
        return versions
    
    def get_current_version(self) -> Optional[str]:
        """
        Get version of currently deployed calibration.
        
        Returns:
            Version string, or None if no config deployed
        """
        if not self.config_path.exists():
            return None
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            return config.get('version')
            
        except Exception as e:
            logger.error(f"Failed to read current version: {e}")
            return None
    
    def _signal_config_reload(self) -> None:
        """
        Signal AI Engine to reload calibration config.
        
        Options:
        1. Touch a sentinel file that AI Engine watches
        2. Publish Redis event
        3. HTTP endpoint call
        
        For now, using sentinel file approach (simplest).
        """
        try:
            sentinel_path = self.config_path.parent / ".calibration_reload"
            sentinel_path.touch()
            logger.info(f"  Signal sent: {sentinel_path}")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to signal reload: {e}")
            logger.warning("   AI Engine will reload on next restart")
    
    def validate_current_config(self) -> bool:
        """
        Validate currently deployed configuration.
        
        Returns:
            True if valid, False otherwise
        """
        if not self.config_path.exists():
            logger.warning("No config file exists")
            return False
        
        try:
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Try to parse as Calibration
            calibration = Calibration.from_dict(config_dict)
            
            # Basic validation
            assert calibration.version
            assert calibration.created_at
            assert calibration.confidence
            assert calibration.weights
            
            logger.info(f"✅ Current config valid: {calibration.version}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Current config invalid: {e}")
            return False
