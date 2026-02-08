"""
Calibration Orchestrator - Main Coordination Logic

Coordinates the complete calibration workflow:
1. Check Learning Cadence authorization
2. Load and analyze CLM data
3. Generate calibration configuration  
4. Validate safety checks
5. Generate human-readable report
6. Deploy configuration (with approval)
7. Mark completion in Learning Cadence

This is the main entry point for calibration operations.
"""
import logging
import requests
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from microservices.learning.calibration_analyzer import CalibrationAnalyzer
from microservices.learning.calibration_report import CalibrationReportGenerator
from microservices.learning.calibration_deployer import CalibrationConfigDeployer
from microservices.learning.calibration_types import (
    Calibration,
    CalibrationJob,
    CalibrationStatus,
    CalibrationMetadata,
    CalibrationNotReadyError,
    CalibrationNotAuthorizedError,
    CalibrationValidationError
)

logger = logging.getLogger(__name__)


class CalibrationOrchestrator:
    """
    Main orchestrator for calibration-only learning.
    
    Workflow:
    1. check_readiness() - Query Learning Cadence
    2. start_calibration() - Run analysis, generate report
    3. approve_and_deploy() - Deploy approved calibration
    4. rollback() - Revert to previous version if needed
    """
    
    def __init__(
        self,
        cadence_api: str = "http://127.0.0.1:8003",
        clm_data_path: str = "/home/qt/quantum_trader/data/clm_trades.jsonl",
        report_dir: str = "/tmp"
    ):
        self.cadence_api = cadence_api
        self.clm_data_path = clm_data_path
        
        # Initialize components
        self.analyzer = CalibrationAnalyzer()
        self.report_gen = CalibrationReportGenerator(output_dir=report_dir)
        self.deployer = CalibrationConfigDeployer()
        
        # Job tracking
        self.jobs: Dict[str, CalibrationJob] = {}
        
        logger.info("[CalibrationOrchestrator] Initialized")
        logger.info(f"  Cadence API: {self.cadence_api}")
        logger.info(f"  CLM Data: {self.clm_data_path}")
        logger.info(f"  Report Dir: {report_dir}")
    
    def check_readiness(self) -> Dict[str, Any]:
        """
        Check if calibration is authorized by Learning Cadence.
        
        Returns:
            Readiness status from Learning Cadence API
        
        Raises:
            ConnectionError: If Learning Cadence API is unavailable
        """
        logger.info("🔍 Checking Learning Cadence authorization...")
        
        try:
            response = requests.get(
                f"{self.cadence_api}/readiness/simple",
                timeout=5.0
            )
            
            if response.status_code != 200:
                raise ConnectionError(f"API returned {response.status_code}")
            
            data = response.json()
            
            ready = data.get('ready', False)
            reason = data.get('reason', 'unknown')
            actions = data.get('actions', [])
            
            status_emoji = "🟢" if ready else "⏸️"
            
            logger.info(f"  {status_emoji} Ready: {ready}")
            logger.info(f"  Reason: {reason}")
            logger.info(f"  Allowed actions: {actions}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to check readiness: {e}")
            raise ConnectionError(f"Learning Cadence API unavailable: {e}")
    
    def start_calibration(
        self,
        dry_run: bool = True,
        force: bool = False
    ) -> CalibrationJob:
        """
        Start calibration analysis workflow.
        
        Steps:
        1. Check authorization (unless force=True)
        2. Load CLM data
        3. Run calibration analysis
        4. Validate results
        5. Generate report
        6. Create job for approval
        
        Args:
            dry_run: If True, generate report but don't deploy
            force: If True, skip authorization check (for testing)
        
        Returns:
            CalibrationJob with analysis results and report
        
        Raises:
            CalibrationNotReadyError: If Learning Cadence hasn't authorized
            CalibrationNotAuthorizedError: If 'calibration' not in allowed actions
            CalibrationValidationError: If calibration fails safety checks
        """
        logger.info(f"🚀 Starting calibration (dry_run={dry_run}, force={force})")
        
        # Step 1: Check authorization
        if not force:
            try:
                readiness = self.check_readiness()
                
                if not readiness['ready']:
                    raise CalibrationNotReadyError(
                        f"Learning Cadence not ready: {readiness['reason']}"
                    )
                
                if 'calibration' not in readiness.get('actions', []):
                    raise CalibrationNotAuthorizedError(
                        f"Only {readiness['actions']} authorized, not 'calibration'"
                    )
                
                logger.info("✅ Learning Cadence authorized calibration")
                
            except ConnectionError as e:
                if not force:
                    raise CalibrationNotReadyError(f"Cannot verify authorization: {e}")
                else:
                    logger.warning(f"⚠️  Authorization check failed (force mode): {e}")
        else:
            logger.warning("⚠️  FORCE MODE - Skipping authorization check")
        
        # Step 2: Load CLM data
        logger.info(f"📊 Loading CLM data from {self.clm_data_path}")
        
        try:
            trades = self.analyzer.load_clm_data(self.clm_data_path)
            logger.info(f"  Loaded {len(trades)} trades")
            
        except Exception as e:
            logger.error(f"❌ Failed to load CLM data: {e}")
            raise
        
        # Step 3: Run calibration analysis
        logger.info("🔬 Running calibration analysis...")
        
        try:
            # Confidence calibration
            conf_cal = self.analyzer.calibrate_confidence(trades)
            
            # Ensemble weights
            weight_cal = self.analyzer.calibrate_ensemble_weights(trades)
            
            # HOLD bias (currently disabled)
            hold_cal = self.analyzer.calibrate_hold_bias(trades)
            
            logger.info("  ✅ Analysis complete")
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise
        
        # Step 4: Create Calibration object
        version = f"cal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        calibration = Calibration(
            version=version,
            created_at=datetime.now(),
            based_on_trades=len(trades),
            approved_by=None,
            approved_at=None,
            confidence=conf_cal,
            weights=weight_cal,
            hold_bias=hold_cal,
            metadata=CalibrationMetadata(
                win_rate_improvement=None,  # TODO: Compute
                confidence_mse_before=conf_cal.mse_before,
                confidence_mse_after=conf_cal.mse_after,
                weight_stability=weight_cal.total_delta,
                validation_status="pending"
            )
        )
        
        # Step 5: Validate calibration
        logger.info("🔍 Validating calibration...")
        
        validation = self.analyzer.validate_calibration(
            conf_cal, weight_cal, hold_cal
        )
        
        calibration.validation_checks = validation.checks
        calibration.metadata.validation_status = "passed" if validation.passed else "failed"
        calibration.metadata.risk_score = validation.risk_score
        
        if not validation.passed:
            logger.error("❌ Calibration validation FAILED")
            logger.error(f"   Errors: {validation.errors}")
            
            if not force:
                raise CalibrationValidationError(
                    f"Validation failed: {'; '.join(validation.errors)}"
                )
            else:
                logger.warning("⚠️  FORCE MODE - Continuing despite validation failure")
        else:
            logger.info(f"  ✅ Validation passed (risk={validation.risk_score:.1%})")
        
        # Step 6: Generate report
        logger.info("📝 Generating report...")
        
        try:
            report_path = self.report_gen.generate_report(
                calibration, trades, validation
            )
            logger.info(f"  Report: {report_path}")
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            report_path = None
        
        # Step 7: Create job
        job_id = version  # Use version as job ID
        
        job = CalibrationJob(
            id=job_id,
            status=CalibrationStatus.PENDING_APPROVAL if validation.passed else CalibrationStatus.REJECTED,
            calibration=calibration,
            report_path=report_path,
            created_at=datetime.now(),
            completed_at=None,
            error=None if validation.passed else "; ".join(validation.errors)
        )
        
        self.jobs[job_id] = job
        
        # Step 8: Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"✅ CALIBRATION ANALYSIS COMPLETE")
        logger.info(f"   Job ID: {job_id}")
        logger.info(f"   Status: {job.status.value}")
        logger.info(f"   Report: {report_path}")
        logger.info(f"   Validation: {'✅ PASSED' if validation.passed else '❌ FAILED'}")
        logger.info(f"   Risk: {validation.risk_score:.1%}")
        
        if dry_run:
            logger.info("")
            logger.info(f"🔍 DRY-RUN MODE - No deployment")
            logger.info(f"   Review report: cat {report_path}")
        else:
            if validation.passed:
                logger.info("")
                logger.info(f"⚠️  READY FOR APPROVAL")
                logger.info(f"   To deploy: calibration_cli.py approve {job_id}")
            else:
                logger.info("")
                logger.info(f"❌ CANNOT DEPLOY - Validation failed")
        
        logger.info("=" * 60)
        logger.info("")
        
        return job
    
    def approve_and_deploy(
        self,
        job_id: str,
        approved_by: str = "manual"
    ) -> bool:
        """
        Approve and deploy a calibration job.
        
        Args:
            job_id: Job ID to deploy
            approved_by: Name/ID of approver
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"✅ Approving calibration: {job_id}")
        
        # Load job
        if job_id not in self.jobs:
            logger.error(f"❌ Job {job_id} not found")
            return False
        
        job = self.jobs[job_id]
        
        # Check status
        if job.status != CalibrationStatus.PENDING_APPROVAL:
            logger.error(f"❌ Job not in PENDING_APPROVAL state (status={job.status.value})")
            return False
        
        # Mark as approved
        job.calibration.approved_by = approved_by
        job.calibration.approved_at = datetime.now()
        job.status = CalibrationStatus.APPROVED
        
        # Deploy
        logger.info("🚀 Deploying calibration...")
        
        try:
            success = self.deployer.deploy_calibration(
                job.calibration,
                dry_run=False
            )
            
            if not success:
                logger.error("❌ Deployment failed")
                job.status = CalibrationStatus.REJECTED
                return False
            
            job.status = CalibrationStatus.DEPLOYED
            job.completed_at = datetime.now()
            
            logger.info("✅ Calibration deployed successfully")
            
        except Exception as e:
            logger.error(f"❌ Deployment error: {e}")
            job.status = CalibrationStatus.REJECTED
            job.error = str(e)
            return False
        
        # Mark completion in Learning Cadence
        try:
            self._mark_training_completed("calibration")
            logger.info("✅ Marked completion in Learning Cadence")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to mark completion: {e}")
            logger.warning("   This won't affect deployment, but Learning Cadence won't be updated")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("🎉 CALIBRATION DEPLOYMENT COMPLETE")
        logger.info(f"   Version: {job.calibration.version}")
        logger.info(f"   Approved by: {approved_by}")
        logger.info(f"   Deployed at: {job.completed_at.isoformat()}")
        logger.info("")
        logger.info("📊 MONITORING RECOMMENDATIONS:")
        logger.info("   - Monitor win rate for 24-48 hours")
        logger.info("   - Check confidence alignment in META-V2 logs")
        logger.info("   - Verify drawdown hasn't increased")
        logger.info("")
        logger.info(f"🔙 ROLLBACK (if needed):")
        logger.info(f"   python calibration_cli.py rollback {job.calibration.version}")
        logger.info("=" * 60)
        logger.info("")
        
        return True
    
    def rollback(self, to_version: Optional[str] = None) -> bool:
        """
        Rollback to previous calibration.
        
        Args:
            to_version: Specific version to rollback to, or None for most recent
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"🔄 Initiating rollback{f' to {to_version}' if to_version else ''}...")
        
        try:
            success = self.deployer.rollback(to_version)
            
            if success:
                logger.info("✅ Rollback complete")
                logger.info("   AI Engine will use previous configuration")
            else:
                logger.error("❌ Rollback failed")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Rollback error: {e}")
            return False
    
    def list_jobs(self) -> Dict[str, CalibrationJob]:
        """Get all calibration jobs"""
        return self.jobs
    
    def get_job(self, job_id: str) -> Optional[CalibrationJob]:
        """Get specific calibration job"""
        return self.jobs.get(job_id)
    
    def _mark_training_completed(self, action: str) -> None:
        """
        Mark calibration completion in Learning Cadence.
        
        This updates the Learning Cadence state so it knows training occurred.
        """
        try:
            response = requests.post(
                f"{self.cadence_api}/training/completed",
                json={"action": action},
                timeout=5.0
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Marked '{action}' completion in Learning Cadence")
            else:
                logger.warning(f"⚠️  Learning Cadence returned {response.status_code}")
                
        except Exception as e:
            logger.warning(f"⚠️  Failed to mark completion: {e}")
            # Non-critical, deployment already succeeded
