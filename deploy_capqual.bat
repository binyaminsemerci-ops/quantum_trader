@echo off
echo === P0.CAP+QUAL DEPLOYMENT ===
echo.
echo Uploading deployment script...
wsl scp -i ~/.ssh/hetzner_fresh /mnt/c/quantum_trader/deploy_capqual_vps.sh root@46.224.116.254:/tmp/
echo.
echo Executing deployment on VPS...
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "bash /tmp/deploy_capqual_vps.sh 2>&1"
echo.
echo === DEPLOYMENT COMPLETE ===
echo Check output above for results.
pause
