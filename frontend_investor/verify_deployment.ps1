# QuantumFond Investor Portal - Pre-Deployment Verification
# Run this script before deploying to production

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   INVESTOR PORTAL PRE-DEPLOYMENT CHECK" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$ErrorCount = 0
$WarningCount = 0

# 1. Check directory structure
Write-Host "1. Checking project structure..." -ForegroundColor Yellow
$RequiredDirs = @("pages", "components", "hooks", "styles", "public")
foreach ($dir in $RequiredDirs) {
    if (Test-Path $dir) {
        Write-Host "   ✅ $dir/ exists" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $dir/ missing" -ForegroundColor Red
        $ErrorCount++
    }
}

# 2. Check configuration files
Write-Host "`n2. Checking configuration files..." -ForegroundColor Yellow
$ConfigFiles = @("package.json", "tsconfig.json", "next.config.js", "tailwind.config.js", ".env.local")
foreach ($file in $ConfigFiles) {
    if (Test-Path $file) {
        Write-Host "   ✅ $file exists" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $file missing" -ForegroundColor Red
        $ErrorCount++
    }
}

# 3. Check node_modules
Write-Host "`n3. Checking dependencies..." -ForegroundColor Yellow
if (Test-Path "node_modules") {
    Write-Host "   ✅ node_modules/ exists" -ForegroundColor Green
    
    $packages = @("next", "react", "react-dom", "recharts", "typescript", "tailwindcss")
    foreach ($pkg in $packages) {
        if (Test-Path "node_modules/$pkg") {
            Write-Host "   ✅ $pkg installed" -ForegroundColor Green
        } else {
            Write-Host "   ❌ $pkg missing" -ForegroundColor Red
            $ErrorCount++
        }
    }
} else {
    Write-Host "   ❌ node_modules not found - run npm install" -ForegroundColor Red
    $ErrorCount++
}

# 4. Check page files
Write-Host "`n4. Checking page files..." -ForegroundColor Yellow
$Pages = @("_app.tsx", "_document.tsx", "index.tsx", "login.tsx", "portfolio.tsx", "performance.tsx", "risk.tsx", "models.tsx", "reports.tsx")
foreach ($page in $Pages) {
    if (Test-Path "pages/$page") {
        Write-Host "   ✅ pages/$page exists" -ForegroundColor Green
    } else {
        Write-Host "   ❌ pages/$page missing" -ForegroundColor Red
        $ErrorCount++
    }
}

# 5. Check component files
Write-Host "`n5. Checking component files..." -ForegroundColor Yellow
$Components = @("InvestorNavbar.tsx", "MetricCard.tsx", "EquityChart.tsx", "ReportCard.tsx", "LoadingSpinner.tsx")
foreach ($comp in $Components) {
    if (Test-Path "components/$comp") {
        Write-Host "   ✅ components/$comp exists" -ForegroundColor Green
    } else {
        Write-Host "   ❌ components/$comp missing" -ForegroundColor Red
        $ErrorCount++
    }
}

# 6. Check hooks
Write-Host "`n6. Checking hooks..." -ForegroundColor Yellow
if (Test-Path "hooks/useAuth.ts") {
    Write-Host "   ✅ hooks/useAuth.ts exists" -ForegroundColor Green
} else {
    Write-Host "   ❌ hooks/useAuth.ts missing" -ForegroundColor Red
    $ErrorCount++
}

# 7. Check deployment scripts
Write-Host "`n7. Checking deployment scripts..." -ForegroundColor Yellow
$DeployScripts = @("deploy.sh", "deploy.ps1")
foreach ($script in $DeployScripts) {
    if (Test-Path $script) {
        Write-Host "   ✅ $script exists" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  $script missing (optional)" -ForegroundColor Yellow
        $WarningCount++
    }
}

# 8. Check documentation
Write-Host "`n8. Checking documentation..." -ForegroundColor Yellow
$Docs = @("README.md", "QUICKSTART.md", "SECURITY_REVIEW.md")
foreach ($doc in $Docs) {
    if (Test-Path $doc) {
        Write-Host "   ✅ $doc exists" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  $doc missing (recommended)" -ForegroundColor Yellow
        $WarningCount++
    }
}

# 9. Check Nginx configuration
Write-Host "`n9. Checking Nginx configuration..." -ForegroundColor Yellow
if (Test-Path "nginx.investor.quantumfond.conf") {
    Write-Host "   ✅ Nginx config exists" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Nginx config missing" -ForegroundColor Yellow
    $WarningCount++
}

# 10. Test TypeScript compilation
Write-Host "`n10. Testing TypeScript compilation..." -ForegroundColor Yellow
try {
    $tscOutput = npx tsc --noEmit 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ TypeScript compilation successful" -ForegroundColor Green
    } else {
        Write-Host "   ❌ TypeScript errors found:" -ForegroundColor Red
        Write-Host $tscOutput -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "   ⚠️  Could not run TypeScript check" -ForegroundColor Yellow
    $WarningCount++
}

# 11. Check environment variables
Write-Host "`n11. Checking environment variables..." -ForegroundColor Yellow
if (Test-Path ".env.local") {
    $envContent = Get-Content ".env.local" -Raw
    $requiredVars = @("NEXT_PUBLIC_API_URL", "NEXT_PUBLIC_AUTH_URL")
    foreach ($var in $requiredVars) {
        if ($envContent -match $var) {
            Write-Host "   ✅ $var configured" -ForegroundColor Green
        } else {
            Write-Host "   ❌ $var missing from .env.local" -ForegroundColor Red
            $ErrorCount++
        }
    }
} else {
    Write-Host "   ❌ .env.local not found" -ForegroundColor Red
    $ErrorCount++
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "            VERIFICATION SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($ErrorCount -eq 0 -and $WarningCount -eq 0) {
    Write-Host "`n✅ ALL CHECKS PASSED!" -ForegroundColor Green
    Write-Host "   The application is ready for deployment.`n" -ForegroundColor Green
} elseif ($ErrorCount -eq 0) {
    Write-Host "`n⚠️  $WarningCount WARNING(S) FOUND" -ForegroundColor Yellow
    Write-Host "   The application can be deployed, but review warnings.`n" -ForegroundColor Yellow
} else {
    Write-Host "`n❌ $ErrorCount ERROR(S) FOUND" -ForegroundColor Red
    Write-Host "   Fix errors before deploying to production.`n" -ForegroundColor Red
    exit 1
}

Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. npm run build         - Build production bundle" -ForegroundColor White
Write-Host "2. npm run start         - Test production locally" -ForegroundColor White
Write-Host "3. .\deploy.ps1          - Deploy to VPS`n" -ForegroundColor White
