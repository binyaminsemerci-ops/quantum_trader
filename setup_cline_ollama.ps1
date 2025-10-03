<#  setup_cline_ollama.ps1
    Setter opp Cline + Ollama lokalt (gratis) i et VS Code-prosjekt.
    - Lager/oppdaterer .vscode\settings.json for Cline → Ollama
    - Sjekker at Ollama kjører og at valgt modell finnes (trekker om nødvendig)
    - Lager .clineignore for fart
    - Skriver en kort statusrapport
#>

param(
  [string]$ProjectPath = (Get-Location).Path,
  [string]$Model = "qwen2.5-coder:7b-instruct",
  [string]$ApiBaseUrl = "http://localhost:11434/v1",
  [switch]$SkipModelPull
)

Write-Host "==> Prosjektmappe:" $ProjectPath -ForegroundColor Cyan
if (-not (Test-Path $ProjectPath)) {
  Write-Error "Finner ikke sti: $ProjectPath"
  exit 1
}

# 1) Sørg for .vscode/
$vscode = Join-Path $ProjectPath ".vscode"
if (-not (Test-Path $vscode)) {
  New-Item -ItemType Directory -Path $vscode | Out-Null
}

# 2) Skriv settings.json for Cline → Ollama
$settingsPath = Join-Path $vscode "settings.json"
$settingsJson = @"
{
  "cline.openaiApiBaseUrl": "$ApiBaseUrl",
  "cline.openaiApiKey": "ollama",
  "cline.model": "$Model",
  "cline.maxModelTokens": 4096,
  "cline.responseTokens": 512,
  "cline.enableCodeEdits": true,
  "cline.diffMaxEdits": 100,
  "cline.autoApproveFileRead": true,
  "cline.autoApproveTerminal": false
}
"@
$settingsJson | Set-Content -Encoding UTF8 $settingsPath
Write-Host "✓ Skrev $settingsPath" -ForegroundColor Green

# 3) Lag .clineignore (for hastighet)
$ignorePath = Join-Path $ProjectPath ".clineignore"
$ignoreContent = @"
node_modules/
.git/
.venv/
dist/
build/
.cache/
coverage/
"@
$ignoreContent | Set-Content -Encoding UTF8 $ignorePath
Write-Host "✓ Skrev $ignorePath" -ForegroundColor Green

# 4) Helsesjekk: Ollama kjører?
function Test-Ollama {
  try {
    $resp = Invoke-WebRequest -Uri "$ApiBaseUrl/models" -UseBasicParsing -TimeoutSec 3
    return $resp.StatusCode -eq 200
  } catch {
    return $false
  }
}

$ollamaOk = Test-Ollama
if (-not $ollamaOk) {
  Write-Warning "Ollama API svarer ikke på $ApiBaseUrl. Start evt. i eget vindu:  ollama serve"
} else {
  Write-Host "✓ Ollama API tilgjengelig på $ApiBaseUrl" -ForegroundColor Green
}

# 5) Modell tilgjengelig? (ollama list)
function Get-OllamaList {
  try { (ollama list) 2>$null } catch { "" }
}
$list = Get-OllamaList

if (-not $list) {
  Write-Warning "Kunne ikke lese 'ollama list'. Er Ollama installert i PATH? (winget install Ollama.Ollama)"
} else {
  if ($list -notmatch [regex]::Escape($Model)) {
    if ($SkipModelPull) {
      Write-Warning "Modellen '$Model' finnes ikke lokalt. (Hoppet over nedlasting pga. -SkipModelPull)"
    } else {
      Write-Host "⏳ Laster ned modell '$Model' (første gang tar det litt tid)..." -ForegroundColor Yellow
      try {
        ollama pull $Model
        Write-Host "✓ Modell hentet: $Model" -ForegroundColor Green
      } catch {
        Write-Warning "Klarte ikke å hente modellen. Du kan prøve manuelt:  ollama pull $Model"
      }
    }
  } else {
    Write-Host "✓ Modell finnes lokalt: $Model" -ForegroundColor Green
  }
}

# 6) Liten ytelsestweak (frivillig info)
Write-Host "`nTips for fart (frivillig):" -ForegroundColor DarkCyan
Write-Host " - Sett miljøvariabel midlertidig før 'ollama serve':  `$env:OLLAMA_MAX_LOADED_MODELS = '1'"
Write-Host " - Med CPU:  `$env:OLLAMA_NUM_THREADS = [int][Math]::Max(2, [Environment]::ProcessorCount - 2) `n"

# 7) Ferdigmelding
Write-Host "✅ Ferdig! Åpne/Reload VS Code i $ProjectPath" -ForegroundColor Green
Write-Host "   Ctrl+Shift+P → Developer: Reload Window"
Write-Host "   Start Cline og bekreft at modellen viser: $Model"
