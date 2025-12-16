# Load environment variables from .env
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        $key = $matches[1]
        $value = $matches[2]
        [Environment]::SetEnvironmentVariable($key, $value, 'Process')
        Write-Host "Loaded: $key"
    }
}

# Set Python path
$env:PYTHONPATH = "c:\quantum_trader"

# Start backend
Write-Host "`nStarting backend on port 8000..."
python -m uvicorn main:app --reload --port 8000 --host 0.0.0.0
