# Update remaining imports after P2-01 migration
$files = Get-ChildItem -Path . -Filter "*.py" -Recurse -Exclude "*_archive*","*__pycache__*","*.venv*" | 
    Where-Object { $_.FullName -notlike "*\node_modules\*" }

$count = 0
$replacements = @{
    'from backend.services.binance_futures_execution_adapter import' = 'from backend.services.execution.binance_futures_execution_adapter import'
    'from backend.services.binance_execution import' = 'from backend.services.execution.binance_execution import'
    'from backend.services.model_supervisor import' = 'from backend.services.ai.model_supervisor import'
    'from backend.services.continuous_learning_manager import' = 'from backend.services.ai.continuous_learning_manager import'
    'from backend.services.rl_v3_training_daemon import' = 'from backend.services.ai.rl_v3_training_daemon import'
    'from backend.services.rl_position_sizing_agent import' = 'from backend.services.ai.rl_position_sizing_agent import'
    'from backend.services.rl_reward_engine_v2 import' = 'from backend.services.ai.rl_reward_engine_v2 import'
    'from backend.services.rl_state_manager_v2 import' = 'from backend.services.ai.rl_state_manager_v2 import'
    'from backend.services.rl_action_space_v2 import' = 'from backend.services.ai.rl_action_space_v2 import'
    'from backend.services.rl_episode_tracker_v2 import' = 'from backend.services.ai.rl_episode_tracker_v2 import'
    'from backend.services.rl_v3_live_orchestrator import' = 'from backend.services.ai.rl_v3_live_orchestrator import'
}

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    $modified = $false
    
    foreach ($old in $replacements.Keys) {
        if ($content -match [regex]::Escape($old)) {
            $content = $content -replace [regex]::Escape($old), $replacements[$old]
            $modified = $true
        }
    }
    
    if ($modified) {
        Set-Content -Path $file.FullName -Value $content -NoNewline
        $count++
        Write-Host "Updated: $($file.FullName)"
    }
}

Write-Host "`nTotal files updated: $count"
