# 1. Force Python to output UTF-8 and run unbuffered
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUNBUFFERED="1"

# 2. Force PowerShell to read and write UTF-8 through its pipes and console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$logDir = "logs"
$ablationLog = "$logDir\master_ablation.log"

# Create logs directory if it doesn't exist
if (-Not (Test-Path -Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

$header = @"
======================================================
Starting GRASP Ablation Study Pipeline
Date: $(Get-Date)
======================================================
"@
$header | Out-File -FilePath $ablationLog -Encoding utf8

# Set sample size (Default to 300)
$numSamples = 300
if ($args.Count -gt 0) {
    if ($args[0] -match '^\d+$') {
        $numSamples = $args[0]
    }
}

Write-Host "`n[INFO] Starting Ablation Runner with N=$numSamples" -ForegroundColor Cyan
Write-Host "[INFO] This script supports incremental saving. If stopped, it will resume where it left off.`n" -ForegroundColor Yellow

$scriptPath = "scripts\run_ablation.py"

try {
    # 3. Force cmd.exe to use UTF-8 (chcp 65001) before running Python
    cmd.exe /c "chcp 65001 >NUL && python $scriptPath -n $numSamples 2>&1" | Tee-Object -FilePath $ablationLog -Append
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[ERROR] Ablation runner failed with exit code $LASTEXITCODE." -ForegroundColor Red
        "[ERROR] Ablation runner failed with exit code $LASTEXITCODE." | Out-File -FilePath $ablationLog -Append -Encoding utf8
    } else {
        Write-Host "`n[SUCCESS] Ablation pipeline completed successfully." -ForegroundColor Green
        "[SUCCESS] Ablation pipeline completed successfully." | Out-File -FilePath $ablationLog -Append -Encoding utf8
    }
} catch {
    Write-Host "`n[CRITICAL ERROR] Could not launch $scriptPath." -ForegroundColor Red
    "[CRITICAL ERROR] Could not launch $scriptPath." | Out-File -FilePath $ablationLog -Append -Encoding utf8
}
