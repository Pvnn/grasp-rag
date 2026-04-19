# 1. Force Python to output UTF-8
$env:PYTHONIOENCODING="utf-8"

# 2. Force PowerShell to read and write UTF-8 through its pipes and console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$logDir = "logs"
$masterLog = "$logDir\master_preprocess.log"

# Create logs directory if it doesn't exist
if (-Not (Test-Path -Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

# Initialize master log
$header = @"
======================================================
Starting Preprocessing Pipeline
Date: $(Get-Date)
======================================================
"@
$header | Out-File -FilePath $masterLog -Encoding utf8

# Exact order of preprocessing scripts
$scripts = @(
    "scripts\preprocess_2wiki.py",
    "scripts\preprocess_tqa.py",
    "scripts\preprocess_hotpotqa.py",
    "scripts\preprocess_asqa.py",
    "scripts\preprocess_nq.py"
)
$total = $scripts.Count

for ($i = 0; $i -lt $total; $i++) {
    $scriptPath = $scripts[$i]
    
    # Extract base name for logging (e.g., preprocess_2wiki)
    $scriptName = [System.IO.Path]::GetFileNameWithoutExtension($scriptPath)
    $scriptLog = "$logDir\$scriptName.log"
    
    $startMsg = "`n------------------------------------------------------`n[INFO] Running: $scriptPath"
    Write-Host $startMsg -ForegroundColor Cyan
    $startMsg | Out-File -FilePath $masterLog -Append -Encoding utf8
    
    try {
        # 3. Force cmd.exe to use UTF-8 (chcp 65001) before running Python
        cmd.exe /c "chcp 65001 >NUL && python $scriptPath 2>&1" | Tee-Object -FilePath $scriptLog | Tee-Object -FilePath $masterLog -Append
        
        # Check if the python executable threw an error code
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] $scriptName failed with exit code $LASTEXITCODE." -ForegroundColor Red
            Write-Host "[INFO] Moving gracefully to the next script..." -ForegroundColor Yellow
            "[ERROR] $scriptName failed with exit code $LASTEXITCODE." | Out-File -FilePath $masterLog -Append -Encoding utf8
        } else {
            Write-Host "[SUCCESS] $scriptName completed successfully." -ForegroundColor Green
            "[SUCCESS] $scriptName completed successfully." | Out-File -FilePath $masterLog -Append -Encoding utf8
        }
    } catch {
        Write-Host "[CRITICAL ERROR] Could not launch $scriptPath." -ForegroundColor Red
        "[CRITICAL ERROR] Could not launch $scriptPath." | Out-File -FilePath $masterLog -Append -Encoding utf8
    }
    
    # Sleep for 1 minute (60s) after every 2 scripts, unless it's the last script
    $scriptNum = $i + 1
    if (($scriptNum % 2 -eq 0) -and ($scriptNum -ne $total)) {
        $sleepMsg = "[WAIT] Sleeping for 60 seconds to let the system cool down..."
        Write-Host $sleepMsg -ForegroundColor DarkYellow
        $sleepMsg | Out-File -FilePath $masterLog -Append -Encoding utf8
        Start-Sleep -Seconds 60
    }
}

$endMsg = "`n======================================================`n[INFO] All preprocessing scripts have finished."
Write-Host $endMsg -ForegroundColor Cyan
$endMsg | Out-File -FilePath $masterLog -Append -Encoding utf8