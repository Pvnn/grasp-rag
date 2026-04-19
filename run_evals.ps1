param (
    [int]$n = 20
)

# 1. Force Python to output UTF-8
$env:PYTHONIOENCODING="utf-8"

# 2. Add PyTorch CUDA allocator config to reduce VRAM fragmentation and prevent OOM
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 3. Force PowerShell to read and write UTF-8 through its pipes and console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$logDir = "logs"
$masterLog = "$logDir\master_eval.log"

# Create logs directory if it doesn't exist
if (-Not (Test-Path -Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

# Initialize master log
$header = @"
======================================================
Starting Evaluation Pipeline with n=$n
Date: $(Get-Date)
======================================================
"@
$header | Out-File -FilePath $masterLog -Encoding utf8

# Exact order of datasets (updated to match your Bash script)
$datasets = @("tqa", "2wiki", "hotpotqa", "nq")
$total = $datasets.Count

for ($i = 0; $i -lt $total; $i++) {
    $dataset = $datasets[$i]
    $scriptLog = "$logDir\${dataset}.log"
    $moduleName = "src.eval.eval_$dataset"
    
    $startMsg = "`n------------------------------------------------------`nRunning evaluation for: $dataset`nCommand: python -m $moduleName -n $n"
    Write-Host $startMsg -ForegroundColor Cyan
    $startMsg | Out-File -FilePath $masterLog -Append -Encoding utf8
    
    try {
        # 4. Force cmd.exe to use UTF-8 (chcp 65001) before running Python
        cmd.exe /c "chcp 65001 >NUL && python -m $moduleName -n $n 2>&1" | Tee-Object -FilePath $scriptLog | Tee-Object -FilePath $masterLog -Append
        
        # Check if the python executable threw an error code
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: $dataset evaluation failed with exit code $LASTEXITCODE." -ForegroundColor Red
            Write-Host "Moving gracefully to the next dataset..." -ForegroundColor Yellow
            "ERROR: $dataset evaluation failed with exit code $LASTEXITCODE." | Out-File -FilePath $masterLog -Append -Encoding utf8
        } else {
            Write-Host "SUCCESS: $dataset evaluation completed." -ForegroundColor Green
            "SUCCESS: $dataset evaluation completed." | Out-File -FilePath $masterLog -Append -Encoding utf8
        }
    } catch {
        Write-Host "CRITICAL ERROR: Could not launch script for $dataset." -ForegroundColor Red
        "CRITICAL ERROR: Could not launch script for $dataset." | Out-File -FilePath $masterLog -Append -Encoding utf8
    }
    
    # Sleep for 2 minutes (120s) to let the GPU rest, unless it's the last dataset
    if ($i -lt ($total - 1)) {
        $sleepMsg = "Sleeping for 2 minutes to let GPU cool down..."
        Write-Host $sleepMsg -ForegroundColor DarkYellow
        $sleepMsg | Out-File -FilePath $masterLog -Append -Encoding utf8
        Start-Sleep -Seconds 120
    }
}

$endMsg = "`n======================================================`nAll evaluations in the pipeline have finished."
Write-Host $endMsg -ForegroundColor Cyan
$endMsg | Out-File -FilePath $masterLog -Append -Encoding utf8