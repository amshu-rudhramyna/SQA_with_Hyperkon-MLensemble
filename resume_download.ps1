$DatasetPath = "c:\Users\NEW\Desktop\hypersoilnet2\data\raw\HYPERVIEW2"
$LogFile = "c:\Users\NEW\Desktop\hypersoilnet2\resume_download.log"

$Date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $LogFile -Value "[$Date] Checking HYPERVIEW2 download status..."

$TargetAssetCount = 9409

if (Test-Path $DatasetPath) {
    $CurrentCount = (Get-ChildItem -Path $DatasetPath -Recurse -File).Count
    if ($CurrentCount -lt $TargetAssetCount) {
        Add-Content -Path $LogFile -Value "[$Date] Found $CurrentCount files. Resuming download..."
        Set-Location "c:\Users\NEW\Desktop\hypersoilnet2"
        $env:PYTHONUTF8="1"
        eotdl datasets get HYPERVIEW2 -v 2 -p data/raw/ -f --assets
        Add-Content -Path $LogFile -Value "[$Date] eotdl command executed."
    } else {
        Add-Content -Path $LogFile -Value "[$Date] Download appears complete ($CurrentCount files)."
    }
} else {
    Add-Content -Path $LogFile -Value "[$Date] Dataset directory not found. Starting download..."
    Set-Location "c:\Users\NEW\Desktop\hypersoilnet2"
    $env:PYTHONUTF8="1"
    eotdl datasets get HYPERVIEW2 -v 2 -p data/raw/ -f --assets
    Add-Content -Path $LogFile -Value "[$Date] eotdl command executed."
}
