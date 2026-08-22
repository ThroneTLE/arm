# Auto-attach Orbbec camera (2bc5) to WSL. busid may change after replug.

$devs = usbipd list | Select-String "2bc5"
if (-not $devs) {
    Write-Host "No Orbbec camera (2bc5) found. Check the USB cable."
    exit 1
}

foreach ($d in $devs) {
    $p = $d.Line -split '\s{2,}'
    $busid = $p[0].Trim()
    $state = $p[-1].Trim()
    Write-Host "Device: $busid  state: $state"

    if ($state -match "Attached") {
        Write-Host "  already attached, skip"
        continue
    }
    if ($state -match "Not shared") {
        Write-Host "  not shared. Run as admin: usbipd bind --busid $busid"
        continue
    }

    Write-Host "  attaching..."
    usbipd attach --wsl --busid $busid
    Write-Host "  attach exit code: $LASTEXITCODE"
}

Write-Host ""
Write-Host "--- usbipd list ---"
usbipd list
