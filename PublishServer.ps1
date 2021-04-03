# Compress published server binaries
$compress = @{
    Path = ".\SmartPhoto\SmartPhoto\publish\*"
    CompressionLevel = "Optimal"
    DestinationPath = ".\Server.zip"
} 
if (Test-Path "Server.zip") {
    Remove-Item "Server.zip"
}
Compress-Archive @compress

# Download additional release assets we need to keep
# Invoke-WebRequest https://github.com/NirooshKa/APS360-Cold-Start-Problem/releases/download/V0.2/Walkthrough.avi -OutFile .\Walkthrough.avi

# Release URL: https://api.github.com/repos/NirooshKa/APS360-Cold-Start-Problem/releases/40588262/assets
# Use `& hub --% api -t repos/NirooshKa/APS360-Cold-Start-Problem/releases/40588262/assets` to view asset url

# Get URL of existing Server.zip asset
$assets = & hub --% api repos/NirooshKa/APS360-Cold-Start-Problem/releases/40588262/assets | ConvertFrom-Json
$assetURL = $assets.Where({$_.name -eq "Server.zip"}).url

# Update Github release
$p1 = 'api'; $p2 = '-X'; $p3 = 'DELETE'; $p4 = $assetURL;
& hub $p1 $p2 $p3 $p4
& hub --% release edit -m "" -a .\Server.zip V0.2