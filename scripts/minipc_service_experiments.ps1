# Service experiments on the mini PC (the quiet machine): exp5, exp9 latency, M6(b) flexdates, exp4.
# Run as a minipc job from the mirrored project dir. Windows PowerShell 5.1.
$ErrorActionPreference = "Continue"
$env:JAVA_HOME = "C:\SullaPortal\projects\staysearch\jdk\jdk-21.0.12.1+1"
Set-Location service
.\gradlew.bat installDist --no-daemon "-Dorg.gradle.workers.max=2" -x test 2>&1 | Select-Object -Last 3
# Gradle's .bat launcher lists every jar and overflows cmd's line limit; call java with a wildcard.
function Svc { & "$env:JAVA_HOME\bin\java.exe" -Xmx6g --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector -cp "build\install\staysearch-service\lib\*" staysearch.cli.MainKt @args }
$bundle = "..\artifacts\bundle"
$idx = "C:\SullaPortal\projects\staysearch\indexes"
New-Item -ItemType Directory -Force -Path $idx | Out-Null
Write-Output "== exp5 $(Get-Date -Format s)"
Svc exp5 --bundle $bundle --out ..\results\m3_exp5.jsonl 2>&1 | Select-Object -Last 5
Write-Output "== latency $(Get-Date -Format s)"
Svc latency --bundle $bundle --out ..\results\m3_latency.jsonl --rates 25,50,100,200 --duration 30 --warmup 15 --updates false,true --update-rate 1000 --mode ivf --ranker dnn 2>&1 | Select-Object -Last 5
Write-Output "== flexdates $(Get-Date -Format s)"
Svc flexdates --bundle $bundle --out ..\results\m6_flexdates.jsonl --n 2000 --mode ivf 2>&1 | Select-Object -Last 5
Write-Output "== exp4 $(Get-Date -Format s)"
Svc exp4 --bundle $bundle --out ..\results\m4_exp4.jsonl --modes ivf,hnsw --rates 0,100,1000,10000 --duration 60 --qps 20 --repeats 3 --index-root $idx 2>&1 | Select-Object -Last 10
Remove-Item -Recurse -Force $idx -ErrorAction SilentlyContinue
Write-Output "== done $(Get-Date -Format s)"
