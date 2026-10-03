# Second pass on the mini PC: latency (exp9) and the index-under-updates experiment (exp4).
# Comma lists are quoted: unquoted, PowerShell passes them as separate arguments (BUG_LOG #8).
$ErrorActionPreference = "Continue"
$env:JAVA_HOME = "C:\SullaPortal\projects\staysearch\jdk\jdk-21.0.12.1+1"
Set-Location service
.\gradlew.bat installDist --no-daemon "-Dorg.gradle.workers.max=2" -x test 2>&1 | Select-Object -Last 2
function Svc { & "$env:JAVA_HOME\bin\java.exe" -Xmx6g --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector -cp "build\install\staysearch-service\lib\*" staysearch.cli.MainKt @args }
$bundle = "..\artifacts\bundle"
$idx = "C:\SullaPortal\projects\staysearch\indexes"
New-Item -ItemType Directory -Force -Path $idx | Out-Null
Write-Output "== latency $(Get-Date -Format s)"
Svc latency --bundle $bundle --out ..\results\m3_latency.jsonl --rates '25,50,100,200' --duration 30 --warmup 15 --updates 'false,true' --update-rate 1000 --mode ivf --ranker dnn --k 100 2>&1 | Where-Object { $_ -notmatch "WARNING" } | Select-Object -Last 5
Write-Output "== exp4 $(Get-Date -Format s)"
Svc exp4 --bundle $bundle --out ..\results\m4_exp4.jsonl --modes 'ivf,hnsw' --rates '0,100,1000,10000' --duration 60 --qps 20 --repeats 3 --k 100 --index-root $idx 2>&1 | Where-Object { $_ -notmatch "WARNING" } | Select-Object -Last 10
Remove-Item -Recurse -Force $idx -ErrorAction SilentlyContinue
Write-Output "== done $(Get-Date -Format s)"
