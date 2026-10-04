param(
    [int[]]$Seeds = @(42, 17, 123),
    [string[]]$Pairs = @('mug', 'apple_explicit'),
    [string]$Tag = 'restoration_v1'
)
$ErrorActionPreference = 'Stop'
# Execute from MUTE/. Sequential runs keep the shared GPU load bounded.
$bank = Get-Content -LiteralPath runs\development_bank_v1\bank_manifest.json | ConvertFrom-Json
foreach ($seed in $Seeds) {
    foreach ($name in $Pairs) {
        $pair = $bank.pairs.$name
        if ($null -eq $pair) { throw "Unknown development pair: $name" }
        $output = "runs\${name}_seed${seed}_${Tag}"
        if (Test-Path -LiteralPath $output) { throw "Output already exists: $output" }
        & ..\.venv\Scripts\python.exe -m mute.run --model checkpoints\pixart512 --local-files-only --output $output --source $pair[0] --target $pair[1] --phrase $pair[2] --seed $seed --offload --conditioning-cache "runs\development_bank_v1\${name}.safetensors" --intervention-start 15 --penalty 2 --modes none directional --identity-controls --restore-background --verify-capture
        if ($LASTEXITCODE -ne 0) { throw "Run failed: $name, seed $seed" }
    }
}
