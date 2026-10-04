$ErrorActionPreference = 'Stop'
# Execute from MUTE/. Settings fixed in PROTOCOL_COVERAGE_DEVELOPMENT_2026-10-04.md.
$bank = Get-Content -LiteralPath runs\development_bank_v1\bank_manifest.json | ConvertFrom-Json
# Last setting is the separately documented post-selection development transfer.
$settings = @(@('mug',42,1,0),@('mug',42,2,0),@('mug',42,2,1),@('apple_explicit',42,2,1),@('mug',123,2,1),@('mug',123,2,0))
foreach ($setting in $settings) {
    $name=$setting[0]; $seed=$setting[1]; $margin=$setting[2]; $feather=$setting[3]
    $pair=$bank.pairs.$name
    $output="runs\${name}_seed${seed}_coverage_m${margin}_f${feather}_v1"
    if (Test-Path -LiteralPath $output) { throw "Output already exists: $output" }
    & ..\.venv\Scripts\python.exe -m mute.run --model checkpoints\pixart512 --local-files-only --output $output --source $pair[0] --target $pair[1] --phrase $pair[2] --seed $seed --offload --conditioning-cache "runs\development_bank_v1\${name}.safetensors" --modes none --identity-controls --restore-background --verify-capture --mask-margin $margin --mask-feather $feather
    if ($LASTEXITCODE -ne 0) { throw "Coverage run failed: $output" }
}
