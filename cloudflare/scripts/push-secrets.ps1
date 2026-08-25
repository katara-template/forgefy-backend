<#
.SYNOPSIS
    Uploads the app's .env to Cloudflare as Worker secrets, under the same names.

.DESCRIPTION
    Reads ../../.env (the same file docker-compose.yml feeds to the services),
    strips the keys that wrangler.json already declares as plaintext `vars`,
    and hands the rest to `wrangler secret bulk`. The Worker then passes every
    one of them through to both containers unchanged — see src/container-env.ts.

    Lines are passed to wrangler verbatim rather than re-parsed here, so quoting
    and escaping behave the same as they do for Docker Compose.

    The .env file is never written anywhere outside a temp file that is deleted
    on the way out, and nothing is committed.

.PARAMETER EnvFile
    Path to the source .env. Defaults to the repo root .env.

.PARAMETER DryRun
    Print the keys that would be uploaded, then stop.

.EXAMPLE
    ./scripts/push-secrets.ps1 -DryRun
    ./scripts/push-secrets.ps1
    ./scripts/push-secrets.ps1 -EnvFile ../.env.production
#>

[CmdletBinding()]
param(
    [string] $EnvFile = (Join-Path $PSScriptRoot '..\..\.env'),
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'

# Declared as plaintext `vars` in wrangler.json. Uploading them as secrets too
# would give the Worker two conflicting sources for the same name.
$DeclaredAsVars = @('APP_ENV', 'PORT', 'WEB_CONCURRENCY', 'PYTHONUNBUFFERED')

if (-not (Test-Path $EnvFile)) {
    throw "Env file not found: $EnvFile"
}

$configPath = Join-Path $PSScriptRoot '..\wrangler.json'

# Keep only real KEY=VALUE lines; drop comments, blanks, and the vars above.
# Duplicate keys resolve to the LAST occurrence, matching how a shell and
# docker-compose's env_file behave — wrangler would treat a repeated key in the
# upload as ambiguous.
$byKey = [ordered]@{}
$skipped = @()
$duplicates = @()

foreach ($line in Get-Content -LiteralPath $EnvFile -Encoding utf8) {
    if ($line -notmatch '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') { continue }

    $key = $Matches[1]
    if ($DeclaredAsVars -contains $key) {
        $skipped += $key
        continue
    }
    if ($byKey.Contains($key)) {
        $duplicates += $key
    }
    $byKey[$key] = $line
}

$kept = $byKey.Keys | ForEach-Object {
    [pscustomobject]@{ Key = $_; Line = $byKey[$_] }
}

if ($kept.Count -eq 0) {
    throw "No uploadable KEY=VALUE lines found in $EnvFile"
}

# wrangler accepts at most 100 secrets per bulk request.
if ($kept.Count -gt 100) {
    throw "$($kept.Count) secrets exceeds wrangler's limit of 100 per bulk request. Split the file."
}

Write-Host "Source:  $EnvFile"
Write-Host "Config:  $configPath"
Write-Host "Upload:  $($kept.Count) secret(s)"
if ($skipped.Count -gt 0) {
    Write-Host "Skipped: $(($skipped | Sort-Object -Unique) -join ', ') (declared as vars in wrangler.json)"
}
if ($duplicates.Count -gt 0) {
    Write-Warning "Duplicate key(s) in ${EnvFile}: $(($duplicates | Sort-Object -Unique) -join ', '). Using the last occurrence of each."
}
Write-Host ''
$kept.Key | Sort-Object | ForEach-Object { Write-Host "  $_" }

if ($DryRun) {
    Write-Host ''
    Write-Host 'Dry run - nothing uploaded.'
    return
}

# A CLOUDFLARE_API_TOKEN exported in THIS shell is what wrangler authenticates
# with. The app has its own CLOUDFLARE_API_TOKEN in .env for a different
# purpose; reading the file does not export it, but a stale shell value will
# silently send this upload to the wrong account.
if ($env:CLOUDFLARE_API_TOKEN) {
    Write-Warning 'CLOUDFLARE_API_TOKEN is set in this shell — wrangler will authenticate with it, not with `wrangler login`.'
}

$tempFile = Join-Path ([System.IO.Path]::GetTempPath()) ("forgefy-secrets-{0}.env" -f ([guid]::NewGuid()))

try {
    Set-Content -LiteralPath $tempFile -Value $kept.Line -Encoding utf8
    Write-Host ''
    Write-Host 'Uploading…'
    & npx wrangler secret bulk $tempFile --config $configPath
    if ($LASTEXITCODE -ne 0) {
        throw "wrangler secret bulk failed with exit code $LASTEXITCODE"
    }
    Write-Host ''
    Write-Host 'Done. Verify with: npx wrangler secret list --config ../wrangler.json'
}
finally {
    if (Test-Path -LiteralPath $tempFile) {
        Remove-Item -LiteralPath $tempFile -Force
    }
}
