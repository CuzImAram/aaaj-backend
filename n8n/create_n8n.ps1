# Ensure Docker CLI exists, create data dir, and create the stopped container named "n8n".

# Configuration (adjust only if needed)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataDir = Join-Path $ScriptDir 'data'
$ContainerName = "aaaj-backend-n8n"
$Image = "n8nio/n8n:latest"
$HostPort = 5678
$ContainerPort = 5678

# Check docker CLI
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker CLI not found. Install Docker Desktop and ensure 'docker' is in PATH."
    exit 1
}

# Ensure data directory exists
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir | Out-Null
}

# Check if container already exists (exact name match)
$exists = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}"
if ($exists -match "^$ContainerName$") {
    Write-Host "Container '$ContainerName' already exists. Use 'docker start $ContainerName' to start it."
    exit 0
}

# Create the container (stopped). It will persist data to $DataDir.
# Use the full path for the bind mount so Docker on Windows can resolve it.
$HostDataPath = (Resolve-Path -LiteralPath $DataDir).ProviderPath

docker create `
    --name $ContainerName `
    -p "${HostPort}:${ContainerPort}" `
    -v "${HostDataPath}:/home/node/.n8n" `
    -e N8N_HOST=localhost `
    -e WEBHOOK_URL=http://localhost:$HostPort `
    -e NODE_ENV=production `
    $Image | Out-Null

Write-Host "Created container '$ContainerName'. Start it with: docker start $ContainerName"
Write-Host "Then open: http://localhost:$HostPort"
