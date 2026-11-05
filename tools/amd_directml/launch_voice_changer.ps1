# Starts VCClient v1.5.3 and re-applies the settings it forgets on restart.
#
# The app saves server-audio/crossfade/threshold settings but never re-applies
# them at startup, and protect is reset every launch. This script starts the
# server (or attaches to a running one), raises its priority, and pushes the
# settings below through the REST API. Put it next to MMVCServerSIO.exe.
#
# Edit the settings here.
$Port = 18888
$Settings = [ordered]@{
    silentThreshold      = 0.001   # gate; the gate-hold patch keeps word endings
    protect              = 0.33    # keeps breaths/consonants from going robotic
    extraConvertSize     = 16384   # context per chunk; 32768 was worse in practice
    crossFadeOverlapSize = 8192    # smoother joins between chunks
    serverReadChunkSize  = 192     # 512 ms; use 128 for less delay if your PC keeps up
}
# Server audio (audio handled by the server itself, not the app window).
$UseServerAudio = $true
$InputName   = "HyperX QuadCast S"      # mic
$OutputName  = "CABLE Input"            # what Discord hears (CABLE Output)
$MonitorName = "Speakers (Realtek"      # where you hear yourself; "" for none
$HostApi     = "Windows WASAPI"
$SampleRate  = 48000

# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$base = "http://127.0.0.1:$Port"
$curl = "$env:SystemRoot\System32\curl.exe"

function Set-VC($key, $val) {
    & $curl -s -o NUL -F "key=$key" -F "val=$val" "$base/update_settings"
}
function Get-Info { Invoke-RestMethod "$base/info" -TimeoutSec 5 }

$running = Get-Process MMVCServerSIO -ErrorAction SilentlyContinue
if (-not $running) {
    Write-Host "Starting voice changer..."
    $argList = @(
        "-p", "$Port", "--https", "false",
        "--content_vec_500", "pretrain/checkpoint_best_legacy_500.pt",
        "--content_vec_500_onnx", "pretrain/content_vec_500.onnx",
        "--content_vec_500_onnx_on", "true",
        "--hubert_base", "pretrain/hubert_base.pt",
        "--hubert_base_jp", "pretrain/rinna_hubert_base_jp.pt",
        "--hubert_soft", "pretrain/hubert/hubert-soft-0d54a1f4.pt",
        "--nsf_hifigan", "pretrain/nsf_hifigan/model",
        "--crepe_onnx_full", "pretrain/crepe_onnx_full.onnx",
        "--crepe_onnx_tiny", "pretrain/crepe_onnx_tiny.onnx",
        "--rmvpe", "pretrain/rmvpe.pt",
        "--model_dir", "model_dir",
        "--samples", "samples.json"
    )
    Start-Process -FilePath (Join-Path $here "MMVCServerSIO.exe") -ArgumentList $argList -WorkingDirectory $here
}

Write-Host "Waiting for it to be ready..."
$info = $null
for ($i = 0; $i -lt 120; $i++) {
    try { $info = Get-Info; if ($info.pipelineInfo) { break } } catch {}
    Start-Sleep -Seconds 1
}
if (-not $info) { Write-Host "Voice changer did not start."; Start-Sleep 10; exit 1 }

Get-Process MMVCServerSIO -ErrorAction SilentlyContinue | ForEach-Object {
    try { $_.PriorityClass = "High" } catch {}
}

foreach ($k in $Settings.Keys) { Set-VC $k $Settings[$k] }

if ($UseServerAudio) {
    function Find-Dev($list, $name) {
        if (-not $name) { return -1 }
        $d = $list | Where-Object { $_.hostAPI -eq $HostApi -and $_.name -like "*$name*" } | Select-Object -First 1
        if ($d) { return [int]$d.index } else { return -1 }
    }
    $in  = Find-Dev $info.serverAudioInputDevices  $InputName
    $out = Find-Dev $info.serverAudioOutputDevices $OutputName
    $mon = Find-Dev $info.serverAudioOutputDevices $MonitorName
    if ($in -lt 0 -or $out -lt 0) {
        Write-Host "Could not find mic '$InputName' or output '$OutputName' - leaving audio mode as is."
    } else {
        Set-VC serverAudioSampleRate $SampleRate
        Set-VC serverInputAudioGain 1.0
        Set-VC serverOutputAudioGain 1.0
        Set-VC serverMonitorAudioGain 1.0
        Set-VC serverInputDeviceId $in
        Set-VC serverOutputDeviceId $out
        Set-VC serverMonitorDeviceId $mon
        Set-VC enableServerAudio 1
        Set-VC serverAudioStated 1
        Write-Host "Server audio on: mic #$in -> CABLE #$out, monitor #$mon"
    }
}

$e = (Get-Info).pipelineInfo.embedder.devType
Write-Host "Ready. Embedder running on: $e"
Start-Sleep -Seconds 3
