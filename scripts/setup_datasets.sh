#!/usr/bin/env bash
# Download and lay out AVRSI datasets. Safe to rerun: finished steps are skipped and failed ones reported.
#   scripts/setup_datasets.sh <root> [--core | --all] [--only a,b] [--dry-run] [--keep-archives]
# --core (default): vggsound avqa cmm avhbench. --all adds daily_omni videomme valor2 worldsense music_avqa.
# Writes <root>/datasets.env with the paths the runbook uses. Needs python3 and pip; installs the
# downloaders with --user if missing. Set HF_TOKEN if a Hugging Face repo needs it.
# AVQA's annotations sit behind a SharePoint page, so put AVQA.zip in <root>/avqa/ by hand (the script says when).
set -uo pipefail

usage() { sed -n 2,6p "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }
[ $# -ge 1 ] || usage
case "$1" in -h|--help) usage 0;; esac
ROOT=$1; shift
CORE="vggsound avqa cmm avhbench"
EXTRA="daily_omni videomme valor2 worldsense music_avqa"
SETS=$CORE; DRY=0; KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --core) SETS=$CORE ;;
    --all) SETS="$CORE $EXTRA" ;;
    --only) shift; SETS=${1//,/ } ;;
    --dry-run) DRY=1 ;;
    --keep-archives) KEEP=1 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1"; usage ;;
  esac
  shift
done

AVQA_PAGE=https://mn.cs.tsinghua.edu.cn/avqa/
AVHBENCH_VIDEOS=10-Qp8zxA3ITT-ileEnCgJkf5Nzx1wry7
AVHBENCH_QA=1KcYDAv9lLy3hsx5rWdfRqMFV2NYcZ94W
FAILED=(); MANUAL=()
log() { echo "[setup] $*"; }
run() { log "+ $*"; [ "$DRY" = 1 ] || "$@"; }
fail() { log "FAILED: $1"; FAILED+=("$1"); }
manual() { log "MANUAL: $1"; MANUAL+=("$1"); }
done_mark() { [ -e "$ROOT/.done_$1" ]; }
mark() { [ "$DRY" = 1 ] || touch "$ROOT/.done_$1"; }

need_tools() {
  [ "$DRY" = 1 ] && return 0
  export PATH="$HOME/.local/bin:$PATH"
  command -v hf >/dev/null || python3 -m pip install --user -q -U 'huggingface_hub[cli]' || return 1
  command -v gdown >/dev/null || python3 -m pip install --user -q gdown || return 1
  command -v hf >/dev/null && command -v gdown >/dev/null
}

need_space() {  # dir, GB
  [ "$DRY" = 1 ] && return 0
  local free; free=$(df -BG --output=avail "$1" | tail -1 | tr -dc 0-9)
  [ "$free" -ge "$2" ] || { log "need ${2} GB in $1, have ${free} GB"; return 1; }
}

hf_get() { run hf download "$1" --repo-type dataset --local-dir "$2" "${@:3}"; }

has_video() {  # file: true if it holds a video stream
  python3 - "$1" <<'P'
import sys
try:
    import av
    sys.exit(0 if av.open(sys.argv[1]).streams.video else 1)
except ImportError:
    head = open(sys.argv[1], "rb").read(64)
    sys.exit(0 if b"ftyp" in head else 1)
P
}

setup_vggsound() {
  local d=$ROOT/vggsound; mkdir -p "$d/clips"
  done_mark vggsound && { log "vggsound: already done"; return 0; }
  need_space "$ROOT" 420 || return 1   # 338 GB of archives, fetched and unpacked one at a time
  hf_get Loie/VGGSound "$d/archives" --include vggsound.csv || return 1
  local i a
  for i in $(seq -w 0 19); do
    a=$d/archives/vggsound_$i.tar.gz
    [ -e "$a.extracted" ] && continue
    hf_get Loie/VGGSound "$d/archives" --include "vggsound_$i.tar.gz" || return 1
    [ "$DRY" = 1 ] && continue
    log "extracting $(basename "$a")"
    tar -xzf "$a" -C "$d/clips" || return 1
    touch "$a.extracted"
    [ "$KEEP" = 1 ] || rm -f "$a"
  done
  [ "$DRY" = 1 ] && return 0
  cp -n "$d/archives/vggsound.csv" "$d/" 2>/dev/null
  local sample; sample=$(find "$d/clips" -type f \( -name '*.mp4' -o -name '*.wav' -o -name '*.flac' \) -print -quit)
  [ -n "$sample" ] || { log "vggsound: no media found after extraction"; return 1; }
  if ! has_video "$sample"; then
    log "vggsound: $sample has NO video stream. XM and AVQA need frames; use the official CSV and YouTube instead."
    return 1
  fi
  log "vggsound: $(find "$d/clips" -type f | wc -l) clips, sample $sample"
  mark vggsound
}

setup_avqa() {
  local d=$ROOT/avqa; mkdir -p "$d/videos"
  done_mark avqa && { log "avqa: already done"; return 0; }
  [ "$DRY" = 1 ] && { log "would unpack $d/AVQA.zip and link clips from vggsound"; return 0; }
  if [ ! -e "$d/train_qa.json" ] || [ ! -e "$d/val_qa.json" ]; then
    if [ -e "$d/AVQA.zip" ]; then
      python3 -m zipfile -e "$d/AVQA.zip" "$d/unzipped" || return 1
      find "$d/unzipped" \( -name train_qa.json -o -name val_qa.json \) -exec cp {} "$d/" \;
    fi
  fi
  if [ ! -e "$d/train_qa.json" ]; then
    manual "AVQA: download AVQA.zip from the Downloads section of $AVQA_PAGE into $d/ and rerun"
    return 0
  fi
  python3 - "$d" "$ROOT/vggsound/clips" <<'P' || return 1
import json, os, sys
from pathlib import Path
d, clips = Path(sys.argv[1]), Path(sys.argv[2])
if not clips.is_dir():
    sys.exit("avqa: no vggsound clips; AVQA videos come from VGGSound")
stem = {}
for dirpath, _, files in os.walk(clips):
    for f in files:
        if f.endswith(".mp4"):
            stem[f[:-4]] = os.path.join(dirpath, f)
linked = missing = 0
for split in ("train_qa.json", "val_qa.json"):
    for row in json.loads((d / split).read_text()):
        name = str(row["video_name"])
        dst = d / "videos" / f"{name}.mp4"
        if dst.exists():
            linked += 1
            continue
        src = stem.get(name)
        if src:
            dst.symlink_to(src)
            linked += 1
        else:
            missing += 1
print(f"avqa: {linked} rows have a clip, {missing} do not")
if linked == 0:
    sys.exit("avqa: no video_name matched a clip name; check the naming in avrsi/data/avqa.py")
P
  mark avqa
}

setup_cmm() {
  local d=$ROOT/cmm; mkdir -p "$d"
  done_mark cmm && { log "cmm: already done"; return 0; }
  need_space "$ROOT" 15 || return 1
  hf_get DAMO-NLP-SG/CMM "$d" || return 1
  [ "$DRY" = 1 ] && return 0
  python3 -m zipfile -e "$d/reorg_raw_files.zip" "$d" || return 1
  [ "$KEEP" = 1 ] || rm -f "$d/reorg_raw_files.zip"
  mark cmm
}

setup_avhbench() {
  local d=$ROOT/avhbench; mkdir -p "$d"
  done_mark avhbench && { log "avhbench: already done"; return 0; }
  run gdown "$AVHBENCH_QA" -O "$d/qa_download" || return 1
  run gdown "$AVHBENCH_VIDEOS" -O "$d/videos_download" || return 1
  [ "$DRY" = 1 ] && return 0
  local f
  for f in qa_download videos_download; do
    if python3 -m zipfile -t "$d/$f" >/dev/null 2>&1; then
      python3 -m zipfile -e "$d/$f" "$d/" || return 1
    elif tar -tf "$d/$f" >/dev/null 2>&1; then
      tar -xf "$d/$f" -C "$d" || return 1
    fi
  done
  log "avhbench: unpacked. Point --qa-json at the QA json in $d"
  mark avhbench
}

setup_hf_only() {  # name, repo, GB
  local d=$ROOT/$1; mkdir -p "$d"
  done_mark "$1" && { log "$1: already done"; return 0; }
  need_space "$ROOT" "$3" || return 1
  hf_get "$2" "$d" || return 1
  mark "$1"
}

mkdir -p "$ROOT" || exit 1
ROOT=$(cd "$ROOT" && pwd)
need_tools || { log "could not install hf and gdown"; exit 1; }
log "root $ROOT, datasets: $SETS"
for s in $SETS; do
  log "=== $s"
  case "$s" in
    vggsound) setup_vggsound || fail vggsound ;;
    avqa) setup_avqa || fail avqa ;;
    cmm) setup_cmm || fail cmm ;;
    avhbench) setup_avhbench || fail avhbench ;;
    daily_omni) setup_hf_only daily_omni liarliar/Daily-Omni 100 || fail daily_omni ;;
    videomme) setup_hf_only videomme lmms-lab/Video-MME 300 || fail videomme ;;
    valor2) setup_hf_only valor2 inesriahi/valor32k-avqa-v2 100 || fail valor2 ;;
    worldsense) setup_hf_only worldsense honglyhly/WorldSense 100 || fail worldsense
                manual "WorldSense: the card points to a separate download page; check $ROOT/worldsense/README.md" ;;
    music_avqa) manual "MUSIC-AVQA: videos are on Google Drive and Baidu; see https://gewu-lab.github.io/MUSIC-AVQA/" ;;
    *) fail "$s (unknown dataset)" ;;
  esac
done

if [ "$DRY" = 0 ]; then
  cat > "$ROOT/datasets.env" <<V
export AVRSI_DATA=$ROOT
export AVQA_ROOT=$ROOT/avqa
export VGGSOUND_CLIPS=$ROOT/vggsound/clips
export VGGSOUND_CSV=$ROOT/vggsound/vggsound.csv
export CMM_ROOT=$ROOT/cmm
export AVHBENCH_ROOT=$ROOT/avhbench
V
  log "wrote $ROOT/datasets.env"
fi
echo
log "manual steps left: ${#MANUAL[@]}"; for m in "${MANUAL[@]:-}"; do [ -n "$m" ] && log "  $m"; done
log "failed: ${#FAILED[@]}"; for f in "${FAILED[@]:-}"; do [ -n "$f" ] && log "  $f"; done
[ ${#FAILED[@]} -eq 0 ]
