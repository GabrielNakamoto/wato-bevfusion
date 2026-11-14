#!/bin/bash
# Stage the minimal nuScenes v1.0-trainval set for BEVFusion dual-head training:
#   - 10 keyframe parts (~44.5 GB compressed): camera keyframes
#   - 10 blobs_lidar parts (~135 GB compressed): LiDAR keyframe + sweep bins
#   - metadata (~0.46 GB) and map expansion (~0.40 GB)
# Total ~180 GB compressed / ~185 GB extracted. Camera sweeps are not needed.
#
# Resumable per part; deletes each tarball after a successful extraction.
# Usage:  ./stage_nuscenes.sh            (concurrency 3)
#         JOBS=4 ./stage_nuscenes.sh
set -uo pipefail

BASE="${BASE:-https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/public/v1.0}"
ROOT="${ROOT:-/users/gnakamot/wato/data/nuscenes}"
STAGE="${STAGE:-/users/gnakamot/wato/data/_stage}"
JOBS="${JOBS:-3}"

mkdir -p "$ROOT" "$STAGE"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

fetch() {
  local url="$1" out="$2"
  for attempt in $(seq 1 30); do
    if curl -sSL --fail --connect-timeout 30 --max-time 14400 -C - -o "$out" "$url"; then
      return 0
    fi
    log "retry $attempt: $(basename "$out") ($(stat -c%s "$out" 2>/dev/null || echo 0) bytes)"
    sleep 15
  done
  return 1
}

extract() {
  local tgz="$1"
  tar --use-compress-program='pigz -p 8' -xf "$tgz" -C "$ROOT"
}

process() {
  local name="$1" url="$2"
  local tgz="$STAGE/$name.tgz" done="$STAGE/$name.done"
  [ -f "$done" ] && { log "skip $name"; return 0; }
  log "start $name"
  fetch "$url" "$tgz" || { log "FAIL download $name"; return 1; }
  log "extract $name ($(du -h "$tgz" | cut -f1))"
  extract "$tgz" || { log "FAIL extract $name"; return 1; }
  rm -f "$tgz"
  touch "$done"
  log "done $name"
}

maps() {
  local name="nuScenes-map-expansion-v1.3"
  local zip="$STAGE/$name.zip" done="$STAGE/$name.done"
  [ -f "$done" ] && { log "skip $name"; return 0; }
  log "start $name"
  fetch "$BASE/$name.zip" "$zip" || { log "FAIL download $name"; return 1; }
  mkdir -p "$ROOT/maps"
  ( cd "$ROOT" && python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('.')" "$zip" ) \
    || { log "FAIL extract $name"; return 1; }
  mkdir -p "$ROOT/maps/expansion"
  if [ -d "$ROOT/expansion" ]; then
    mv "$ROOT/expansion/"* "$ROOT/maps/expansion/" && rmdir "$ROOT/expansion"
  fi
  rm -f "$zip"
  touch "$done"
  log "done $name"
}

# self-invocation worker for xargs
if [ "${1:-}" = "--one" ]; then
  process "$2" "$3"
  exit $?
fi

log "staging into $ROOT (concurrency $JOBS)"

process v1.0-trainval_meta "$BASE/v1.0-trainval_meta.tgz" || exit 1
maps || exit 1

for i in $(seq -w 1 10); do
  printf 'v1.0-trainval%s_keyframes %s/v1.0-trainval%s_keyframes.tgz\n' "$i" "$BASE" "$i"
  printf 'v1.0-trainval%s_blobs_lidar %s/v1.0-trainval%s_blobs_lidar.tgz\n' "$i" "$BASE" "$i"
done | xargs -P "$JOBS" -n 2 "$0" --one

rc=$?
log "all parts processed (rc=$rc)"
du -sh "$ROOT" 2>/dev/null
find "$ROOT/samples/LIDAR_TOP" -name '*.pcd.bin' 2>/dev/null | wc -l
find "$ROOT/samples/CAM_FRONT" -name '*.jpg' 2>/dev/null | wc -l
exit "$rc"
