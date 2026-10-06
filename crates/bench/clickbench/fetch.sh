#!/bin/sh
# Downloads ClickBench's first file, a million rows, into DIR (by default
# target/clickbench), where the instruction benchmarks read it, unless it's
# there already, and checks it against hits_0.sha256. It's ClickBench's own
# copy, so nothing of ours is hosted.
set -eu
dir=${1:-target/clickbench}
here=$(dirname "$0")
url=https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet
mkdir -p "$dir"
check() {
  (cd "$dir" && sed "s/hits_0.parquet/$1/" "$OLDPWD/$here/hits_0.sha256" | shasum -a 256 -c - >/dev/null)
}
if [ -f "$dir/hits_0.parquet" ] && check hits_0.parquet; then
  exit 0
fi
curl -sSfL --retry 3 -o "$dir/hits_0.parquet.part" "$url"
check hits_0.parquet.part || { echo "hits_0.parquet doesn't match hits_0.sha256" >&2; exit 1; }
mv "$dir/hits_0.parquet.part" "$dir/hits_0.parquet"
