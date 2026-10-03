#!/bin/sh
# Sums section sizes rather than file size, which includes page padding.
# .relro_padding is page padding too, as a section.
set -eu
target=$1

cargo build -q --release --target "$target" -p pipitdb-size
case $target in
  wasm32-*) file=pipitdb_size.wasm ;;
  *-apple-*) file=libpipitdb_size.dylib ;;
  *) file=libpipitdb_size.so ;;
esac

llvm_size=$(find "$(rustc --print sysroot)" -name llvm-size -type f | head -n 1)
sections=$("$llvm_size" -A "target/$target/release/$file")
size=$(echo "$sections" | awk 'NR > 2 && $1 != "Total" && $1 != ".relro_padding" { s += $2 } END { print s }')
budget=$(awk -v target="$target" '$1 == target { print $2 }' crates/size/budgets)

echo "$target: $size bytes (budget: $budget)"
if [ "$size" -gt "$budget" ]; then
  echo "$sections"
  exit 1
fi
