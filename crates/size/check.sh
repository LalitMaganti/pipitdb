#!/bin/sh
# Sums section sizes rather than file size, which includes page padding.
set -eu
target=$1

cargo build -q --release --target "$target" -p pipitdb-size
case $target in
  wasm32-*) file=pipitdb_size.wasm ;;
  *-apple-*) file=libpipitdb_size.dylib ;;
  *) file=libpipitdb_size.so ;;
esac

llvm_size=$(find "$(rustc --print sysroot)" -name llvm-size -type f | head -n 1)
size=$("$llvm_size" -A "target/$target/release/$file" | awk '$1 == "Total" { print $2 }')
budget=$(awk -v target="$target" '$1 == target { print $2 }' crates/size/budgets)

echo "$target: $size bytes (budget: $budget)"
[ "$size" -le "$budget" ]
