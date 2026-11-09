#!/bin/zsh
cd ~/weight-space-programming
for n in r10_compress r10_compress_xs bl_drop_alt3 bl_mag_uniform50 c_posfix c_unlate c_revertlate r25_compress_revertlate r25_compress_xs_revertlate; do
  uv run python -m wsp.evaluate --model checkpoints/$n --name $n 2>&1 | grep -E "^$n "
done
