#!/usr/bin/env zsh
# Re-captures every BENCH_PAPER.md chart at one or more sizing ratios, one pass per ratio.
# usage: zsh run_ratio_campaign.zsh [ratio ...]   -- default: 1.01 1.02; `1.001` re-baselines in-session.
# Captures land in caps/r<R>/ under the SAME basenames as the archived caps/*.txt; caps/ root is never
# written. Two exceptions: chart 7's q ladder is per-ratio now (see the arrays below), so an r=1.01 pass
# writes variant-*-q<q>.txt under its own solved q values — new basenames beside caps/r1.01/'s archived
# old-ladder files, which stay untouched; and the hbarray step's shapes-sat99-hbarray.txt is a capture
# no archived pass has, the byte-keyed hashbrown arm (BENCH_PAPER's charts 1-4 section).
# SMOKE=1 zsh run_ratio_campaign.zsh <ratio>   -- miniature end-to-end validation into /tmp, never caps/.
#
# Two knobs for a partial re-capture. When a harness fix invalidates some captures and not others
# (BENCH_PAPER's hashbrown-arm section), what has to run is a subset of the steps into a fresh dir:
#   STEPS=shapes,hbarray,hbref  -- run only those steps; unset = all of them. The names are
#     shapes hbarray lfsweep hbref parallel chart7 cachedemo, and the campaign's order never
#     changes with the selection: a subset is the full pass with cells left out, never re-ordered.
#   TAG=hbfix  -- write to caps/r<R>-<TAG>/ instead of caps/r<R>/. No spelling of TAG reaches caps/
#     or a bare caps/r<R>/, so a partial pass cannot land on an archived one.
#   SHAPES_BS=8,16  -- run the shapes step once per bucket size (8, 16 or 32; a rebuild apiece).
#     Unset is the compiled-in default alone and is byte-identical to the archived passes; size 8
#     keeps shapes-sat99-{filtered,plain}.txt, the others get -s$S. Chart 5 sweeps 8/16/32 already.
# A target directory that already holds captures is refused before the first cell runs; RESUME=1
# keeps its finished cells and runs the rest, FORCE=1 re-runs every selected cell over them.
emulate -L zsh
setopt pipe_fail
set -u

cd "${0:a:h}" || exit 1
[[ -f BENCH_PAPER.md && -f Cargo.toml ]] || { print -u2 "run_ratio_campaign: run it from the repo root"; exit 1; }

# One flag set for every capture, passed per invocation, never an exported RUSTFLAGS (BENCH_PAPER ground rule).
FLAGS='build.rustflags=["-C","llvm-args=-align-all-functions=6","-C","target-cpu=native"]'

# The hbarray step's criterion filter, the same in every mode because it is the step's definition
# rather than a size knob: the four 4/4 group names with array mode's `_hbarray` suffix welded on.
# `_hbarray/` has to follow the operation immediately, so the other twelve groups cannot match --
# `insert_kv16_hbarray/` and its kind carry the width tag in between.
HB_ARRAY_FILTER='^(insert|get_existing|get_nonexisting|get_mixed)_hbarray/'

RATIOS=("$@")
(( ${#RATIOS} )) || RATIOS=(1.01 1.02)
for R in $RATIOS; do
  [[ $R =~ '^[0-9]+(\.[0-9]+)?$' ]] || { print -u2 "run_ratio_campaign: ratio '$R' is not a plain decimal"; exit 1; }
done

# Which steps run. Unset means all of them: the campaign as it has always been, plus the hbarray
# cell. The loop below keeps its order either way, so a selection only ever removes cells.
ALL_STEPS=(shapes hbarray lfsweep hbref parallel chart7 cachedemo)
typeset -i steps_given=0
if [[ -n "${STEPS+set}" ]]; then   # set-but-empty is a mistake, not a request for all 20 h
  steps_given=1
  SEL=(${(s:,:)${STEPS//[[:space:]]/,}})   # comma- or space-separated, both spellings accepted
  SEL=(${SEL:#})                           # doubled separators leave empty words behind
  (( ${#SEL} )) || { print -u2 "run_ratio_campaign: STEPS is set but names no step"; exit 1; }
  for ST in $SEL; do
    (( ${ALL_STEPS[(Ie)$ST]} )) || { print -u2 "run_ratio_campaign: unknown step '$ST' (pick from ${(j:, :)ALL_STEPS})"; exit 1; }
  done
  SEL=(${ALL_STEPS:*SEL})   # back into campaign order, duplicates dropped
else
  SEL=($ALL_STEPS)
fi

# TAG suffixes the per-ratio directory instead of replacing it: caps/r1.01-hbfix/, never caps/r1.01/
# and never caps/ root. The suffix is structural, so no value of TAG can reach an archived pass.
TAG=${TAG:-}
[[ -z $TAG || $TAG =~ '^[A-Za-z0-9][A-Za-z0-9._-]*$' ]] ||
  { print -u2 "run_ratio_campaign: TAG '$TAG' must be a plain [A-Za-z0-9._-] token"; exit 1; }

# Which bucket sizes charts 1-4 run at. Unset puts nothing in the cell's environment at all, so the
# crate's compiled default (8) stands and the pass is byte-identical to every archived one. Listed,
# the shapes step runs a filtered+plain pair per size with BUCKET_SIZE=$S, each size a rebuild the
# way the plain/filtered flip already is; size 8 keeps the archive's basenames and the others take
# -s$S beside them. Chart 5 has always swept 8/16/32; this is the same sweep for charts 1-4.
typeset -i bs_given=0
BSIZES=(8)
if [[ -n "${SHAPES_BS+set}" ]]; then   # set-but-empty is a mistake, not a request for the default
  bs_given=1
  BSIZES=(${(s:,:)${SHAPES_BS//[[:space:]]/,}})   # comma- or space-separated, as STEPS is
  BSIZES=(${BSIZES:#})
  (( ${#BSIZES} )) || { print -u2 "run_ratio_campaign: SHAPES_BS is set but names no bucket size"; exit 1; }
  for BS in $BSIZES; do
    [[ $BS == (8|16|32) ]] ||
      { print -u2 "run_ratio_campaign: SHAPES_BS: '$BS' is not one of 8, 16, 32 (the filtered table caps at 32)"; exit 1; }
  done
  BSIZES=(${(uon)BSIZES})   # ascending and deduplicated: a repeated size is one cell, run twice
fi

# Every iterated list below is a zsh ARRAY: an unquoted scalar does not word-split in zsh,
# and the string form already cost one campaign (see BENCH_PAPER's chart-7 note).
if [[ "${SMOKE:-0}" == 1 ]]; then
  CAPROOT=/tmp/ratio-campaign-smoke.$$
  SHAPE_ANCHORS=12,14 LF_ANCHORS=12,14
  LFS_FILTERED=0.47,0.77 LFS_PLAIN=0.86,0.90
  GOOD_1001=(8517) BAD_1001=(16058) GOOD_101=(8673) BAD_101=(16362) KS=(16 17)
  # The explicit anchor list mirrors the full run's: the parallel bench takes its
  # anchors from the campaign now, and smoke has to exercise that hand-off too.
  PAR_ENV=(MT_BENCH_SMOKE=1 MT_BENCH_THREADS=1,2 MT_BENCH_ROUNDS=1 MT_BENCH_TARGET_MS=60 MT_BENCH_ANCHORS=10,11)
  CRIT=(--sample-size 10 --warm-up-time 0.3 --measurement-time 0.5)
  SHAPE_FILTER=('^get_mixed/') SHAPE_FILTER_PLAIN=('^get_mixed_plain/')
else
  CAPROOT=caps
  # Ladders top at 2^28 (~232.5M keys). The sparse sets gain the one anchor; a
  # smoother curve is one edit away (24 and/or 26 into SHAPE_ANCHORS/LF_ANCHORS).
  SHAPE_ANCHORS=12,14,16,18,20,22,28 LF_ANCHORS=16,22,28
  # Same LF ladders as the r=1.001 campaign: the paper_plan feasibility sweep showed the
  # 0.88 / 0.977 tops and every step below them stay clean at 1.01 and 1.02 too.
  # One 2^28 caveat: plain 0.977 degenerates at that anchor at s=8 (realizes ~0.813, every
  # ratio); the step stays for the 2^16/2^22 points, the 2^28 plain curve tops at 0.97.
  LFS_FILTERED=0.47,0.57,0.67,0.72,0.77,0.80,0.82,0.84,0.86,0.88
  LFS_PLAIN=0.86,0.88,0.90,0.92,0.94,0.95,0.96,0.97,0.977
  # Chart 7's q ladders are solved placements, and a solved placement is per-ratio:
  # r=1.01 (the paper's) runs the ladder solved so agile level 0 lands exactly on 2^k
  # at that ratio (BENCH_PAPER's chart-7 section); every other ratio keeps the
  # r=1.001-solved lists the caps/ root and the campaign passes were captured with,
  # so re-runs stay comparable to the archive.
  GOOD_1001=(8517 16893 33612 67000 133530 267059 534118 1068237 2136474 4261030)
  BAD_1001=(16058 31828 63700 126855 253710 507411 1014822 2029652 4050000)
  GOOD_101=(8673 17203 34228 68221 136130 271841 543106 1085417 2169728 4337914)
  BAD_101=(16362 32541 64853 129411 258409 516256 1031737 2062392 4123279)
  KS=(20 21 22 23 24 25 26 27 28)
  PAR_ENV=(MT_BENCH_THREADS=1,2,4,6,12 MT_BENCH_ANCHORS=16,21,24,28)
  CRIT=()
  SHAPE_FILTER=() SHAPE_FILTER_PLAIN=()
fi

typeset -i fails=0 guard_done=0 ratio_bad=0

step_on() { # step_on <name>: selected, and this ratio's pass still trusted?
  (( ratio_bad )) && return 1   # the r guard tripped; every later step of this ratio is skipped
  (( ${SEL[(Ie)$1]} ))
}

run_cell() { # run_cell <basename> <env/cmd ...>: tee into $RDIR, log a failure, keep going.
  local out=$RDIR/$1; shift
  if [[ "${RESUME:-0}" == 1 && -s $out ]]; then
    # RESUME cannot tell a finished cell from one that died after writing output -- read
    # FAILURES.txt before trusting a resumed directory.
    print -r -- "==> $out (RESUME: kept)"
    return 0
  fi
  print -r -- "==> $out"
  if ! "$@" 2>&1 | tee -- "$out"; then
    print -r -- "$(date +%FT%T%z) FAILED: ${out:t}" >> "$RDIR/FAILURES.txt"
    print -u2 -r -- "!! cell failed, campaign continues: $out (see $RDIR/FAILURES.txt)"
    (( fails++ ))
  fi
}

ratio_guard() { # ratio_guard <basename>: the pass's FIRST capture must say r = $R, whichever it is.
  # A capture at the wrong r is worse than none, so the check follows the first cell of whatever
  # step leads the selection -- not a fixed file, which a subset need not have run.
  (( guard_done )) && return 0
  guard_done=1
  grep -q -- "sizing ratio r = $R," "$RDIR/$1" 2>/dev/null && return 0
  print -u2 -r -- "!! $RDIR/$1 does not print 'sizing ratio r = $R' -- skipping ratio $R"
  print -r -- "$(date +%FT%T%z) RATIO GUARD: first capture not at r = $R, ratio pass skipped" >> "$RDIR/FAILURES.txt"
  ratio_bad=1
  (( fails++ ))
  return 1
}

run_variant() { # run_variant <variant tag> [cargo feature args ...]
  local VTAG=$1; shift
  local Q
  for Q in $GOOD $BAD; do
    (( ratio_bad )) && return 1
    run_cell variant-$VTAG-q$Q.txt env MT_PAPER_RATIO=$R MT_BENCH_ANCHORS=10 MT_DEMO_N=$Q MT_DEMO_LF=0.77 \
      cargo bench --config "$FLAGS" --bench comparison "$@" -- "^get_mixed/MultiTable/$Q\$" $CRIT
    ratio_guard variant-$VTAG-q$Q.txt || return 1
  done
}

# A directory that already holds captures is never written into: a pass costs hours, and an archived
# pass is not re-creatable at all on a machine that has moved on. Checked for every ratio up front,
# so a multi-ratio invocation cannot meet the clash half a day in.
if [[ "${FORCE:-0}" != 1 && "${RESUME:-0}" != 1 ]]; then
  for R in $RATIOS; do
    D=$CAPROOT/r$R${TAG:+-$TAG}
    HAVE=($D/*.txt(N))
    (( ${#HAVE} )) || continue
    print -u2 -r -- "run_ratio_campaign: $D already holds ${#HAVE} capture file(s) -- refusing to write into it."
    print -u2 -r -- "  Give TAG a value of its own, or RESUME=1 to keep the finished cells and run the"
    print -u2 -r -- "  rest, or FORCE=1 to re-run every selected cell over them."
    exit 1
  done
fi

print -r -- "=== ratio campaign: ratios (${(j:, :)RATIOS}) -> $CAPROOT/r<R>${TAG:+-$TAG}/ ==="
print -r -- "Keep the sandbox VM (and everything else) idle for the whole run -- it shares the"
print -r -- "silicon and read up to ~25% into chart 8's large-K points once already (BENCH_PAPER)."
print -r -- "Wall-clock estimate per ratio, from the archive's mtimes scaled to the 2^28 tops"
print -r -- "(the big cells' cost is mostly per-cell key-stream setup, not criterion time):"
print -r -- "  charts 1-4 ~3 h (+~50 min for the hbarray arm) | chart 5 ~14 h | chart 6 ~35 min"
print -r -- "  chart 7 ~25 min | chart 8 ~2.5 h"
print -r -- "  plus ~15 min of rebuilds  =>  ~21 h per ratio, ~$(( ${#RATIOS} * 21 )) h for this invocation."
(( ${#BSIZES} > 1 )) &&
  print -r -- "  SHAPES_BS=${(j:,:)BSIZES}: charts 1-4 run a filtered+plain pair per bucket size, ~3 h each."
if (( steps_given )); then
  # What the r=1.01 pass in caps/r1.01/ actually took, step by step, off its own mtimes -- a
  # measurement rather than the projection above, and the only honest basis for a subset's cost.
  # hbarray is the one step that pass never ran, so its 50 min is derived from the same capture
  # rather than measured: the four 4/4 groups are 16 min of the 78 min of criterion time in
  # shapes-sat99-filtered.txt, and a criterion filter skips the timed loops only -- every group
  # still builds its ladder either way, which is why lf-sweep-hbref.txt spends 794 of its 978 s
  # outside criterion with one group of sixteen measured. So: that setup floor plus the 4/4
  # quarter, not a quarter of the 170 min the filtered cell took. The loose upper bound is
  # ~110 min, reached only if none of the twelve skipped groups' untimed table builds fall away.
  typeset -A STEP_MIN=(shapes 302 hbarray 50 lfsweep 538 hbref 16 parallel 19 chart7 12 cachedemo 160)
  typeset -i sel_min=0
  # shapes is the one step SHAPES_BS multiplies: a filtered+plain pair per bucket size.
  for ST in $SEL; do
    if [[ $ST == shapes ]]; then (( sel_min += STEP_MIN[$ST] * ${#BSIZES} ))
    else (( sel_min += STEP_MIN[$ST] )); fi
  done
  print -r -- "STEPS: ${(j:, :)SEL} only, in the campaign's own order."
  if (( ${#BSIZES} > 1 )) && (( ${SEL[(Ie)shapes]} )); then
    print -r -- "  shapes counts ${#BSIZES}x for SHAPES_BS=${(j:,:)BSIZES}; every other step, hbarray included,"
    print -r -- "  runs at the compiled-in bucket size once."
  fi
  print -r -- "  Those steps come to ~$(( sel_min / 60 )) h $(( sel_min % 60 )) min at the archived r=1.01 pass's"
  print -r -- "  own per-step durations (caps/r1.01/ mtimes; hbarray derived, shapes once per size)"
  print -r -- "  => ~$(( ${#RATIOS} * sel_min / 60 )) h $(( ${#RATIOS} * sel_min % 60 )) min for this invocation, first build on top."
  if (( ${#SEL} == 1 )) && [[ $SEL[1] == parallel ]]; then
    print -r -- "  NOTE: a parallel-only selection runs unguarded -- benches/parallel.rs never"
    print -r -- "  prints the cascade line the wrong-r guard greps for."
  fi
fi
if [[ "${SMOKE:-0}" != 1 ]]; then
  print -r -- "RAM: the 2^28 cells peak near 29 GiB (charts 1-4 and 5, the 16/16 groups' setup)"
  print -r -- "and 22 GiB (chart 8 K=28) -- a 32 GiB host with everything else closed; a 16 GiB"
  print -r -- "host will swap or OOM on every chart but 7. BENCH_PAPER has the per-cell numbers."
else
  print -r -- "SMOKE mode: shrunken lists, output under $CAPROOT, minutes not hours."
fi
print -r -- ""

for R in $RATIOS; do
  RDIR=$CAPROOT/r$R${TAG:+-$TAG}
  mkdir -p "$RDIR"
  guard_done=0 ratio_bad=0
  print -r -- "--- ratio $R -> $RDIR (nothing outside this directory is ever written) ---"

  # Charts 1-4: four shapes, filtered+hashbrown then plain, one pair per SHAPES_BS bucket size.
  if step_on shapes; then
    for BS in $BSIZES; do
      BSENV=() SFX=
      if (( bs_given )); then
        BSENV=(BUCKET_SIZE=$BS)
        [[ $BS == 8 ]] || SFX=-s$BS   # the default size keeps the archive's basename
      fi
      run_cell shapes-sat99-filtered$SFX.txt env MT_PAPER_RATIO=$R $BSENV MT_BENCH_ANCHORS=$SHAPE_ANCHORS MT_BENCH_SAT=0.99 MT_BENCH_HMAP=1 \
        cargo bench --config "$FLAGS" --bench comparison -- $SHAPE_FILTER $CRIT
      ratio_guard shapes-sat99-filtered$SFX.txt || break
      run_cell shapes-sat99-plain$SFX.txt env MT_PAPER_RATIO=$R MT_BENCH_TABLE=plain $BSENV MT_BENCH_ANCHORS=$SHAPE_ANCHORS MT_BENCH_SAT=0.99 \
        cargo bench --config "$FLAGS" --bench comparison -- $SHAPE_FILTER_PLAIN $CRIT
    done
  fi

  # Chart 1's full picture: the same 4/4 cells with hashbrown keyed on the raw byte keys rather
  # than the width's integer (MT_BENCH_HB_KEY=array), which is hashbrown taking the bytes the way
  # MultiTable does. Same anchors, saturation and ladder as the cell above -- array mode changes
  # nothing but the key type and the `_hbarray` group names -- so the two files' cells pair off.
  # SHAPES_BS does not reach it: hashbrown has no buckets to size, and the MultiTable arm it
  # carries is a same-session reference, not a bucket-size sweep of its own.
  if step_on hbarray; then
    run_cell shapes-sat99-hbarray.txt env MT_PAPER_RATIO=$R MT_BENCH_ANCHORS=$SHAPE_ANCHORS MT_BENCH_SAT=0.99 MT_BENCH_HMAP=1 MT_BENCH_HB_KEY=array \
      cargo bench --config "$FLAGS" --bench comparison -- $HB_ARRAY_FILTER $CRIT
    ratio_guard shapes-sat99-hbarray.txt || true   # nothing else in this step; ratio_bad stops the rest
  fi

  # Chart 5: LF x bucket size, fixed q per anchor, MT-only cells plus the hashbrown reference.
  if step_on lfsweep; then
    for S in 8 16 32; do
      run_cell lf-sweep-filtered-s$S.txt env MT_PAPER_RATIO=$R BUCKET_SIZE=$S MT_BENCH_ANCHORS=$LF_ANCHORS MT_BENCH_MT_LF=$LFS_FILTERED \
        cargo bench --config "$FLAGS" --bench comparison -- '^get_mixed/' $CRIT
      ratio_guard lf-sweep-filtered-s$S.txt || break
    done
  fi
  if step_on hbref; then
    run_cell lf-sweep-hbref.txt env MT_PAPER_RATIO=$R MT_BENCH_ANCHORS=$LF_ANCHORS MT_BENCH_HMAP=1 \
      cargo bench --config "$FLAGS" --bench comparison -- '^get_mixed/' $CRIT
    ratio_guard lf-sweep-hbref.txt || true   # nothing else in this step; ratio_bad stops the rest
  fi
  if step_on lfsweep; then
    for S in 8 16 32; do
      run_cell lf-sweep-plain-s$S.txt env MT_PAPER_RATIO=$R MT_BENCH_TABLE=plain BUCKET_SIZE=$S MT_BENCH_ANCHORS=$LF_ANCHORS MT_BENCH_MT_LF=$LFS_PLAIN \
        cargo bench --config "$FLAGS" --bench comparison -- '^get_mixed_plain/' $CRIT
    done
  fi

  # Chart 6: parallel get_mixed; the filtered run carries the HB arm, the plain run cross-checks it.
  # No ratio_guard here: this bench prints no cascade line (see the STEPS note in the banner).
  if step_on parallel; then
    run_cell parallel-filtered.txt env MT_PAPER_RATIO=$R MT_BENCH_LFS=0.77 $PAR_ENV \
      cargo bench --config "$FLAGS" --bench parallel
    run_cell parallel-plain.txt env MT_PAPER_RATIO=$R MT_BENCH_TABLE=plain MT_BENCH_LFS=0.77 $PAR_ENV \
      cargo bench --config "$FLAGS" --bench parallel
  fi

  # Chart 7: the q ladder is per-ratio (solved placements; see the arrays above).
  if step_on chart7; then
    if [[ $R == 1.01 ]]; then GOOD=($GOOD_101) BAD=($BAD_101); else GOOD=($GOOD_1001) BAD=($BAD_1001); fi
    run_variant agile
    run_variant pow2  --no-default-features --features powers-of-2
    run_variant first --no-default-features --features first-power
  fi

  # Chart 8: cache-residency sweep, same q both structures.
  if step_on cachedemo; then
    for K in $KS; do
      Q=$(( 7 * (1 << (K - 3)) + 1 ))
      run_cell cachedemo-filtered-q$Q.txt env MT_PAPER_RATIO=$R MT_BENCH_ANCHORS=10 MT_DEMO_N=$Q MT_DEMO_LF=0.866 MT_BENCH_HMAP=1 \
        cargo bench --config "$FLAGS" --bench comparison -- "^(insert|get_existing|get_nonexisting|get_mixed)/.*/$Q\$" $CRIT
      ratio_guard cachedemo-filtered-q$Q.txt || break
      run_cell cachedemo-plain-q$Q.txt env MT_PAPER_RATIO=$R MT_BENCH_TABLE=plain MT_BENCH_ANCHORS=10 MT_DEMO_N=$Q MT_DEMO_LF=0.97 \
        cargo bench --config "$FLAGS" --bench comparison -- "^(insert|get_existing|get_nonexisting|get_mixed)_plain/.*/$Q\$" $CRIT
    done
  fi
done

print -r -- ""
if (( fails )); then
  print -u2 -r -- "=== campaign finished with $fails failed cell(s); see $CAPROOT/r<R>${TAG:+-$TAG}/FAILURES.txt ==="
  exit 1
fi
print -r -- "=== campaign finished clean: ${(j:, :)RATIOS} -> $CAPROOT/r<R>${TAG:+-$TAG}/ ==="
