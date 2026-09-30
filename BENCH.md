# Benchmarks: method and reproduction

This file explains how the paper's comparison between MultiTable and hashbrown
is set up and how to re-run its captures from a clean checkout. It carries no
results.

## What is measured

One bench produces everything: `benches/comparison.rs` (criterion 0.5.1)
times single-threaded `insert`, `get_existing`, `get_nonexisting` and
`get_mixed` for MultiTable and, with `MT_BENCH_HMAP=1`, for hashbrown 0.16.1
(`inline-more`). Both are keyed with `FxBuildHasher` from rustc-hash 2.1.1,
since it's a fast hash and minimizes hash performance role in comparison.

The MultiTable arm is the filtered table (`MultiTableFiltered`) by default, or
the plain table when the crate is built with `MT_BENCH_TABLE=plain`. Plain
runs register their groups with a `_plain` suffix, so the two never share IDs.

Each cell draws its keys from seeded `ChaCha8Rng` streams: `n` present keys
inserted in stream order, `n` keys guaranteed absent, a separately shuffled
copy of the present keys for lookups, and a shuffled stream that is half
present and half absent for `get_mixed`. So:

- `insert` fills a table that was pre-sized for `n` and pre-touched in untimed
  setup, so page faults and growth stay out of the timed loop. The hashbrown
  arm asserts that the `n` inserts never grew the array it was handed.
- `get_existing` looks up every present key in an order unrelated to
  insertion order.
- `get_nonexisting` looks up `n` absent keys.
- `get_mixed` looks up `n` keys, half hits and half misses, interleaved at
  random.

Only the result of each operation goes through `black_box`, in both arms. The
hashbrown arm is keyed by default on the integer the byte key decodes to (`u32`
at 4 bytes, `u128` at 16), and that conversion is timed and inlined.
`MT_BENCH_HB_KEY=array` keys hashbrown on the raw `[u8; N]` instead, the way
MultiTable takes its keys. Array mode appends `_hbarray` to every group name,
so its output cannot be mistaken for the default arm's.

Criterion reports the time for a whole pass over the cell's `n` keys.
Cells above 2^21 keys run with `sample_size(10)`.
Otherwise criterion keeps its defaults.

## Sizing: both structures at the same footprint

Comparing at equal key counts would mostly measure memory, because hashbrown's
footprint is a step function of `n`. The benches therefore compare at an equal
**footprint load factor**, meaning payload bytes over allocated bytes.

- An anchor is a hashbrown bucket count, 2^12 through 2^28 in the paper.
  `MT_BENCH_ANCHORS` takes a comma list, where a value of 30 or less is an
  exponent and anything larger is a literal power-of-two bucket count.
- For each anchor, hashbrown is sized first and its allocated bytes are read
  back. Then `n` is chosen so that hashbrown and MultiTable reach the same
  footprint load factor.
- `MT_BENCH_SAT=0.99` sets that load factor to 99% of the shape's own
  saturation, which is hashbrown's 7/8 slot load times pair/(pair+1). This
  keeps hashbrown one step short of doubling at every width.
  `MT_BENCH_LF` takes absolute load factors instead. The two cannot be set
  together.
- `MT_BENCH_MT_LF` resizes the MultiTable alone, at footprints hashbrown cannot
  reach, and keeps the ladder's key counts. It refuses to run alongside
  `MT_BENCH_HMAP=1`, `MT_BENCH_SAT` or `MT_BENCH_LF`.
- `MT_DEMO_N` and `MT_DEMO_LF` append one hand-picked cell with a given count
  and MultiTable footprint. Figure 13 uses this cell, with
  `MT_BENCH_ANCHORS=10` shrinking the regular ladder to a trivial cell that
  the criterion filter excludes.
- MultiTable's per-level bucket counts come from the sizing driver at overfill
  ratio `MT_PAPER_RATIO`, which defaults to 1.01. That is the paper's ratio.

Every capture prints the realized footprint of both structures on untimed
`cell K/V: n = …, footprint LF = …` lines, and each MultiTable build prints
`sizing ratio r = …, buckets per level = […]`. The paper's figures plot realized
values, not requested ones. The bench never prints its bucket size, so the
`-s16` style suffix in a capture's file name is the only label a capture
carries. `cargo run --release --example paper_plan -- <lfs> <qs>` prints the
cascade plan and realized footprint for any cell without running the bench,
and it honours `BUCKET_SIZE`, `MT_PAPER_RATIO` and the variant features.

## Shapes, bucket sizes and variants

- Shapes: 4/4 and 16/16 key/value bytes (groups `<op>` and `<op>_kv16`), and
  keys-only 4/0 and 16/0 (`<op>_set4`, `<op>_set16`). All sixteen groups run
  unless a criterion filter narrows them. Every group still builds its key
  streams whether or not the filter keeps it, so a filtered run still spends
  that setup time.
- Bucket size: `BUCKET_SIZE=8|16|32` at build time, and the SIMD width follows
  it. **The filtered table defaults to 16 and the plain table to 8.** The
  paper's captures run at 8 unless a command sets another size, so the
  commands below pin `BUCKET_SIZE=8` wherever they would otherwise take the
  default.
- Variants: `agile` is the default feature. `--no-default-features --features
  powers-of-2` and `--no-default-features --features first-power` select the
  other two level-sizing modes.

`MT_BENCH_TABLE`, `BUCKET_SIZE`, `MT_PAPER_RATIO`, the variant features, the
toolchain and the `--config` flag string are all build-time settings. Cargo
rebuilds on its own when any of them changes. Every other `MT_BENCH_*` and
`MT_DEMO_*` variable is read when the bench starts.

## Machine and ground rules

- Reference host: an Apple M2 Pro (12 cores) with 32 GiB. The 2^28 cells need
  close to 29 GiB, so close everything else. A 16 GiB machine cannot run them in RAM.
- Use one flag string for every capture, passed through `--config` and never
  through an exported `RUSTFLAGS`. An exported `RUSTFLAGS` replaces the
  `.cargo/config.toml` rustflags and silently drops the function-alignment
  floor:

  ```sh
  FLAGS='build.rustflags=["-C","llvm-args=-align-all-functions=6","-C","target-cpu=native"]'
  ```

- Use one toolchain for the whole campaign. `build.rs` switches the SIMD scans
  on when it detects a nightly `rustc`.
- Keep the machine idle. On Linux, pin each run to one core (for example
  `taskset -c 4 …`).
- Criterion history is shared across bucket sizes and variants,
  because they reuse benchmark IDs.
- Budget roughly 21 hours for one full pass at the 2^28 tops. Most of that is
  untimed key-stream setup in the large cells.

## Reproducing

Run everything from the repo root. You need Rust and zsh (bash works too with small changes
for the commands written out below, but the runner itself is zsh).

### 1. Build and smoke-test

```sh
FLAGS='build.rustflags=["-C","llvm-args=-align-all-functions=6","-C","target-cpu=native"]'
cargo bench --config "$FLAGS" --bench comparison --no-run
SMOKE=1 zsh run_ratio_campaign.zsh 1.01
```

The smoke run is the whole campaign in miniature, with small anchors and short
criterion times. It writes under `/tmp` and never touches `caps/`.

### 2. The campaign runner

`run_ratio_campaign.zsh <ratio>` runs its capture steps in a fixed order and
tees each one into `caps/r<ratio>[-<TAG>]/`. It takes these knobs:

- `STEPS` picks a subset of the steps. The paper's figures need `shapes`,
  `hbarray`, `lfsweep` and `hbref`.
- `TAG` appends a suffix to the directory name.
- `SHAPES_BS=8,16` runs the shapes step once per bucket size. Size 8 keeps the
  plain file names and other sizes get an `-s<S>` suffix.
- `RESUME=1` keeps finished cells and runs the rest. `FORCE=1` re-runs cells
  over existing files. Without either, the runner refuses to write into a
  directory that already holds captures.

Failed cells are logged to `FAILURES.txt` in the target directory. The first
capture of a pass must print the requested `r`, or the runner abandons that
ratio.

The archived captures are checked in under `caps/`, and the commands below
write over them in place with `FORCE=1`. `git checkout -- caps` restores the
archive afterwards. `BUCKET_SIZE=8` is exported for the whole run, so every
step that does not set its own size runs at 8. Cells that set their own size
(`SHAPES_BS`, and the sweeps for Figure 12) override it.

The runner's captures for Figures 6, 7 and 9 to 12 go into
`caps/r1.01-hbfix-perf/`:

```sh
BUCKET_SIZE=8 SHAPES_BS=8,16 STEPS=shapes,hbarray,lfsweep,hbref \
  TAG=hbfix-perf FORCE=1 zsh run_ratio_campaign.zsh 1.01
```

### 3. What the runner does, figure by figure

These are the invocations the runner makes, written out for re-running a
single figure by hand. `R=caps/r1.01-hbfix-perf` stands in for the target
directory.

Figures 6 and 9 to 11, per bucket size S in 8 and 16. The file suffix is empty
for 8 and `-s16` for 16. The figures take the filtered table at 16 and the
plain one at 8, and the other pair backs the paper's bucket-size comparison in
its text:

```sh
MT_PAPER_RATIO=1.01 BUCKET_SIZE=$S MT_BENCH_ANCHORS=12,14,16,18,20,22,28 MT_BENCH_SAT=0.99 MT_BENCH_HMAP=1 \
  cargo bench --config "$FLAGS" --bench comparison 2>&1 | tee $R/shapes-sat99-filtered$SFX.txt
MT_PAPER_RATIO=1.01 MT_BENCH_TABLE=plain BUCKET_SIZE=$S MT_BENCH_ANCHORS=12,14,16,18,20,22,28 MT_BENCH_SAT=0.99 \
  cargo bench --config "$FLAGS" --bench comparison 2>&1 | tee $R/shapes-sat99-plain$SFX.txt
```

To run one shape only, append a filter after `--`:
`'^(insert|get_existing|get_nonexisting|get_mixed)/'` for 4/4, or the same
pattern with `_kv16`, `_set4` or `_set16` after each operation name. Add
`_plain` after that for the plain run.

The 4/4 shape of Figures 6 and 7 with hashbrown keyed on the bytes. The MultiTable arm runs alongside
as a same-session reference:

```sh
MT_PAPER_RATIO=1.01 BUCKET_SIZE=8 MT_BENCH_ANCHORS=12,14,16,18,20,22,28 MT_BENCH_SAT=0.99 \
  MT_BENCH_HMAP=1 MT_BENCH_HB_KEY=array \
  cargo bench --config "$FLAGS" --bench comparison \
  -- '^(insert|get_existing|get_nonexisting|get_mixed)_hbarray/' 2>&1 | tee $R/shapes-sat99-hbarray.txt
```

Figure 12, including the hashbrown reference:

```sh
LFS_FILTERED=0.47,0.57,0.67,0.72,0.77,0.80,0.82,0.84,0.86,0.88
LFS_PLAIN=0.86,0.88,0.90,0.92,0.94,0.95,0.96,0.97,0.977
for S in 8 16 32; do
  MT_PAPER_RATIO=1.01 BUCKET_SIZE=$S MT_BENCH_ANCHORS=16,22,28 MT_BENCH_MT_LF=$LFS_FILTERED \
    cargo bench --config "$FLAGS" --bench comparison -- '^get_mixed/' 2>&1 | tee $R/lf-sweep-filtered-s$S.txt
done
MT_PAPER_RATIO=1.01 BUCKET_SIZE=8 MT_BENCH_ANCHORS=16,22,28 MT_BENCH_HMAP=1 \
  cargo bench --config "$FLAGS" --bench comparison -- '^get_mixed/' 2>&1 | tee $R/lf-sweep-hbref.txt
for S in 8 16 32; do
  MT_PAPER_RATIO=1.01 MT_BENCH_TABLE=plain BUCKET_SIZE=$S MT_BENCH_ANCHORS=16,22,28 MT_BENCH_MT_LF=$LFS_PLAIN \
    cargo bench --config "$FLAGS" --bench comparison -- '^get_mixed_plain/' 2>&1 | tee $R/lf-sweep-plain-s$S.txt
done
```

The three filtered sweeps and the plain s = 8 sweep end with a panic in the
4/0 shape, which their filter never times, because a 4/0 table cannot reach
the top of the list. All the timed cells come before the panic, so
`FAILURES.txt` logs the panic and no data is lost. Plain 0.977 at 2^28 with s = 8
does not size cleanly and realizes about 0.81, so that point is left off its
line.

### 4. Captures the runner does not make

The 16/16 shape at 2^27, which Figure 9 uses in place of its 2^28 cells, at
both bucket sizes and for both tables:

```sh
KV16='^(insert|get_existing|get_nonexisting|get_mixed)_kv16'
for S in 8 16; do
  SFX=$([ $S = 16 ] && echo -s16)
  MT_PAPER_RATIO=1.01 BUCKET_SIZE=$S MT_BENCH_ANCHORS=27 MT_BENCH_SAT=0.99 MT_BENCH_HMAP=1 \
    cargo bench --config "$FLAGS" --bench comparison -- "$KV16/" \
    2>&1 | tee caps/r1.01-hbfix-perf/shapes-sat99-filtered$SFX-kv16-a27.txt
  MT_PAPER_RATIO=1.01 MT_BENCH_TABLE=plain BUCKET_SIZE=$S MT_BENCH_ANCHORS=27 MT_BENCH_SAT=0.99 \
    cargo bench --config "$FLAGS" --bench comparison -- "${KV16}_plain/" \
    2>&1 | tee caps/r1.01-hbfix-perf/shapes-sat99-plain$SFX-kv16-a27.txt
done
```

Byte-keyed hashbrown for the other three shapes, Figures 9 to 11. Only the hashbrown arm runs:

```sh
MT_PAPER_RATIO=1.01 BUCKET_SIZE=8 MT_BENCH_ANCHORS=12,14,16,18,20,22,27 MT_BENCH_SAT=0.99 \
  MT_BENCH_HMAP=1 MT_BENCH_HB_KEY=array cargo bench --config "$FLAGS" --bench comparison \
  -- '^(insert|get_existing|get_nonexisting|get_mixed)_kv16_hbarray/HashMap/' \
  2>&1 | tee caps/r1.01-hbfix-perf/shapes-sat99-hbarray-kv16.txt
MT_PAPER_RATIO=1.01 BUCKET_SIZE=8 MT_BENCH_ANCHORS=12,14,16,18,20,22,28 MT_BENCH_SAT=0.99 \
  MT_BENCH_HMAP=1 MT_BENCH_HB_KEY=array cargo bench --config "$FLAGS" --bench comparison \
  -- '^(insert|get_existing|get_nonexisting|get_mixed)_set(4|16)_hbarray/HashMap/' \
  2>&1 | tee caps/r1.01-hbfix-perf/shapes-sat99-hbarray-sets.txt
```

Figure 13 goes into `caps/r1.01-sel/`. This is zsh, and the ladders must stay
arrays. A quoted string would reach `MT_DEMO_N` as a single count, and the
bench would abort.

```sh
GOOD=(15758 31376 62571 124907 249494 498556 996519 1992220 3983297 7964997)
BAD=(29836 59482 118729 237162 473902 947227 1893654 3786192 7570826)
run_variant () {
  TAG=$1; shift
  for Q in $GOOD $BAD; do
    BUCKET_SIZE=16 MT_PAPER_RATIO=1.01 MT_BENCH_ANCHORS=10 MT_DEMO_N=$Q MT_DEMO_LF=0.77 \
    cargo bench --config "$FLAGS" --bench comparison "$@" \
      -- "^get_mixed/MultiTable/$Q\$" 2>&1 | tee caps/r1.01-sel/variant-$TAG-s16-q$Q.txt
  done
}
run_variant agile
run_variant pow2  --no-default-features --features powers-of-2
run_variant first --no-default-features --features first-power
```

That is the whole reproduction. It ends with criterion's raw output, one
text file per cell, under `caps/r1.01-hbfix-perf/` and `caps/r1.01-sel/`.
