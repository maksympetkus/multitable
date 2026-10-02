### Disclaimer: this library isn't production-grade, use at your own risk.

# MultiTable

A cascading multi-level hash table, built to hold a key-value pairs in
substantially less memory. A key hashes to one bucket per level and settles in
the first level whose bucket has room, so the table runs at load factors well
past the point where an open-addressing table has doubled its array; the trade
is a probe that may touch a few levels instead of one. Two implementations
share the crate: `MultiTableFiltered`, where a per-slot filter byte
gates every bucket probe, and the plain `MultiTable` that achieves load factor 0.9999.

<img width="1536" height="1024" alt="1" src="https://github.com/user-attachments/assets/d64436a0-f1a3-4bc3-b546-56801143cae0" />

## Paper

Read the paper at https://arxiv.org/abs/2609.39233

## Project structure

- `src/lib.rs` is the crate root: the variant-feature checks, the shared
  bucket storage, the level hash selectors and the default bucket sizes. It
  pulls both tables in with `include!`.
- `src/filtered.rs` holds `MultiTableFiltered` and `InsertError`: the
  filter-byte walk, `insert`, `upsert`, `remove` and `get`.
- `src/plain.rs` holds the filterless `MultiTable`.
- `src/sizing.rs` is the paper's level-size computation that both tables'
  `with_capacity` goes through, including the power-of-two roundings.
- `src/utils.rs` has the key hashing, the bucket and filter scans (portable
  SIMD on nightly, NEON and SSE2/AVX-512 intrinsics on stable), and the
  prefetch and branch-hint helpers.
- `src/tests.rs` is the test suite; there is no `tests/` directory.
- `build.rs` turns on the nightly SIMD paths and reads `MT_BENCH_TABLE`.
- `Cargo.toml` defines the features: the variants `agile`, `powers-of-2` and
  `first-power`, plus `nightly` (force the SIMD paths), `jema` and `mima`
  (jemalloc or mimalloc as the global allocator) and `profiling` (keeps the
  hot functions out of line so a profiler can see them).
- `.cargo/config.toml` sets the `-align-all-functions=6` floor every published
  number is measured with.
- `benches/comparison.rs` is the head-to-head against hashbrown
- `BENCH.md` describes the benchmark method and the commands that capture
  the data behind the paper's charts, `BENCH_PAPER.md` is the capture plan and log for those
  charts, and `BENCH_X86.md` covers running the bench on an x86_64 host.

Building and testing are ordinary:

```bash
cargo build
cargo test
```

Stable and nightly both work out of the box — build.rs detects a nightly
compiler and turns the SIMD paths on itself. Exactly one variant feature must
be enabled: `agile` (the default), or `powers-of-2` / `first-power` via
`--no-default-features --features <variant>`, which round the level sizes to
powers of two (every level, or only the first) so those levels are addressed
by a shift rather than a multiply.



## Perfect Table

Perfect table is when every available slot is filled, i.e., load factor 1, and practically 
all the allocated bytes are occupied by the key-value payload, as if you'd have a regular array.

To run a demonstration of the "perfect" table use:

```bash 
cargo run --example perfect --release
```

This prints the realized physical load factor.

## Benchmarks

The head-to-head against hashbrown is the comparison bench:

```bash
MT_BENCH_HMAP=1 cargo bench --bench comparison
```

Two knobs are read at build time: `MT_BENCH_TABLE=filtered|plain` picks the
table the MultiTable arm runs, and `BUCKET_SIZE=<n>` overrides the default
slots per bucket (eight for the plain table, sixteen for the filtered one).
The rest are read when the bench runs: `MT_BENCH_HMAP=1` turns the hashbrown arms on,
and the load-factor ladder is steered by `MT_BENCH_LF` (absolute footprint
load factors, payload bytes over allocated bytes), `MT_BENCH_SAT` (the same
list as fractions of the shape's own saturation) or `MT_BENCH_MT_LF`
(MultiTable-only footprints past hashbrown's ceiling, with no comparison arm).
Each knob is documented in full at the helper in `benches/comparison.rs` that
parses it. [BENCH.md](BENCH.md) explains how the knobs combine into
the paper's captures. It also covers the benchmark methodology, which captures each of
the paper's charts reads, and the commands to re-run them from a clean
checkout.

## Implementation Status

Neither table grows (not implemented). Its levels are sized once, at construction, and an
insert whose chain has no room left on any level is refused with an `Err`.
The filtered table's `insert` and `upsert` return `Result<(), InsertError>`:
`InsertError::Full` when there is no room, `InsertError::AlreadyPresent` when
`insert` meets a key it already holds, and in either case nothing has been
written. The plain table's `insert` reports both cases as an `Err` carrying a
message string.

Max bucket size is currently 64.

| | `MultiTableFiltered` | `MultiTable` |
|---|---|---|
| `insert`, `get` | yes | yes |
| `upsert` | yes | no |
| `remove` | yes | no |
| Auto-resizing | no | no |
| Deletion churn management | no | no |
| Iteration, `len`, `clear`, `get_mut`, Entry API | no | no |
| Storing on filesystem | no | no |
| Keys and values | fixed-size byte arrays, `[u8; KEY_LEN]` and `[u8; VAL_LEN]` | the same |
| Hasher | any `BuildHasher + Default`, `FxBuildHasher` by default | `FxBuildHasher` only |
| Variant features | `agile`, `powers-of-2`, `first-power` | ignored; always the `agile` sizing |
| SIMD scans | portable SIMD on nightly, NEON or SSE2/AVX-512 on stable | the same |

What there is beyond that is introspection: `total_buckets` and
`level_bucket_cnts` on both, `bucket_fill` on the plain table, and
`DeepSizeOf` for the memory footprint. Keys are raw bytes, so anything else
has to be encoded into a fixed-width array by the caller.

## Choice of Hash Function

For the benchmarks we use FxHash for both MultiTable and hashbrown, since it one 
of the cheapest hash functions, which is important to reduce hash performance
influence over the MultiTable's performance.

In our tests FxHash was sufficient where keys weren't structured, e.g., pseudorandom keys or sequential keys, but it's not ideal for structured keys, such as `i * 64`, and might lead to premature terminal overflow, for such cases FoldHash performed better without overflows observed.

Use a strong cryptography-grade hash function if your use-case requires it.

## Optimizations

Some effort have been directed towards optimizing the performance of the library (primarily manual and often a result of a lengthy trial and error exploration), especially since the hashbrown baseline is already heavily optimized. However, there are still potential optimization opportunities left.

## Deletion Churn

Deletion churn is handled only as far as reuse goes. `remove` clears the slot's
filter byte (a tombstone in a bucket's last slot), and from the first delete
on, inserts take the recording walk, which offers the shallowest freed slot
on the key's chain. Nothing cleans up beyond that: there is no rehash, no
compaction and no tombstone sweep, and a freed slot only serves a later key
whose own chain crosses that bucket. 

Heavy churn settles: the test that swaps a third of 20,000 keys per round for sixteen rounds sees the frontier
stop moving by the halfway round. Light churn at capacity does not: with a table sized for
those 20,000 keys at 0.8 and a twentieth of them replaced each round, level 0
drains into tail levels that were sized for a fresh fill, and the cascade
runs out of room under `agile` at round 35 (1.8 turnovers) at bucket size
16 and at round 13 at size 8, and between rounds 9 and 19 under the other two
variants. That case is pinned as an ignored test,
`churn_at_capacity_by_a_twentieth_a_round_never_runs_out_of_room`, which
sees the overflow as `InsertError::Full`. 

See the paper for the churn management strategies (not implemented).

