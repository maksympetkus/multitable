use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};
use deepsize::DeepSizeOf;
use foldhash::fast::RandomState as FoldRandomState;
use hashbrown::hash_map::HashMap as HashBrownMap;
use multitable::sizing_ratio;
#[cfg(mt_bench_plain)]
use multitable::MultiTable as MultiTablePlain;
#[cfg(not(mt_bench_plain))]
use multitable::MultiTableFiltered;
#[cfg(mt_bench_plain)]
use multitable::PlainSearchResult;
#[cfg(not(mt_bench_plain))]
use multitable::SearchResult;
#[cfg(mt_bench_plain)]
use multitable::DEFAULT_BUCKET_SIZE;
#[cfg(not(mt_bench_plain))]
use multitable::DEFAULT_FILTERED_BUCKET_SIZE;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustc_hash::FxBuildHasher;

use std::hash::BuildHasher;
use std::hash::{Hash, Hasher};
use std::hint::black_box;

const T: bool = true;

type HasherBuilderUsed = FxBuildHasher;
// type HasherBuilderUsed = FoldRandomState;

const RUN_MULTITABLE: bool = T;
const MAX_LEVELS: usize = 16;

// Four key/value widths share one set of bench functions: 4/4 keeps the original group names for
// saved baselines, 16/16 uses `_kv16`, and keys-only 4/0 and 16/0 use `_set4`/`_set16`.

/// hashbrown grows once an insert would push occupancy past 7/8 of the bucket array: a slot
/// load, distinct from the footprint ruler below, which differs by the control byte.
const HMAP_MAX_SLOT_LOAD: f64 = 0.875;

/// Bucket counts the ladder anchors to: one table that fits in cache, one mostly in RAM.
/// A step then picks the count within an anchor that realizes its load factor.
const LADDER_ANCHORS: &[usize] = &[1 << 16, 1 << 21];

/// Ladder steps, as absolute footprint load factors (payload bytes over allocated bytes).
/// Empty runs the default ladder: five steps up to the width's own saturation point.
const LADDER_STEPS: &[f64] = &[0.77];

/// The keys-only ladder's steps, as fractions of the shape's own saturation: an absolute
/// step wouldn't transfer, and the fraction keeps hashbrown one insert short of doubling.
const SET_LADDER_STEPS: &[f64] = &[0.99];

/// The counts this bench has always run sit just under exact saturation; the saturated
/// cell keeps them verbatim so criterion history for those IDs carries over.
const LEGACY_SATURATED: &[(usize, usize)] = &[(1 << 16, 8 * 7167), (1 << 24, 2 * 1024 * 7167)];

/// The MultiTable footprint LF a demo cell defaults to: the ladder's own, so the demo
/// differs only in hashbrown's luck. Load factor trades footprint for probe depth.
const DEMO_LF: f64 = 0.77;

/// One cell of the ladder: `n` items in an anchor's `buckets` buckets, carrying both the
/// slot load hashbrown grows on and the footprint load factor both structures are sized to.
#[derive(Clone, Copy, Debug)]
struct LadderCell {
    n: usize,
    buckets: usize,
    slot_load: f64,
    footprint: f64,
    /// A demo cell's MultiTable footprint, sized apart from hashbrown's on purpose. `None`
    /// on a ladder cell, which sizes both structures to `footprint`.
    mt_footprint: Option<f64>,
    /// An `MT_BENCH_MT_LF` cell: MultiTable-only, at a footprint no hashbrown of this width
    /// can reach; `buckets`/`slot_load` describe the count's provenance, not what gets built.
    mt_only: bool,
}

impl LadderCell {
    fn mt_lf(&self) -> f64 {
        self.mt_footprint.unwrap_or(self.footprint)
    }

    /// The criterion id the MultiTable arm registers under. An MT-only cell spells out its
    /// footprint since it shares a count with the ladder cell it came from.
    fn mt_arm_id(&self) -> String {
        match self.mt_only {
            true => format!("MultiTable_mtlf{:.4}", self.mt_lf()),
            false => "MultiTable".to_string(),
        }
    }
}

/// The footprint a full-to-saturation hashbrown of this width reaches:
/// 7/8 of the slots filled, each slot costing its pair plus a control byte.
fn max_footprint<const KEY_LEN: usize, const VAL_LEN: usize>() -> f64 {
    let pair = (KEY_LEN + VAL_LEN) as f64;
    HMAP_MAX_SLOT_LOAD * pair / (pair + 1.0)
}

/// The footprint the *filtered* MultiTable reaches at occupancy 1.0: the same per-slot byte
/// hashbrown spends, but with no 7/8 stop. That gap is what `MT_BENCH_MT_LF` exists to measure.
#[cfg(not(mt_bench_plain))]
fn mt_max_footprint<const KEY_LEN: usize, const VAL_LEN: usize>() -> f64 {
    let pair = (KEY_LEN + VAL_LEN) as f64;
    pair / (pair + 1.0)
}

/// The same for the *plain* table, which spends its byte per bucket rather than per slot:
/// a full bucket holds `s * pair` payload bytes in `s * pair + 1` allocated.
#[cfg(mt_bench_plain)]
fn mt_max_footprint<const KEY_LEN: usize, const VAL_LEN: usize>() -> f64 {
    let bucket = (BUCKET_SIZE * (KEY_LEN + VAL_LEN)) as f64;
    bucket / (bucket + 1.0)
}

/// `MT_BENCH_MT_LF`: comma-separated MultiTable-only footprint load factors, sized on the
/// default ladder's counts. `None` when unset, the only state where both arms pair up.
fn mt_only_steps<const KEY_LEN: usize, const VAL_LEN: usize>() -> Option<Vec<f64>> {
    let spec = std::env::var("MT_BENCH_MT_LF").ok()?;
    for rival in ["MT_BENCH_LF", "MT_BENCH_SAT"] {
        assert!(
            std::env::var(rival).is_err(),
            "MT_BENCH_MT_LF and {rival} are rival rulers for one step: set one or the other"
        );
    }
    assert!(
        std::env::var("MT_BENCH_HMAP").as_deref() != Ok("1"),
        "MT_BENCH_MT_LF runs no hashbrown arm: it exists for footprints hashbrown \
         cannot reach at all, where there is nothing left to compare against"
    );
    let ceiling = mt_max_footprint::<KEY_LEN, VAL_LEN>();
    let steps: Vec<f64> = spec
        .split(',')
        .map(|part| {
            part.trim()
                .parse()
                .unwrap_or_else(|_| panic!("MT_BENCH_MT_LF: `{part}` is not a number"))
        })
        .collect();
    for &lf in &steps {
        assert!(
            lf > 0.0 && lf <= ceiling,
            "MT_BENCH_MT_LF {lf} is outside (0, {ceiling:.6}], the footprints a \
             {KEY_LEN}/{VAL_LEN} {TABLE_KIND} MultiTable can reach -- its own \
             ceiling, not the {:.6} a hashbrown of this width stops at",
            max_footprint::<KEY_LEN, VAL_LEN>()
        );
    }
    Some(steps)
}

/// `MT_BENCH_ANCHORS`: comma-separated anchors, else `LADDER_ANCHORS`. A value at most 30
/// is an exponent (`2^k`); anything larger is a bucket count, which must be a power of two.
fn ladder_anchors() -> Vec<usize> {
    let Ok(spec) = std::env::var("MT_BENCH_ANCHORS") else {
        return LADDER_ANCHORS.to_vec();
    };
    spec.split(',')
        .map(|part| {
            let v: usize = part
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("MT_BENCH_ANCHORS: `{part}` is not a count"));
            if v <= 30 {
                1usize << v
            } else {
                v
            }
        })
        .collect()
}

/// `MT_BENCH_SAT`: comma-separated fractions of the shape's saturation footprint (1.0 is
/// saturation itself). `None` when unset, the only state where the ladders below apply.
fn saturation_steps() -> Option<Vec<f64>> {
    let spec = std::env::var("MT_BENCH_SAT").ok()?;
    assert!(
        std::env::var("MT_BENCH_LF").is_err(),
        "MT_BENCH_LF and MT_BENCH_SAT are rival rulers for one step: set one or the other"
    );
    Some(
        spec.split(',')
            .map(|part| {
                part.trim()
                    .parse()
                    .unwrap_or_else(|_| panic!("MT_BENCH_SAT: `{part}` is not a number"))
            })
            .collect(),
    )
}

/// `MT_BENCH_LF`: comma-separated absolute footprint load factors, else `MT_BENCH_SAT`,
/// `LADDER_STEPS`, or the width's default ladder, in that priority order.
fn ladder_steps<const KEY_LEN: usize, const VAL_LEN: usize>() -> Vec<f64> {
    let max = max_footprint::<KEY_LEN, VAL_LEN>();
    if let Some(fractions) = saturation_steps() {
        let steps = fractions.iter().map(|fraction| fraction * max).collect();
        return checked_steps::<KEY_LEN, VAL_LEN>(steps, max);
    }
    let steps: Vec<f64> = match std::env::var("MT_BENCH_LF") {
        Ok(spec) => spec
            .split(',')
            .map(|part| {
                part.trim()
                    .parse()
                    .unwrap_or_else(|_| panic!("MT_BENCH_LF: `{part}` is not a number"))
            })
            .collect(),
        // A keys-only shape saturates too low for an absolute step to carry over, so it
        // takes the same fraction of its own maximum whatever `LADDER_STEPS` holds.
        Err(_) if VAL_LEN == 0 => SET_LADDER_STEPS
            .iter()
            .map(|fraction| fraction * max)
            .collect(),
        Err(_) if LADDER_STEPS.is_empty() => [0.5, 0.625, 0.75, 0.875, 1.0]
            .iter()
            .map(|fraction| fraction * max)
            .collect(),
        Err(_) => LADDER_STEPS.to_vec(),
    };
    checked_steps::<KEY_LEN, VAL_LEN>(steps, max)
}

/// Returns `steps` once every one is a footprint this width can reach on a single bucket
/// array; anything else is a hard error naming the valid range.
fn checked_steps<const KEY_LEN: usize, const VAL_LEN: usize>(
    steps: Vec<f64>,
    max: f64,
) -> Vec<f64> {
    for &lf in &steps {
        assert!(
            lf >= max / 2.0 && lf <= max,
            "load factor {lf} is outside [{:.6}, {max:.6}], the footprints a {KEY_LEN}/{VAL_LEN} \
             hashbrown can reach on one bucket array",
            max / 2.0
        );
    }
    steps
}

/// The real heap bytes a hashbrown lands on at `buckets` capacity: built at the array's own
/// 7/8 top and read back via `allocation_size()`, hashbrown's own number.
fn hashbrown_allocation_size<const KEY_LEN: usize, const VAL_LEN: usize, HK: HmapKey<KEY_LEN>>(
    buckets: usize,
) -> usize {
    HashBrownMap::<HK, [u8; VAL_LEN], HasherBuilderUsed>::with_capacity_and_hasher(
        buckets * 7 / 8,
        HasherBuilderUsed::default(),
    )
    .allocation_size()
}

/// The ladder, derived before any group is registered: hashbrown sizes first, since it only
/// has powers of two to choose from, and MultiTable takes the same count and footprint.
fn bench_ladder<const KEY_LEN: usize, const VAL_LEN: usize, HK: HmapKey<KEY_LEN>>(
) -> Vec<LadderCell> {
    let pair = (KEY_LEN + VAL_LEN) as f64;
    let steps = ladder_steps::<KEY_LEN, VAL_LEN>();
    // The one mode whose top step means the array filled to capacity and
    // nothing else, so the legacy count a step under it is not that cell.
    let to_capacity = saturation_steps().is_some();
    let anchors = ladder_anchors();
    let mut cells = Vec::with_capacity(anchors.len() * steps.len());
    for &buckets in &anchors {
        // The array's real allocated bytes at this anchor, hashbrown's own
        // number in place of the reconstructed buckets*(pair+1) guess.
        let alloc_bytes = hashbrown_allocation_size::<KEY_LEN, VAL_LEN, HK>(buckets) as f64;
        for &footprint in &steps {
            let saturated = buckets * 7 / 8;
            // Inverts the footprint into `n`: payload pairs over the array's measured bytes,
            // clamped to keep the count above half saturation.
            let exact = footprint * alloc_bytes / pair;
            let mut n = (exact.round() as usize).clamp(saturated / 2 + 1, saturated);
            if n == saturated && !to_capacity {
                if let Some((_, legacy)) = LEGACY_SATURATED.iter().find(|(b, _)| *b == buckets) {
                    n = *legacy;
                }
            }
            cells.push(LadderCell {
                n,
                buckets,
                slot_load: n as f64 / buckets as f64,
                footprint: n as f64 * pair / alloc_bytes,
                mt_footprint: None,
                mt_only: false,
            });
        }
    }

    // The default ladder tops out at saturation, so its top cell must match this bench's
    // legacy count. Keys-only shapes stop short of it; `MT_BENCH_SAT` fills to capacity instead.
    if VAL_LEN != 0
        && LADDER_STEPS.is_empty()
        && std::env::var("MT_BENCH_LF").is_err()
        && std::env::var("MT_BENCH_ANCHORS").is_err()
        && !to_capacity
    {
        for &(buckets, legacy) in LEGACY_SATURATED {
            assert!(
                cells.iter().any(|c| c.buckets == buckets && c.n == legacy),
                "default ladder no longer tops out at {legacy} for {buckets} buckets"
            );
        }
    }

    // `MT_BENCH_MT_LF` keeps every count above and re-sizes only the MultiTable, so a denser
    // table is read against the ladder cell it came from, over the same keys.
    if let Some(mt_steps) = mt_only_steps::<KEY_LEN, VAL_LEN>() {
        cells = cells
            .iter()
            .flat_map(|cell| {
                mt_steps.iter().map(move |&lf| LadderCell {
                    mt_footprint: Some(lf),
                    mt_only: true,
                    ..*cell
                })
            })
            .collect();
    }

    cells.extend(demo_cell::<KEY_LEN, VAL_LEN, HK>());
    cells
}

/// `MT_DEMO_N`/`MT_DEMO_LF`: an extra cell's exact count and MultiTable footprint LF,
/// appended after the ladder so ladder cells keep their criterion IDs.
fn demo_cell<const KEY_LEN: usize, const VAL_LEN: usize, HK: HmapKey<KEY_LEN>>(
) -> Option<LadderCell> {
    let n: usize = match std::env::var("MT_DEMO_N") {
        Ok(spec) => spec
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("MT_DEMO_N: `{spec}` is not a count")),
        Err(_) => {
            assert!(
                std::env::var("MT_DEMO_LF").is_err(),
                "MT_DEMO_LF sizes the demo cell MT_DEMO_N appends, and nothing appends \
                 it: set MT_DEMO_N too, or neither"
            );
            return None;
        }
    };
    let mt_footprint = match std::env::var("MT_DEMO_LF") {
        Ok(spec) => spec
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("MT_DEMO_LF: `{spec}` is not a number")),
        Err(_) => DEMO_LF,
    };
    // hashbrown's realized rulers at this count, not a target it was derived from: a demo
    // count is chosen for its geometry, whatever LF results.
    let hmap = HashBrownMap::<HK, [u8; VAL_LEN], HasherBuilderUsed>::with_capacity_and_hasher(
        n,
        HasherBuilderUsed::default(),
    );
    let (buckets, slot_load, footprint) =
        realized_ruler::<KEY_LEN, VAL_LEN>(n, hmap.capacity(), hmap.allocation_size());
    Some(LadderCell {
        n,
        buckets,
        slot_load,
        footprint,
        mt_footprint: Some(mt_footprint),
        mt_only: false,
    })
}

/// SplitMix64 finalizer. Bijective on `u64`, same argument.
#[inline(always)]
pub const fn mix64(mut z: u64) -> u64 {
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Murmur3 32-bit finalizer, bijective on `u32` (both multipliers are odd,
/// every xor-shift is invertible). Used by `expand_key` (the 16-byte arm).
#[inline(always)]
pub const fn mix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x85eb_ca6b);
    x ^= x >> 13;
    x = x.wrapping_mul(0xc2b2_ae35);
    x ^ (x >> 16)
}

/// Expands a sampled `u32` into a 16-byte key: the low 4 bytes are the `u32` itself, which
/// alone gives uniqueness; the rest is mixing to avoid a visible zero-run.
#[inline(always)]
fn expand_key(v: u32) -> [u8; 16] {
    let mut k = [0u8; 16];
    k[0..4].copy_from_slice(&v.to_le_bytes());
    k[4..12].copy_from_slice(&mix64((v as u64) ^ 0x9e37_79b9_7f4a_7c15).to_le_bytes());
    k[12..16].copy_from_slice(&mix32(v ^ 0x85eb_ca6b).to_le_bytes());
    k
}

/// The value paired with index `i`, salted so `value_for(0)` isn't all-zero: a zero value
/// would be a fixed point for any benchmark chaining a loaded value back into its address.
#[inline(always)]
pub fn value_for<const VAL_LEN: usize>(i: u64) -> [u8; VAL_LEN] {
    let mut v = [0u8; VAL_LEN];
    let mut off = 0;
    let mut block = 0u64;
    while off < VAL_LEN {
        let chunk = mix64(i.wrapping_add(0xa076_1d64_78bd_642f) ^ (block << 32)).to_le_bytes();
        let n = core::cmp::min(8, VAL_LEN - off);
        v[off..off + n].copy_from_slice(&chunk[..n]);
        off += n;
        block += 1;
    }
    v
}

// Every size draws from one seeded stream where `distinct_u32_prefix(n)[..k] ==
// distinct_u32_prefix(k)`, so a size's keys don't shift as other sizes come and go.

/// Draws `u32`s from `ChaCha8Rng` seed `1`, keeping the first `n` distinct values in draw
/// order. Dedups with a flat bitset rather than a `HashSet`, for speed at large draws.
fn distinct_u32_prefix(n: usize) -> Vec<u32> {
    assert!(
        n as u64 <= 1u64 << 32,
        "{n} distinct u32s exceeds the u32 key space"
    );
    let mut seen = vec![0u64; (1usize << 32) / 64];
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let v = rng.next_u32();
        let word = (v >> 6) as usize;
        let bit = 1u64 << (v & 63);
        if seen[word] & bit == 0 {
            seen[word] |= bit;
            out.push(v);
        }
    }
    out
}

/// Keys, values and lookup order for one `BENCH_SIZES` entry.
pub struct SizeKeys<const KEY_LEN: usize, const VAL_LEN: usize> {
    /// The `size` keys inserted into the table under test.
    pub present: Vec<[u8; KEY_LEN]>,
    /// Values paired 1:1 with `present`.
    pub present_values: Vec<[u8; VAL_LEN]>,
    /// `size` keys guaranteed absent from any table built from `present`.
    pub absent: Vec<[u8; KEY_LEN]>,
    /// `present`, shuffled: decouples lookup order from insertion order while guaranteeing
    /// every lookup hits, for `get_existing`.
    pub lookup_order: Vec<[u8; KEY_LEN]>,
}

/// Seed for the mixed stream's shuffle, distinct from `lookup_order`'s 2.
const MIXED_SHUFFLE_SEED: u64 = 3;

impl<const KEY_LEN: usize, const VAL_LEN: usize> SizeKeys<KEY_LEN, VAL_LEN> {
    /// `encode` is the per-`KEY_LEN` helper turning a sampled `u32` into a `KEY_LEN`-byte
    /// key: `u32::to_ne_bytes` for 4/4, `expand_key` for 16/16.
    fn new(size: usize, encode: fn(u32) -> [u8; KEY_LEN]) -> Self {
        let stream = distinct_u32_prefix(2 * size);
        let present: Vec<[u8; KEY_LEN]> = stream[..size].iter().map(|&v| encode(v)).collect();
        let absent: Vec<[u8; KEY_LEN]> = stream[size..].iter().map(|&v| encode(v)).collect();
        let present_values = (0..size as u64).map(value_for::<VAL_LEN>).collect();

        let mut lookup_order = present.clone();
        lookup_order.shuffle(&mut ChaCha8Rng::seed_from_u64(2));

        Self {
            present,
            present_values,
            absent,
            lookup_order,
        }
    }

    /// A `size`-long query stream, half hits and half misses, shuffled so a lookup's
    /// outcome is unpredictable. Built on demand, not held for every cell at once.
    fn mixed_order(&self) -> Vec<[u8; KEY_LEN]> {
        let size = self.present.len();
        let hits = size / 2;
        let mut mixed: Vec<[u8; KEY_LEN]> = self.present[..hits]
            .iter()
            .chain(self.absent[..size - hits].iter())
            .copied()
            .collect();
        mixed.shuffle(&mut ChaCha8Rng::seed_from_u64(MIXED_SHUFFLE_SEED));
        mixed
    }
}

fn init_multitable<const KEY_LEN: usize, const VAL_LEN: usize>(
    size: usize,
    footprint: f64,
    keys: &SizeKeys<KEY_LEN, VAL_LEN>,
) -> (Table<KEY_LEN, VAL_LEN>, f32) {
    let mut mtable = with_utilization::<KEY_LEN, VAL_LEN>(size as u64, footprint);

    for i in 0..size {
        let key = keys.present[i];
        let value = keys.present_values[i];
        let res = mtable.insert(key, value);
        if res.is_err() {
            println!("Err: {:?}, key: {:?}", res, key);
        }
        assert!(res.is_ok(), "Shall succeed {:?}", i);
    }

    let key = keys.present[0];
    let value = keys.present_values[0];
    let res = mtable.insert(key, value);
    assert!(res.is_err(), "Duplicate key should not be inserted");

    for i in 0..size {
        let key = &keys.present[i];
        let value = &keys.present_values[i];

        assert_eq!(
            mtable.get(key),
            Some(value),
            "Inconsistent key#{i} {:?}",
            key
        );
    }

    for i in 0..size {
        let key = &keys.absent[i];
        assert_eq!(mtable.get(key), None, "Inconsistent {i}");
    }

    let byte_size = mtable.deep_size_of();
    let a = (size * (KEY_LEN + VAL_LEN)) as f32 / byte_size as f32;
    (mtable, a)
}

/// What the hashbrown comparison map is keyed on: `u32`/`u128` by default (FxHash's integer
/// path), or `[u8; KEY_LEN]` under `MT_BENCH_HB_KEY=array`, matching MultiTable's raw bytes.
pub trait HmapKey<const KEY_LEN: usize>: Hash + Eq + Copy {
    /// The benchmark's byte key as this map's key. On the timed loop's dependence chain,
    /// so every impl is `#[inline(always)]`.
    fn from_key(key: [u8; KEY_LEN]) -> Self;

    /// A distinct key per index, for the insert arm's untimed pre-touch. Byte-array keys
    /// mirror the integers they stand in for, so both modes touch the same bytes.
    fn from_index(i: u32) -> Self;
}

/// The suffix array mode puts on every group name.
const HB_ARRAY_SUFFIX: &str = "_hbarray";

impl HmapKey<4> for u32 {
    #[inline(always)]
    fn from_key(key: [u8; 4]) -> Self {
        u32::from_ne_bytes(key)
    }
    #[inline(always)]
    fn from_index(i: u32) -> Self {
        i
    }
}

impl HmapKey<16> for u128 {
    #[inline(always)]
    fn from_key(key: [u8; 16]) -> Self {
        u128::from_ne_bytes(key)
    }
    #[inline(always)]
    fn from_index(i: u32) -> Self {
        u128::from(i)
    }
}

impl HmapKey<4> for [u8; 4] {
    #[inline(always)]
    fn from_key(key: [u8; 4]) -> Self {
        key
    }
    #[inline(always)]
    fn from_index(i: u32) -> Self {
        i.to_ne_bytes()
    }
}

impl HmapKey<16> for [u8; 16] {
    #[inline(always)]
    fn from_key(key: [u8; 16]) -> Self {
        key
    }
    #[inline(always)]
    fn from_index(i: u32) -> Self {
        u128::from(i).to_ne_bytes()
    }
}

/// hashbrown comparison map for one size, keyed by `HK` via `HmapKey::from_key`, using the
/// same `FxBuildHasher` as MultiTable for a fair comparison.
fn init_hashmap<const KEY_LEN: usize, const VAL_LEN: usize, HK: HmapKey<KEY_LEN>>(
    size: usize,
    keys: &SizeKeys<KEY_LEN, VAL_LEN>,
) -> (HashBrownMap<HK, [u8; VAL_LEN], HasherBuilderUsed>, f32) {
    let mut hmap = HashBrownMap::with_capacity_and_hasher(size, HasherBuilderUsed::default());
    let array = hmap.allocation_size();
    for i in 0..size {
        hmap.insert(HK::from_key(keys.present[i]), keys.present_values[i]);
    }

    // Mirrors `init_multitable`'s warm-up, but not via `insert`, which reserves a slot
    // before looking and would double an array already filled to capacity.
    let duplicate = hmap.get_mut(&HK::from_key(keys.present[0]));
    assert!(
        duplicate.is_some(),
        "Duplicate key should already be present"
    );
    *duplicate.unwrap() = keys.present_values[0];
    assert_eq!(
        hmap.allocation_size(),
        array,
        "building the {size}-key map grew the array `with_capacity({size})` handed back"
    );

    for i in 0..size {
        assert_eq!(
            hmap.get(&HK::from_key(keys.present[i])),
            Some(&keys.present_values[i]),
            "Inconsistent key#{i}"
        );
    }

    for i in 0..size {
        assert_eq!(
            hmap.get(&HK::from_key(keys.absent[i])),
            None,
            "Inconsistent {i}"
        );
    }

    let byte_size = hmap.allocation_size();
    let a = (size * (KEY_LEN + VAL_LEN)) as f32 / byte_size as f32;
    (hmap, a)
}

/// Total slots (from `capacity`), slot load (`n / B`), and footprint LF (`n * pair_bytes /
/// allocation_size`) for `n` items, using hashbrown's own reported bytes, not a reconstruction.
fn realized_ruler<const KEY_LEN: usize, const VAL_LEN: usize>(
    n: usize,
    capacity: usize,
    allocation_size: usize,
) -> (usize, f64, f64) {
    let pair = (KEY_LEN + VAL_LEN) as f64;
    let buckets = (capacity + 1).next_power_of_two();
    (
        buckets,
        n as f64 / buckets as f64,
        n as f64 * pair / allocation_size as f64,
    )
}

/// Untimed, once per cell: checks the count lands on the anchor's bucket array and that a
/// real allocation realizes the derived footprint. Prints both rulers.
fn describe_cell<const KEY_LEN: usize, const VAL_LEN: usize, HK: HmapKey<KEY_LEN>>(
    cell: &LadderCell,
) {
    let hmap = HashBrownMap::<HK, [u8; VAL_LEN], HasherBuilderUsed>::with_capacity_and_hasher(
        cell.n,
        HasherBuilderUsed::default(),
    );
    let (realized, _, measured) =
        realized_ruler::<KEY_LEN, VAL_LEN>(cell.n, hmap.capacity(), hmap.allocation_size());
    assert_eq!(
        realized, cell.buckets,
        "n = {} lands on {realized} hashbrown buckets, not the anchor's {}",
        cell.n, cell.buckets
    );

    assert!(
        (measured - cell.footprint).abs() < 1e-3,
        "n = {} realizes footprint {measured:.6}, not the {:.6} it was derived from",
        cell.n,
        cell.footprint
    );
    if cell.mt_footprint.is_none() {
        println!(
            "cell {KEY_LEN}/{VAL_LEN}: n = {}, buckets = {}, slot load = {:.6}, \
             footprint LF = {measured:.6} (both structures)",
            cell.n, cell.buckets, cell.slot_load
        );
        return;
    }

    // MT-only cell: the map above only derives the count and is never benchmarked.
    if cell.mt_only {
        let mtable = with_utilization::<KEY_LEN, VAL_LEN>(cell.n as u64, cell.mt_lf());
        let mt_bytes = mtable.deep_size_of();
        let mt_lf = (cell.n * (KEY_LEN + VAL_LEN)) as f64 / mt_bytes as f64;
        let levels = mtable.level_bucket_cnts();
        println!(
            "MT-ONLY cell {KEY_LEN}/{VAL_LEN}: n = {} (the count the ladder picks at \
             {} buckets), no HashMap arm -- hashbrown of this width stops at \
             {:.6}\n  MultiTable: {mt_bytes} bytes ({:.1} KiB), footprint LF = \
             {mt_lf:.6} (requested {:.6}), sized depth = {} levels\n  \
             buckets per level = {levels:?}",
            cell.n,
            cell.buckets,
            max_footprint::<KEY_LEN, VAL_LEN>(),
            mt_bytes as f64 / 1024.0,
            cell.mt_lf(),
            levels.len(),
        );
        return;
    }

    // Demo cell: the two are sized apart on purpose, so print both
    // footprints and the ratio that is the whole claim.
    let alloc = hmap.allocation_size();
    let mt_bytes = with_utilization::<KEY_LEN, VAL_LEN>(cell.n as u64, cell.mt_lf()).deep_size_of();
    let mt_lf = (cell.n * (KEY_LEN + VAL_LEN)) as f64 / mt_bytes as f64;
    println!(
        "demo cell {KEY_LEN}/{VAL_LEN}: n = {}, buckets = {}, slot load = {:.6}\n  \
         HashMap:   {alloc} bytes ({:.1} KiB), footprint LF = {measured:.6}\n  \
         MultiTable: {mt_bytes} bytes ({:.1} KiB), footprint LF = {mt_lf:.6}\n  \
         footprint ratio = {:.3}x",
        cell.n,
        cell.buckets,
        cell.slot_load,
        alloc as f64 / 1024.0,
        mt_bytes as f64 / 1024.0,
        alloc as f64 / mt_bytes as f64,
    );
}

/// hashbrown's own realized numbers for the map `get_existing` actually benchmarks:
/// `buckets`/slot load/footprint LF from its measured `capacity()`, not theoretical values.
fn print_hashmap_load_factor<const KEY_LEN: usize, const VAL_LEN: usize, HK: HmapKey<KEY_LEN>>(
    size: usize,
    hmap: &HashBrownMap<HK, [u8; VAL_LEN], HasherBuilderUsed>,
) {
    let (buckets, slot_load, footprint) =
        realized_ruler::<KEY_LEN, VAL_LEN>(size, hmap.capacity(), hmap.allocation_size());
    println!(
        "HashMap (size {size}): buckets = {buckets}, slot load = {slot_load:.6}, footprint LF = {footprint:.6}"
    );
}

fn print_mtable_load_factor<const KEY_LEN: usize, const VAL_LEN: usize>(
    size: usize,
    mtable: &Table<KEY_LEN, VAL_LEN>,
) {
    let byte_size = mtable.deep_size_of();
    let a = (size * (KEY_LEN + VAL_LEN)) as f32 / byte_size as f32;
    let theoretical_a = size as f32 / (mtable.total_buckets() * BUCKET_SIZE) as f32;
    println!(
        "MultiTable (size {}): {} bytes, {} bytes per item, a = {a}, theoretical a = {theoretical_a}",
        size,
        byte_size,
        byte_size as f32 / size as f32
    );
}

/// What a build revealed about the cascade it filled.
struct CascadeShape {
    /// Levels that took at least one key -- the table's `first_empty_level`.
    depth: usize,
    /// Buckets on every level the cascade has, level 0 first, off the
    /// table's own level table. Whole list, never a prefix.
    buckets_per_level: Vec<usize>,
}

/// Where `insert` will put `key`, as `(level, bucket)`, off the same probe `insert` runs.
/// `None` means no room was found, which widens the table's frontier by one level.
#[cfg(not(mt_bench_plain))]
fn insert_landing<const KEY_LEN: usize, const VAL_LEN: usize>(
    mtable: &Table<KEY_LEN, VAL_LEN>,
    key: &[u8; KEY_LEN],
) -> Option<(usize, usize)> {
    match mtable.is_present::<false, true>(key) {
        SearchResult::NotFoundFirstEmpty(bucket, _, level, _) => Some((level, bucket as usize)),
        _ => None,
    }
}
#[cfg(mt_bench_plain)]
fn insert_landing<const KEY_LEN: usize, const VAL_LEN: usize>(
    mtable: &Table<KEY_LEN, VAL_LEN>,
    key: &[u8; KEY_LEN],
) -> Option<(usize, usize)> {
    match mtable.is_present::<false, true>(key) {
        PlainSearchResult::AbsentFirstEmpty(bucket, _, level) => Some((level, bucket as usize)),
        _ => None,
    }
}

/// Membership, the way the hot path means it: `GET_VALUE` off since a set has no value to
/// return, `FIND_FIRST_EMPTY` off, which is what separates a lookup from `insert`.
#[cfg(not(mt_bench_plain))]
#[inline(always)]
fn mt_is_present<const KEY_LEN: usize, const VAL_LEN: usize>(
    mtable: &Table<KEY_LEN, VAL_LEN>,
    key: &[u8; KEY_LEN],
) -> bool {
    matches!(
        mtable.is_present::<false, false>(key),
        SearchResult::Found(..)
    )
}
#[cfg(mt_bench_plain)]
#[inline(always)]
fn mt_is_present<const KEY_LEN: usize, const VAL_LEN: usize>(
    mtable: &Table<KEY_LEN, VAL_LEN>,
    key: &[u8; KEY_LEN],
) -> bool {
    matches!(
        mtable.is_present::<false, false>(key),
        PlainSearchResult::Present
    )
}

/// One timed MultiTable lookup: `get` where there are values, `is_present` at `VAL_LEN =
/// 0`. `VAL_LEN` is const, so one instantiation only ever compiles one of the two.
#[inline(always)]
fn probe<const KEY_LEN: usize, const VAL_LEN: usize>(
    mtable: &Table<KEY_LEN, VAL_LEN>,
    key: &[u8; KEY_LEN],
) {
    if VAL_LEN == 0 {
        black_box(mt_is_present(mtable, key));
    } else {
        black_box(mtable.get(key));
    }
}

/// Builds the cell's table as `init_multitable` does, minus the verification the benchmarked
/// build still runs, and records how deep the keys went.
fn build_and_probe<const KEY_LEN: usize, const VAL_LEN: usize>(
    size: usize,
    footprint: f64,
    keys: &SizeKeys<KEY_LEN, VAL_LEN>,
) -> (Table<KEY_LEN, VAL_LEN>, CascadeShape) {
    let mut mtable = with_utilization::<KEY_LEN, VAL_LEN>(size as u64, footprint);
    let mut depth = 1;

    for i in 0..size {
        match insert_landing(&mtable, &keys.present[i]) {
            Some((level, _)) => depth = depth.max(level + 1),
            // No room inside the frontier: `insert` widens it by one, and
            // that level has never been probed, so the key lands there.
            None => depth += 1,
        }
        let res = mtable.insert(keys.present[i], keys.present_values[i]);
        assert!(res.is_ok(), "Shall succeed {:?}", i);
    }

    let buckets_per_level = mtable.level_bucket_cnts();
    assert_eq!(
        buckets_per_level.iter().sum::<usize>(),
        mtable.total_buckets(),
        "the level table doesn't account for every bucket"
    );

    (
        mtable,
        CascadeShape {
            depth,
            buckets_per_level,
        },
    )
}

/// Untimed, once per cell: how far down the cascade the keys actually went, the ratio it
/// was sized with, and every level it has, not just the ones a replayed probe could show.
fn print_mtable_cascade(size: usize, shape: &CascadeShape) {
    println!(
        "MultiTable (size {size}): cascade depth = {} levels of {}, sizing ratio r = {}, \
         buckets per level = {:?}",
        shape.depth,
        shape.buckets_per_level.len(),
        sizing_ratio(),
        shape.buckets_per_level
    );
}

/// Forces every page backing `slice` resident during setup, not the timed loop, since
/// zero-fill vecs are lazily mapped. `write_volatile` keeps the touch from being optimized away.
fn touch_slice<T: Copy>(slice: &mut [T]) {
    for elem in slice.iter_mut() {
        unsafe { std::ptr::write_volatile(elem, *elem) };
    }
}

// The table under test, picked at *build* time: build.rs turns MT_BENCH_TABLE into
// `mt_bench_plain` and `Table` resolves to one concrete type -- no trait, no runtime branch.

#[cfg(mt_bench_plain)]
type Table<const KEY_LEN: usize, const VAL_LEN: usize> =
    MultiTablePlain<KEY_LEN, VAL_LEN, MAX_LEVELS, HasherBuilderUsed>;
#[cfg(not(mt_bench_plain))]
type Table<const KEY_LEN: usize, const VAL_LEN: usize> =
    MultiTableFiltered<KEY_LEN, VAL_LEN, MAX_LEVELS, HasherBuilderUsed>;

/// Slots per bucket of the table under test: each table reads its own default, and
/// `BUCKET_SIZE=<n>` at build time overrides both.
#[cfg(mt_bench_plain)]
const BUCKET_SIZE: usize = DEFAULT_BUCKET_SIZE;
#[cfg(not(mt_bench_plain))]
const BUCKET_SIZE: usize = DEFAULT_FILTERED_BUCKET_SIZE;

/// Group-name suffix, so a run of one table never writes the other's
/// criterion history.
#[cfg(mt_bench_plain)]
const TABLE_GROUP_SUFFIX: &str = "_plain";
#[cfg(not(mt_bench_plain))]
const TABLE_GROUP_SUFFIX: &str = "";

/// What `Table` resolved to, for messages that quote a per-table number --
/// `mt_max_footprint`'s ceiling above all, which differs between the two.
#[cfg(mt_bench_plain)]
const TABLE_KIND: &str = "plain";
#[cfg(not(mt_bench_plain))]
const TABLE_KIND: &str = "filtered";

/// Sized to a footprint load factor (payload bytes over allocated bytes), exactly what
/// `with_capacity` reads its utilization as, so a cell's step goes in unscaled.
fn with_utilization<const KEY_LEN: usize, const VAL_LEN: usize>(
    size: u64,
    footprint: f64,
) -> Table<KEY_LEN, VAL_LEN> {
    Table::with_capacity(size, footprint)
}

/// Faults the table's pages in during setup rather than in the timed loop. `filters` comes
/// from `alloc_zeroed` and needs the touch; `buckets_data` is hand-zeroed, already resident.
#[cfg(not(mt_bench_plain))]
fn pre_touch<const KEY_LEN: usize, const VAL_LEN: usize>(t: &mut Table<KEY_LEN, VAL_LEN>) {
    touch_slice(&mut t.filters);
    touch_slice(&mut t.buckets_data);
}
/// The plain table's buckets, likewise. They already arrive resident, from the
/// `vec![Default::default(); n]` clone loop that writes every one; touching them says so.
#[cfg(mt_bench_plain)]
fn pre_touch<const KEY_LEN: usize, const VAL_LEN: usize>(t: &mut Table<KEY_LEN, VAL_LEN>) {
    touch_slice(&mut t.bucket_data);
}

// hashbrown arms, one per operation and generic over the key type so both widths share them;
// `MT_BENCH_HMAP=1` adds them to a run.

/// `MT_BENCH_HMAP`: `1` runs the hashbrown comparison arms, `0` or unset skips them. The
/// ladder is still sized off hashbrown either way.
fn bench_hashbrown() -> bool {
    match std::env::var("MT_BENCH_HMAP").as_deref() {
        Ok("0") | Err(_) => false,
        Ok("1") => true,
        Ok(other) => panic!("MT_BENCH_HMAP must be 0 or 1, got `{other}`"),
    }
}

/// `MT_BENCH_HB_KEY`: `int` (default) hashes the byte key as its width's integer, the arm
/// every capture used; `array` hashes the raw bytes, as MultiTable does; each renames its group.
fn hmap_array_keys() -> bool {
    let array = match std::env::var("MT_BENCH_HB_KEY").as_deref() {
        Ok("int") | Err(_) => false,
        Ok("array") => true,
        Ok(other) => panic!("MT_BENCH_HB_KEY must be `int` or `array`, got `{other}`"),
    };
    if array {
        static BANNER: std::sync::Once = std::sync::Once::new();
        BANNER.call_once(|| {
            println!(
                "\n\
                 #############################################################\n\
                 ##  MT_BENCH_HB_KEY=array: the hashbrown arm is keyed on   ##\n\
                 ##  the raw [u8; KEY_LEN] byte keys, NOT the u32/u128 the  ##\n\
                 ##  published captures used. Every group name carries the  ##\n\
                 ##  `{HB_ARRAY_SUFFIX}` suffix, and these numbers are NOT           ##\n\
                 ##  comparable to a default-mode run.                      ##\n\
                 #############################################################\n"
            );
        });
    }
    array
}

/// What array mode adds to every group name, and nothing at all in the
/// default mode, so the published criterion IDs stay exactly as they were.
fn hb_group_suffix() -> &'static str {
    if hmap_array_keys() {
        HB_ARRAY_SUFFIX
    } else {
        ""
    }
}

/// One timed hashbrown lookup, mirroring `probe`'s split: `get` with values, else
/// `contains_key`, the call `HashSet::contains` forwards to.
#[inline(always)]
fn hmap_probe<const VAL_LEN: usize, HK: Hash + Eq>(
    hmap: &HashBrownMap<HK, [u8; VAL_LEN], HasherBuilderUsed>,
    key: HK,
) {
    if VAL_LEN == 0 {
        black_box(hmap.contains_key(&key));
    } else {
        black_box(hmap.get(&key));
    }
}

fn hmap_insert_arm<const KEY_LEN: usize, const VAL_LEN: usize, HK: HmapKey<KEY_LEN>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    size: usize,
    pairs: &[([u8; KEY_LEN], [u8; VAL_LEN])],
) {
    group.bench_with_input(BenchmarkId::new("HashMap", size), &size, |b, &size| {
        b.iter_with_setup(
            || {
                let mut hmap: HashBrownMap<HK, [u8; VAL_LEN], HasherBuilderUsed> =
                    HashBrownMap::with_capacity_and_hasher(size, HasherBuilderUsed::default());
                let array = hmap.allocation_size();
                // Pre-touch: bulk-insert then clear (capacity is
                // retained) so pages fault in during setup, not below.
                for i in 0..size {
                    hmap.insert(HK::from_index(i as u32), [0u8; VAL_LEN]);
                }
                // The timed loop must fill the array it was handed, never one it grew into:
                // at `MT_BENCH_SAT=1` that means filled to exactly capacity.
                assert_eq!(
                    hmap.allocation_size(),
                    array,
                    "{size} inserts grew the array `with_capacity({size})` handed back"
                );
                hmap.clear();
                hmap
            },
            |mut hmap| {
                // bytes -> HK is timed here, like MultiTable's own byte keys. Only the
                // result is `black_box`ed, which carries the anti-DCE.
                for (key, value) in pairs {
                    black_box(hmap.insert(HK::from_key(*key), *value));
                }
                hmap
            },
        );
    });
}

fn hmap_get_existing_arm<const KEY_LEN: usize, const VAL_LEN: usize, HK: HmapKey<KEY_LEN>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    size: usize,
    keys: &SizeKeys<KEY_LEN, VAL_LEN>,
) {
    print_hashmap_load_factor::<KEY_LEN, VAL_LEN, HK>(
        size,
        &init_hashmap::<KEY_LEN, VAL_LEN, HK>(size, keys).0,
    );

    group.bench_with_input(BenchmarkId::new("HashMap", size), &size, |b, &size| {
        let (hmap, _a) = init_hashmap::<KEY_LEN, VAL_LEN, HK>(size, keys);
        b.iter(|| {
            // bytes -> HK conversion timed alongside the lookup itself.
            for key in &keys.lookup_order {
                hmap_probe(&hmap, HK::from_key(*key));
            }
        });
    });
}

fn hmap_get_nonexisting_arm<const KEY_LEN: usize, const VAL_LEN: usize, HK: HmapKey<KEY_LEN>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    size: usize,
    keys: &SizeKeys<KEY_LEN, VAL_LEN>,
) {
    group.bench_with_input(BenchmarkId::new("HashMap", size), &size, |b, &size| {
        let (hmap, _a) = init_hashmap::<KEY_LEN, VAL_LEN, HK>(size, keys);
        b.iter(|| {
            // bytes -> HK conversion timed alongside the lookup itself.
            for key in &keys.absent {
                hmap_probe(&hmap, HK::from_key(*key));
            }
        });
    });
}

fn hmap_get_mixed_arm<const KEY_LEN: usize, const VAL_LEN: usize, HK: HmapKey<KEY_LEN>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    size: usize,
    keys: &SizeKeys<KEY_LEN, VAL_LEN>,
    mixed: &[[u8; KEY_LEN]],
) {
    group.bench_with_input(BenchmarkId::new("HashMap", size), &size, |b, &size| {
        let (hmap, _a) = init_hashmap::<KEY_LEN, VAL_LEN, HK>(size, keys);
        b.iter(|| {
            // bytes -> HK conversion timed alongside the lookup itself.
            for key in mixed {
                hmap_probe(&hmap, HK::from_key(*key));
            }
        });
    });
}

// Generic bench bodies, instantiated for KEY_LEN/VAL_LEN = 4/4 and 16/16 below.

fn bench_insert<const KEY_LEN: usize, const VAL_LEN: usize, HK: HmapKey<KEY_LEN>>(
    c: &mut Criterion,
    group_name: &str,
    encode: fn(u32) -> [u8; KEY_LEN],
) where
    [u8; KEY_LEN]: HmapKey<KEY_LEN>,
{
    let mut group = c.benchmark_group(format!(
        "{group_name}{TABLE_GROUP_SUFFIX}{}",
        hb_group_suffix()
    ));

    let run_hmap = bench_hashbrown();
    let array_keys = hmap_array_keys();
    for cell in bench_ladder::<KEY_LEN, VAL_LEN, HK>() {
        let size = cell.n;
        if size > 2usize.pow(21) {
            group.sample_size(10);
        }
        group.throughput(criterion::Throughput::Elements(size as u64));
        describe_cell::<KEY_LEN, VAL_LEN, HK>(&cell);

        let keys = SizeKeys::<KEY_LEN, VAL_LEN>::new(size, encode);

        let pairs = keys
            .present
            .iter()
            .copied()
            .zip(keys.present_values.iter().copied())
            .collect::<Vec<_>>();

        if RUN_MULTITABLE {
            let mt_id = cell.mt_arm_id();
            group.bench_with_input(BenchmarkId::new(mt_id, size), &size, |b, &size| {
                b.iter_with_setup(
                    || {
                        let mut mtable =
                            with_utilization::<KEY_LEN, VAL_LEN>(size as u64, cell.mt_lf());
                        pre_touch(&mut mtable);
                        mtable
                    },
                    |mut mtable| {
                        for pair in &pairs {
                            let (key, value) = pair;
                            let _ = black_box(mtable.insert(*key, *value));
                        }
                        // Returned instead of dropped, so criterion's own accounting pays
                        // the drop cost.
                        mtable
                    },
                );
            });
        }

        if run_hmap {
            if array_keys {
                hmap_insert_arm::<KEY_LEN, VAL_LEN, [u8; KEY_LEN]>(&mut group, size, &pairs);
            } else {
                hmap_insert_arm::<KEY_LEN, VAL_LEN, HK>(&mut group, size, &pairs);
            }
        }
    }

    group.finish();
}

fn bench_get_existing<const KEY_LEN: usize, const VAL_LEN: usize, HK: HmapKey<KEY_LEN>>(
    c: &mut Criterion,
    group_name: &str,
    encode: fn(u32) -> [u8; KEY_LEN],
) where
    [u8; KEY_LEN]: HmapKey<KEY_LEN>,
{
    let mut group = c.benchmark_group(format!(
        "{group_name}{TABLE_GROUP_SUFFIX}{}",
        hb_group_suffix()
    ));

    let run_hmap = bench_hashbrown();
    let array_keys = hmap_array_keys();
    for cell in bench_ladder::<KEY_LEN, VAL_LEN, HK>() {
        let size = cell.n;
        if size > 2usize.pow(21) {
            group.sample_size(10);
        }
        group.throughput(criterion::Throughput::Elements(size as u64));
        describe_cell::<KEY_LEN, VAL_LEN, HK>(&cell);

        let keys = SizeKeys::<KEY_LEN, VAL_LEN>::new(size, encode);

        if RUN_MULTITABLE {
            {
                // Scoped so this table is gone before the arm builds its own.
                let (mtable, shape) =
                    build_and_probe::<KEY_LEN, VAL_LEN>(size, cell.mt_lf(), &keys);
                print_mtable_load_factor(size, &mtable);
                print_mtable_cascade(size, &shape);
            }

            let mt_id = cell.mt_arm_id();
            group.bench_with_input(BenchmarkId::new(mt_id, size), &size, |b, &size| {
                let (mtable, _) = init_multitable::<KEY_LEN, VAL_LEN>(size, cell.mt_lf(), &keys);

                b.iter(|| {
                    for key in &keys.lookup_order {
                        probe(&mtable, key);
                    }
                });
            });
        }

        if run_hmap {
            if array_keys {
                hmap_get_existing_arm::<KEY_LEN, VAL_LEN, [u8; KEY_LEN]>(&mut group, size, &keys);
            } else {
                hmap_get_existing_arm::<KEY_LEN, VAL_LEN, HK>(&mut group, size, &keys);
            }
        }
    }

    group.finish();
}

fn bench_get_nonexisting<const KEY_LEN: usize, const VAL_LEN: usize, HK: HmapKey<KEY_LEN>>(
    c: &mut Criterion,
    group_name: &str,
    encode: fn(u32) -> [u8; KEY_LEN],
) where
    [u8; KEY_LEN]: HmapKey<KEY_LEN>,
{
    let mut group = c.benchmark_group(format!(
        "{group_name}{TABLE_GROUP_SUFFIX}{}",
        hb_group_suffix()
    ));

    let run_hmap = bench_hashbrown();
    let array_keys = hmap_array_keys();
    for cell in bench_ladder::<KEY_LEN, VAL_LEN, HK>() {
        let size = cell.n;
        if size > 2usize.pow(21) {
            group.sample_size(10);
        }
        group.throughput(criterion::Throughput::Elements(size as u64));
        describe_cell::<KEY_LEN, VAL_LEN, HK>(&cell);

        let keys = SizeKeys::<KEY_LEN, VAL_LEN>::new(size, encode);

        if RUN_MULTITABLE {
            let mt_id = cell.mt_arm_id();
            group.bench_with_input(BenchmarkId::new(mt_id, size), &size, |b, &size| {
                let (mtable, _) = init_multitable::<KEY_LEN, VAL_LEN>(size, cell.mt_lf(), &keys);

                b.iter(|| {
                    for key in &keys.absent {
                        probe(&mtable, key);
                    }
                });
            });
        }

        if run_hmap {
            if array_keys {
                hmap_get_nonexisting_arm::<KEY_LEN, VAL_LEN, [u8; KEY_LEN]>(
                    &mut group, size, &keys,
                );
            } else {
                hmap_get_nonexisting_arm::<KEY_LEN, VAL_LEN, HK>(&mut group, size, &keys);
            }
        }
    }

    group.finish();
}

/// The 50/50 group: the same cells as `get_existing`/`get_nonexisting`, queried with a
/// stream half in the table and half not, in random order.
fn bench_get_mixed<const KEY_LEN: usize, const VAL_LEN: usize, HK: HmapKey<KEY_LEN>>(
    c: &mut Criterion,
    group_name: &str,
    encode: fn(u32) -> [u8; KEY_LEN],
) where
    [u8; KEY_LEN]: HmapKey<KEY_LEN>,
{
    let mut group = c.benchmark_group(format!(
        "{group_name}{TABLE_GROUP_SUFFIX}{}",
        hb_group_suffix()
    ));

    let run_hmap = bench_hashbrown();
    let array_keys = hmap_array_keys();
    for cell in bench_ladder::<KEY_LEN, VAL_LEN, HK>() {
        let size = cell.n;
        if size > 2usize.pow(21) {
            group.sample_size(10);
        }
        group.throughput(criterion::Throughput::Elements(size as u64));
        describe_cell::<KEY_LEN, VAL_LEN, HK>(&cell);

        let keys = SizeKeys::<KEY_LEN, VAL_LEN>::new(size, encode);
        let mixed = keys.mixed_order();

        if RUN_MULTITABLE {
            let mt_id = cell.mt_arm_id();
            group.bench_with_input(BenchmarkId::new(mt_id, size), &size, |b, &size| {
                let (mtable, _) = init_multitable::<KEY_LEN, VAL_LEN>(size, cell.mt_lf(), &keys);

                b.iter(|| {
                    for key in &mixed {
                        probe(&mtable, key);
                    }
                });
            });
        }

        if run_hmap {
            if array_keys {
                hmap_get_mixed_arm::<KEY_LEN, VAL_LEN, [u8; KEY_LEN]>(
                    &mut group, size, &keys, &mixed,
                );
            } else {
                hmap_get_mixed_arm::<KEY_LEN, VAL_LEN, HK>(&mut group, size, &keys, &mixed);
            }
        }
    }

    group.finish();
}

// Top-level instantiations, fixing KEY_LEN/VAL_LEN and the group name per width: 4/4 keeps
// the original names for saved baselines, 16/16 gets its own; `HK` is the width's integer key.

fn bench_insert_4(c: &mut Criterion) {
    let (n, e) = ("insert", u32::to_ne_bytes as fn(u32) -> [u8; 4]);
    bench_insert::<4, 4, u32>(c, n, e);
}

fn bench_get_existing_4(c: &mut Criterion) {
    let (n, e) = ("get_existing", u32::to_ne_bytes as fn(u32) -> [u8; 4]);
    bench_get_existing::<4, 4, u32>(c, n, e);
}

fn bench_get_nonexisting_4(c: &mut Criterion) {
    let (n, e) = ("get_nonexisting", u32::to_ne_bytes as fn(u32) -> [u8; 4]);
    bench_get_nonexisting::<4, 4, u32>(c, n, e);
}

fn bench_get_mixed_4(c: &mut Criterion) {
    let (n, e) = ("get_mixed", u32::to_ne_bytes as fn(u32) -> [u8; 4]);
    bench_get_mixed::<4, 4, u32>(c, n, e);
}

fn bench_insert_kv16(c: &mut Criterion) {
    let (n, e) = ("insert_kv16", expand_key as fn(u32) -> [u8; 16]);
    bench_insert::<16, 16, u128>(c, n, e);
}

fn bench_get_existing_kv16(c: &mut Criterion) {
    let (n, e) = ("get_existing_kv16", expand_key as fn(u32) -> [u8; 16]);
    bench_get_existing::<16, 16, u128>(c, n, e);
}

fn bench_get_nonexisting_kv16(c: &mut Criterion) {
    let (n, e) = ("get_nonexisting_kv16", expand_key as fn(u32) -> [u8; 16]);
    bench_get_nonexisting::<16, 16, u128>(c, n, e);
}

fn bench_get_mixed_kv16(c: &mut Criterion) {
    let (n, e) = ("get_mixed_kv16", expand_key as fn(u32) -> [u8; 16]);
    bench_get_mixed::<16, 16, u128>(c, n, e);
}

// The keys-only shapes: same bench bodies, key streams and ladder machinery, with `VAL_LEN = 0`
// turning `get` into `is_present` on one arm and into `contains_key` on the other.

fn bench_insert_set4(c: &mut Criterion) {
    let (n, e) = ("insert_set4", u32::to_ne_bytes as fn(u32) -> [u8; 4]);
    bench_insert::<4, 0, u32>(c, n, e);
}

fn bench_get_existing_set4(c: &mut Criterion) {
    let (n, e) = ("get_existing_set4", u32::to_ne_bytes as fn(u32) -> [u8; 4]);
    bench_get_existing::<4, 0, u32>(c, n, e);
}

fn bench_get_nonexisting_set4(c: &mut Criterion) {
    let (n, e) = (
        "get_nonexisting_set4",
        u32::to_ne_bytes as fn(u32) -> [u8; 4],
    );
    bench_get_nonexisting::<4, 0, u32>(c, n, e);
}

fn bench_get_mixed_set4(c: &mut Criterion) {
    let (n, e) = ("get_mixed_set4", u32::to_ne_bytes as fn(u32) -> [u8; 4]);
    bench_get_mixed::<4, 0, u32>(c, n, e);
}

fn bench_insert_set16(c: &mut Criterion) {
    let (n, e) = ("insert_set16", expand_key as fn(u32) -> [u8; 16]);
    bench_insert::<16, 0, u128>(c, n, e);
}

fn bench_get_existing_set16(c: &mut Criterion) {
    let (n, e) = ("get_existing_set16", expand_key as fn(u32) -> [u8; 16]);
    bench_get_existing::<16, 0, u128>(c, n, e);
}

fn bench_get_nonexisting_set16(c: &mut Criterion) {
    let (n, e) = ("get_nonexisting_set16", expand_key as fn(u32) -> [u8; 16]);
    bench_get_nonexisting::<16, 0, u128>(c, n, e);
}

fn bench_get_mixed_set16(c: &mut Criterion) {
    let (n, e) = ("get_mixed_set16", expand_key as fn(u32) -> [u8; 16]);
    bench_get_mixed::<16, 0, u128>(c, n, e);
}

use blake2::{digest::Mac, Blake2b512, Blake2bMac512, Digest};

// A Blake2b `BuildHasher`, unkeyed.

pub struct Blake2bHasher {
    state: Blake2b512,
}

impl Default for Blake2bHasher {
    fn default() -> Self {
        Self {
            state: Blake2b512::new(),
        }
    }
}

impl Hasher for Blake2bHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        Digest::update(&mut self.state, bytes);
    }

    #[inline]
    fn finish(&self) -> u64 {
        // finalize() consumes self, so clone the running state
        let digest = self.state.clone().finalize();
        u64::from_le_bytes(digest[..8].try_into().unwrap())
    }
}

#[derive(Clone, Default)]
pub struct Blake2bBuilder;

impl BuildHasher for Blake2bBuilder {
    type Hasher = Blake2bHasher;

    fn build_hasher(&self) -> Self::Hasher {
        Blake2bHasher::default()
    }
}

// The keyed (Blake2b-MAC) counterpart, for the HashMap arms.

pub struct Blake2bKeyedHasher {
    state: Blake2bMac512,
}

impl Hasher for Blake2bKeyedHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        Mac::update(&mut self.state, bytes);
    }

    #[inline]
    fn finish(&self) -> u64 {
        let tag = self.state.clone().finalize().into_bytes();
        u64::from_le_bytes(tag[..8].try_into().unwrap())
    }
}

#[derive(Clone)]
pub struct Blake2bKeyedBuilder {
    key: [u8; 64],
}

impl Blake2bKeyedBuilder {
    /// Construct with an explicit 64-byte key.
    pub fn with_key(key: [u8; 64]) -> Self {
        Self { key }
    }

    /// Construct with a cryptographically random key.
    pub fn random() -> Self {
        let key: [u8; 64] = rand::random();
        Self { key }
    }
}

impl Default for Blake2bKeyedBuilder {
    fn default() -> Self {
        Self::random()
    }
}

impl BuildHasher for Blake2bKeyedBuilder {
    type Hasher = Blake2bKeyedHasher;

    fn build_hasher(&self) -> Self::Hasher {
        Blake2bKeyedHasher {
            // 64-byte key is always valid; unwrap is safe
            state: Blake2bMac512::new_from_slice(&self.key).unwrap(),
        }
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
    bench_insert_4,
    bench_get_existing_4,
    bench_get_nonexisting_4,
    bench_get_mixed_4,
    bench_insert_kv16,
    bench_get_existing_kv16,
    bench_get_nonexisting_kv16,
    bench_get_mixed_kv16,
    bench_insert_set4,
    bench_get_existing_set4,
    bench_get_nonexisting_set4,
    bench_get_mixed_set4,
    bench_insert_set16,
    bench_get_existing_set16,
    bench_get_nonexisting_set16,
    bench_get_mixed_set16,
);
criterion_main!(benches);
