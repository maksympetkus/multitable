// The filtered table, `include!`d into lib.rs so its items keep crate-root paths.

/// One bucket's key/value pairs, interleaved so a hit reads its key and value together.
pub type BucketPairs<
    const KEY_LEN: usize,
    const VAL_LEN: usize,
    const BUCKET_SIZE: usize = DEFAULT_FILTERED_BUCKET_SIZE,
> = [([u8; KEY_LEN], [u8; VAL_LEN]); BUCKET_SIZE];

/// A level's bucket count, or under `powers-of-2` the right shift that addresses the level.
type LevelWidth = u32;

/// A bucket count as the stored width `32 - log2(cnt)`, applied to a widened `u64`: 32 and an
/// absent level's 33 are real shifts landing on index 0 rather than sentinels.
#[cfg(feature = "powers-of-2")]
const fn level_width(bucket_cnt: BucketCnts) -> LevelWidth {
    bucket_cnt.leading_zeros() + 1
}

#[cfg(feature = "powers-of-2")]
const fn level_bucket_cnt(width: LevelWidth) -> BucketCnts {
    if width > 32 {
        0
    } else {
        1 << (32 - width)
    }
}

/// Every level is `map_rand_bytes`-addressed here, so the width is the count.
#[cfg(not(feature = "powers-of-2"))]
const fn level_width(bucket_cnt: BucketCnts) -> LevelWidth {
    bucket_cnt
}

#[cfg(not(feature = "powers-of-2"))]
const fn level_bucket_cnt(width: LevelWidth) -> BucketCnts {
    width
}

/// Why [`MultiTableFiltered::insert`] or [`MultiTableFiltered::upsert`] refused a pair. Either
/// way the table is left as it was: nothing is written, and the frontier has not moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertError {
    /// `insert` found the key already stored; `upsert` writes over it instead.
    AlreadyPresent,
    /// The key's bucket has no room on any level the cascade has, so there is nowhere to put it.
    /// The table is sized up front and never grows.
    Full,
}

impl std::fmt::Display for InsertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            InsertError::AlreadyPresent => "key already present in the multitable",
            InsertError::Full => "no space to insert: the key's bucket is full on every level",
        })
    }
}

impl std::error::Error for InsertError {}

/// The filtered table: a filter byte per slot, so a level is ruled out with no key read.
/// `MultiTable`'s doc has the `BUCKET_SIZE`/`SIMD_ARITY` contract, enforced here too.
pub struct MultiTableFiltered<
    const KEY_LEN: usize,
    const VAL_LEN: usize,
    const MAX_LEVELS: usize = 8usize,
    H = HasherBuilder,
    const BUCKET_SIZE: usize = DEFAULT_FILTERED_BUCKET_SIZE,
    const SIMD_ARITY: usize = DEFAULT_FILTERED_SIMD_ARITY,
> {
    /// Per level, its offset and its [`LevelWidth`], in an array so it sits on the stack.
    levels_meta: [(LevelWidth, LevelOffset); MAX_LEVELS],
    /// Every bucket's key/value pairs, interleaved so a hit reads one line.
    pub buckets_data: BucketStore<BucketPairs<KEY_LEN, VAL_LEN, BUCKET_SIZE>>,
    /// One filter byte per slot, plus a spare bucket so a `SIMD_ARITY`-wide read stays in bounds.
    pub filters: Vec<[u8; BUCKET_SIZE]>,
    hasher: H,
    first_empty_level: usize,
    /// The frontier the plain insert walk stops at: a copy of `first_empty_level` until a
    /// delete leaves room behind, and zero from then on, which turns that walk's level-0 stop
    /// into the handover to the recording walk. A real frontier is never zero, so the walk
    /// reads both meanings off the one load it already made, and every delete has to zero this
    /// or the room it leaves is never offered. A byte because a table has at most 50 levels,
    /// and because a word grew the struct and cost 9% on the 4-byte insert bench.
    plain_insert_frontier: u8,
    #[cfg(feature = "first-power")]
    first_shift: u8,
}

impl<
        const KEY_LEN: usize,
        const VAL_LEN: usize,
        const MAX_LEVELS: usize,
        H: Sized,
        const BUCKET_SIZE: usize,
        const SIMD_ARITY: usize,
    > DeepSizeOf for MultiTableFiltered<KEY_LEN, VAL_LEN, MAX_LEVELS, H, BUCKET_SIZE, SIMD_ARITY>
{
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.levels_meta.deep_size_of_children(context)
            + self.buckets_data.len() * std::mem::size_of::<BucketPairs<KEY_LEN, VAL_LEN, BUCKET_SIZE>>()
            // The Vec header and the hasher already sit in `size_of::<Self>()`, which
            // `DeepSizeOf::deep_size_of` adds on top of this.
            + self.filters.capacity() * std::mem::size_of::<[u8; BUCKET_SIZE]>()
    }
}

/// Whether the last filter byte is below `MIN_KEY_FILTER`, so empty or tombstoned. Read as the top
/// byte of the window's last word, which tests as one shift against zero rather than a byte mask.
#[inline(always)]
fn last_slot_below_keys<const BUCKET_SIZE: usize, const SIMD_ARITY: usize>(
    window: &[u8; SIMD_ARITY],
) -> bool {
    if BUCKET_SIZE >= 8 {
        let word = u64::from_le_bytes(window[BUCKET_SIZE - 8..BUCKET_SIZE].try_into().unwrap());
        word < (MIN_KEY_FILTER as u64) << 56
    } else {
        window[BUCKET_SIZE - 1] < MIN_KEY_FILTER
    }
}

/// Identity through a general register: pins a reduced scan mask to the scalar it lives in, so a
/// later test cannot be re-derived from the compare vector. Not `pure`, so it cannot be hoisted.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn pin_to_scalar(mask: u32) -> u32 {
    let mut pinned = mask;
    unsafe {
        std::arch::asm!(
            "/* {0:w} */",
            inout(reg) pinned,
            options(nomem, nostack, preserves_flags),
        )
    };
    pinned
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn pin_to_scalar(mask: u32) -> u32 {
    mask
}

impl<
        const KEY_LEN: usize,
        const VAL_LEN: usize,
        const MAX_LEVELS: usize,
        H: BuildHasher + Default,
        const BUCKET_SIZE: usize,
        const SIMD_ARITY: usize,
    > MultiTableFiltered<KEY_LEN, VAL_LEN, MAX_LEVELS, H, BUCKET_SIZE, SIMD_ARITY>
{
    /// `MultiTable::CONFIG_OK` plus a 32-slot cap: the filter scans return their mask as a `u32`.
    const CONFIG_OK: () = assert!(
        BUCKET_SIZE <= SIMD_ARITY
            && SIMD_ARITY <= 2 * BUCKET_SIZE
            && matches!(SIMD_ARITY, 8 | 16 | 32 | 64)
            && BUCKET_SIZE <= 32
            && SIMD_ARITY <= 32,
        "SIMD_ARITY must be one of 8/16/32/64 and satisfy BUCKET_SIZE <= SIMD_ARITY <= 2 * BUCKET_SIZE; \
         additionally the filtered table caps BUCKET_SIZE and SIMD_ARITY at 32, because its filter \
         scans return match masks as u32 (find_all_simd_u8_overread's `to_bitmask() as u32`) and a \
         64-lane scan would silently truncate every match in slot 32 and above"
    );

    /// A cascade with `bucket_cnts` buckets per level, zeroed, with every filter byte empty.
    pub fn new_from_cnts(bucket_cnts: Vec<u32>) -> Self {
        let () = Self::CONFIG_OK;

        assert!(MAX_LEVELS <= 50, "At most 50 levels are supported");

        assert!(
            bucket_cnts.len() <= MAX_LEVELS,
            "More than {MAX_LEVELS} levels isn't supported, got {}",
            bucket_cnts.len()
        );

        assert!(
            !bucket_cnts.is_empty(),
            "A cascade needs at least one level, but the level list is empty"
        );
        assert!(
            bucket_cnts.iter().all(|&cnt| cnt >= 1),
            "Every level needs at least one bucket, got {bucket_cnts:?}"
        );
        let total_buckets: u64 = bucket_cnts.iter().map(|&cnt| cnt as u64).sum();
        assert!(
            total_buckets <= u32::MAX as u64,
            "The cascade's {total_buckets} buckets overflow the u32 that numbers a bucket"
        );

        #[cfg(feature = "powers-of-2")]
        for (level, &cnt) in bucket_cnts.iter().enumerate() {
            assert!(
                cnt.is_power_of_two(),
                "`powers-of-2` shift-addresses every level, but level {level} has {cnt} buckets"
            );
        }

        #[cfg(feature = "first-power")]
        assert!(
            bucket_cnts[0].is_power_of_two(),
            "`first-power` addresses level 0 by a shift, but it has {} buckets",
            bucket_cnts[0]
        );

        let level_offset: Vec<u32> = once(&0u32)
            .chain(bucket_cnts.iter().take(bucket_cnts.len() - 1))
            .scan(0, |sum, &x| {
                *sum += x;
                Some(*sum)
            })
            .collect();

        let total_cnt = bucket_cnts.iter().sum::<u32>() as usize;

        #[cfg(feature = "first-power")]
        let first_shift = (bucket_cnts[0].leading_zeros() + 1) as u8;

        #[cfg(not(feature = "powers-of-2"))]
        let levels_meta: Vec<_> = bucket_cnts.into_iter().zip(level_offset).collect();
        #[cfg(feature = "powers-of-2")]
        let levels_meta: Vec<_> = bucket_cnts
            .into_iter()
            .map(level_width)
            .zip(level_offset)
            .collect();

        let table = Self {
            // A level the cascade lacks pads to `level_width(0)`: a zero count, or shift 33.
            levels_meta: std::array::from_fn(|i| {
                levels_meta.get(i).copied().unwrap_or((level_width(0), 0))
            }),
            // The spare bucket keeps `filter_window`'s SIMD_ARITY-wide read inside the allocation.
            filters: vec![[EMPTY_FILTER; BUCKET_SIZE]; total_cnt + 1],
            // SAFETY: a bucket is byte arrays throughout, so all-zero is the empty bucket.
            buckets_data: unsafe { BucketStore::new_zeroed(total_cnt) },
            #[cfg(feature = "first-power")]
            first_shift,
            first_empty_level: 1,
            plain_insert_frontier: 1,
            hasher: H::default(),
        };

        advise_hugepages(&table.buckets_data);
        advise_hugepages(&table.filters);

        table
    }

    /// The paper's levels computation; `powers-of-2` rounds every level, `first-power` the first.
    pub fn with_capacity(capacity: u64, utilization: f64) -> Self {
        let () = Self::CONFIG_OK;

        assert!(
            utilization > 0.0 && utilization < 1.0,
            "The utilization must lie in (0, 1), got {utilization}"
        );

        let meta_size_const = size_of::<Self>();
        let pair_size = KEY_LEN + VAL_LEN;

        // The bucket budget the load factor derives from, once the table's own bytes are paid for.
        let payload = (capacity as usize * pair_size) as f64 - meta_size_const as f64 * utilization;
        assert!(
            payload > 0.0,
            "A capacity of {capacity} is too small at utilization {utilization}: the table's own \
             {meta_size_const} bytes take the whole budget"
        );

        let a = (capacity as f64 * (BUCKET_SIZE + pair_size * BUCKET_SIZE) as f64 * utilization)
            / (BUCKET_SIZE as f64 * payload);
        let a = f64::min(a, 0.997);

        #[cfg(feature = "agile")]
        let bucket_cnts = sizing::bucket_cnts(capacity, BUCKET_SIZE as u32, a, MAX_LEVELS);
        #[cfg(feature = "powers-of-2")]
        let bucket_cnts =
            bucket_cnts_for_utilization_powers(capacity, BUCKET_SIZE as u32, a, MAX_LEVELS);
        #[cfg(feature = "first-power")]
        let bucket_cnts =
            bucket_cnts_for_utilization_first_power(capacity, BUCKET_SIZE as u32, a, MAX_LEVELS);

        Self::new_from_cnts(bucket_cnts)
    }

    #[cfg_attr(not(feature = "profiling"), inline(always))]
    #[cfg_attr(feature = "profiling", inline(never))]
    fn filter_window(&self, bucket_idx: usize) -> &[u8; SIMD_ARITY] {
        // SAFETY: the spare bucket and `SIMD_ARITY <= 2 * BUCKET_SIZE` keep this read in bounds.
        unsafe {
            &*self
                .filters
                .as_ptr()
                .cast::<u8>()
                .add(bucket_idx * BUCKET_SIZE)
                .cast::<[u8; SIMD_ARITY]>()
        }
    }

    /// Unpacks the walk's one-register record of a bucket with room. Re-deriving the slot is sound
    /// because the record was made under `available_lanes != 0` on a filter the walk never writes.
    /// Inlined at each stop site: the out-of-line cold call measured slower on the host.
    #[inline(always)]
    fn recorded_first_available(&self, packed: u64, key_filter: u8) -> SearchResult<'_, VAL_LEN> {
        let bucket_num = packed as BucketNumber;
        let available =
            available_lanes::<BUCKET_SIZE, SIMD_ARITY>(self.filter_window(bucket_num as usize));
        SearchResult::NotFoundFirstEmpty(
            bucket_num,
            unsafe { cttz_nonzero(available) },
            (packed >> 32) as Level,
            key_filter,
        )
    }

    /// Buckets across the whole cascade, every level together.
    pub fn total_buckets(&self) -> usize {
        self.buckets_data.len()
    }

    /// Buckets per level, level 0 first, read off the level table.
    pub fn level_bucket_cnts(&self) -> Vec<usize> {
        self.levels_meta
            .iter()
            .map(|&(width, _)| level_bucket_cnt(width) as usize)
            .take_while(|&cnt| cnt != 0)
            .collect()
    }

    /// Whether the cascade has this level at all, off the padding `new_from_cnts` leaves.
    #[inline(always)]
    fn has_level(&self, level: usize) -> bool {
        // Callers check `first_empty_level < MAX_LEVELS` first, so `level` indexes the array.
        let width = unsafe { self.levels_meta.get_unchecked(level) }.0;
        level_bucket_cnt(width) != 0
    }

    /// The bucket `hash` selects on `level`: with power-of-two counts the multiply becomes a shift.
    #[cfg(feature = "powers-of-2")]
    #[cfg_attr(not(feature = "profiling"), inline(always))]
    #[cfg_attr(feature = "profiling", inline(never))]
    fn level_bucket(&self, level: usize, hash: u32) -> u32 {
        let (shift, offset) = self.levels_meta[level];
        offset + ((hash as u64) >> shift) as u32
    }

    /// Adds a pair the table does not hold yet, onto the shallowest available slot of its chain;
    /// `Err(AlreadyPresent)` on a duplicate and `Err(Full)` when the chain has no room left, with
    /// the table unchanged in both cases.
    #[cfg_attr(not(feature = "profiling"), inline)]
    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn insert(&mut self, key: [u8; KEY_LEN], value: [u8; VAL_LEN]) -> Result<(), InsertError> {
        // The plain walk, the one this table had before tombstones: it costs no record per
        // bucket, and its level-0 stop hands the key to `grow_search_depth_and_insert` once a
        // delete has left room, which is where the recording walk takes over.
        self.insert_walking::<false>(key, value)
    }

    /// The insert body over either probe; `RECORD` picks the walk that offers a delete's room.
    #[cfg_attr(not(feature = "profiling"), inline(always))]
    #[cfg_attr(feature = "profiling", inline(never))]
    fn insert_walking<const RECORD: bool>(
        &mut self,
        key: [u8; KEY_LEN],
        value: [u8; VAL_LEN],
    ) -> Result<(), InsertError> {
        let found = if RECORD {
            self.is_present_recording::<false, true>(&key)
        } else {
            self.is_present::<false, true>(&key)
        };

        if let SearchResult::NotFoundFirstEmpty(bucket_num, next_spot, _level, key_filter) = found {
            let next_spot = next_spot as usize;
            // The recording walk names a bucket it read and a slot it found available in it.
            let meta = unsafe { self.buckets_data.get_unchecked_mut(bucket_num as usize) };

            let data = unsafe { meta.get_unchecked_mut(next_spot) };
            *data = (key, value);
            *unsafe {
                self.filters
                    .get_unchecked_mut(bucket_num as usize)
                    .get_unchecked_mut(next_spot)
            } = key_filter;

            Ok(())
        } else {
            if found == SearchResult::NotFound {
                return self.grow_search_depth_and_insert::<RECORD>(key, value);
            }

            // The spelled-out `return` stays: the tail-expression form re-lays out `insert`.
            #[allow(clippy::needless_return)]
            return Err(InsertError::AlreadyPresent);
        }
    }

    /// Widens `first_empty_level` by one and retries, stopping at the first absent level; and,
    /// off the plain walk's level-0 stop, the handover to the recording walk once a delete has
    /// left room. Cold either way, so the room test costs the inserts nothing.
    ///
    /// `Full` writes nothing and leaves the frontier where it was: a level this widens into has
    /// never held a key, so the retry always finds room there, and `Full` is only reached with
    /// the cascade's last level already inside the frontier.
    #[cold]
    #[inline(never)]
    fn grow_search_depth_and_insert<const RECORD: bool>(
        &mut self,
        key: [u8; KEY_LEN],
        value: [u8; VAL_LEN],
    ) -> Result<(), InsertError> {
        if !RECORD && self.plain_insert_frontier == 0 {
            return self.insert_walking::<true>(key, value);
        }

        if self.first_empty_level < MAX_LEVELS && self.has_level(self.first_empty_level) {
            self.first_empty_level += 1;
            // Zero stays zero: once a delete has handed the inserts over, they stay handed over.
            if self.plain_insert_frontier != 0 {
                self.plain_insert_frontier = self.first_empty_level as u8;
            }
            return self.insert_walking::<RECORD>(key, value);
        }

        Err(InsertError::Full)
    }

    /// Adds the pair onto the shallowest available slot of its chain, or writes the value over
    /// the one stored under `key`. `Err(Full)` when a new key's chain has no room left, with the
    /// table unchanged; it never answers `AlreadyPresent`.
    #[cfg_attr(not(feature = "profiling"), inline)]
    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn upsert(&mut self, key: [u8; KEY_LEN], value: [u8; VAL_LEN]) -> Result<(), InsertError> {
        // Spelled out here rather than left to the walk's level-0 stop as in `insert`: an
        // upsert of a key the table already holds has to reach the walk, not the stop.
        if self.plain_insert_frontier == 0 {
            return self.upsert_walking::<true>(key, value);
        }

        self.upsert_walking::<false>(key, value)
    }

    /// The upsert body over either probe, as `insert_walking` is for `insert`.
    #[cfg_attr(not(feature = "profiling"), inline(always))]
    #[cfg_attr(feature = "profiling", inline(never))]
    fn upsert_walking<const RECORD: bool>(
        &mut self,
        key: [u8; KEY_LEN],
        value: [u8; VAL_LEN],
    ) -> Result<(), InsertError> {
        let found = if RECORD {
            self.is_present_recording::<false, true>(&key)
        } else {
            self.is_present::<false, true>(&key)
        };

        match found {
            SearchResult::NotFoundFirstEmpty(bucket_num, next_spot, _, key_filter) => {
                let next_spot = next_spot as usize;
                // The recording walk names a bucket it read and a slot it found available in it.
                let meta = unsafe { self.buckets_data.get_unchecked_mut(bucket_num as usize) };

                let data = unsafe { meta.get_unchecked_mut(next_spot) };
                *data = (key, value);
                *unsafe {
                    self.filters
                        .get_unchecked_mut(bucket_num as usize)
                        .get_unchecked_mut(next_spot)
                } = key_filter;
            }
            SearchResult::Found(bucket_num, index) => {
                // `Found` names the bucket and the slot the walk just compared this key in.
                let value_cell = unsafe {
                    &mut self
                        .buckets_data
                        .get_unchecked_mut(bucket_num as usize)
                        .get_unchecked_mut(index as usize)
                        .1
                };
                *value_cell = value;
            }
            SearchResult::NotFound => {
                return self.grow_search_depth_and_insert::<RECORD>(key, value)
            }
            _ => unreachable!(),
        };

        Ok(())
    }

    /// Walks the cascade for `key`: `FIND_FIRST_EMPTY` reports a free slot, `GET_VALUE` the value.
    /// Under `FIND_FIRST_EMPTY` this is the plain insert probe, so it answers `NotFound` at once
    /// once a delete has left room; `is_present_recording` is the walk that offers that room.
    #[cfg_attr(not(feature = "profiling"), inline(always))]
    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn is_present<const GET_VALUE: bool, const FIND_FIRST_EMPTY: bool>(
        &self,
        key: &[u8; KEY_LEN],
    ) -> SearchResult<'_, VAL_LEN> {
        self.walk::<GET_VALUE, FIND_FIRST_EMPTY, false>(key)
    }

    /// The inserts' probe: `is_present` with the walk recording the shallowest available slot of
    /// the chain, a zero hole or a tombstoned last slot, so a delete's room is taken again.
    #[cfg_attr(not(feature = "profiling"), inline(always))]
    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn is_present_recording<const GET_VALUE: bool, const FIND_FIRST_EMPTY: bool>(
        &self,
        key: &[u8; KEY_LEN],
    ) -> SearchResult<'_, VAL_LEN> {
        self.walk::<GET_VALUE, FIND_FIRST_EMPTY, true>(key)
    }

    /// The frontier this walk stops at. The plain insert walk reads its own copy, which a
    /// delete zeroes to hand the key over at level 0; every other walk wants the real one.
    #[inline(always)]
    fn walk_frontier<const FIND_FIRST_EMPTY: bool, const RECORD_FIRST: bool>(&self) -> usize {
        if FIND_FIRST_EMPTY && !RECORD_FIRST {
            self.plain_insert_frontier as usize
        } else {
            self.first_empty_level
        }
    }

    /// The cascade walk behind both probes; `RECORD_FIRST` serves the inserts, compiling in the
    /// record of the shallowest room, and stays out of the lookups.
    #[cfg_attr(not(feature = "profiling"), inline(always))]
    #[cfg_attr(feature = "profiling", inline(never))]
    fn walk<const GET_VALUE: bool, const FIND_FIRST_EMPTY: bool, const RECORD_FIRST: bool>(
        &self,
        key: &[u8; KEY_LEN],
    ) -> SearchResult<'_, VAL_LEN> {
        // The record-first walk's memory: the shallowest bucket seen with room and its level
        // packed into one register (`level << 32 | bucket`, `u64::MAX` for nothing yet).
        let mut first_available_packed = u64::MAX;

        let next_hash_raw = hash_key_v3(&self.hasher, key);

        // A `max` against `MIN_KEY_FILTER`, written to keep the same dependency depth.
        let key_filter = (next_hash_raw as u8).saturating_sub(MIN_KEY_FILTER) + MIN_KEY_FILTER;

        // Level 0's bucket up front; `read_next_meta` fills each deeper one a level ahead.
        let mut buckets_chosen = [0u32; MAX_LEVELS];
        #[cfg(feature = "agile")]
        buckets_chosen[0..1]
            .copy_from_slice(&[first_level_bucket(next_hash_raw, self.levels_meta[0].0)]);
        // The shift alone: level 0's offset is always zero, so `level_bucket` would only add it.
        #[cfg(feature = "powers-of-2")]
        buckets_chosen[0..1].copy_from_slice(&[
            ((first_hash_selector(next_hash_raw) as u64) >> self.levels_meta[0].0) as u32
        ]);
        // Only level 0 is a power of two here, so only it is shift-addressed.
        #[cfg(feature = "first-power")]
        buckets_chosen[0..1].copy_from_slice(&[
            ((first_hash_selector(next_hash_raw) as u64) >> self.first_shift) as u32
        ]);

        macro_rules! loop_body {
            ($level:expr) => {
                // A frontier is never zero, so this is never taken, but it hoists the frontier
                // load into the prologue under the hash; without it the inserts pay the load at
                // their first test. It is also where the plain insert walk gives up once a
                // delete has zeroed its copy: no test of its own, which measured 13 to 29% on
                // the insert bench, and `insert`'s cold grow path takes the key from here to
                // the recording walk.
                if $level == 0
                    && FIND_FIRST_EMPTY
                    && unlikely($level == self.walk_frontier::<FIND_FIRST_EMPTY, RECORD_FIRST>())
                {
                    return SearchResult::NotFound;
                }

                // Worked out before this level was reached, by the level above or the seed.
                let bucket_num = buckets_chosen[$level];

                // No `#[inline]`: expression attributes need nightly, and LLVM inlines it anyway.
                let mut read_next_meta = || {
                    if  $level + 1 != MAX_LEVELS {
                        let hash = level_hash_selector(next_hash_raw, $level + 1);

                        #[cfg(not(feature = "powers-of-2"))]
                        let next_bucket = self.levels_meta[$level+1].1 + map_rand_bytes(hash, self.levels_meta[$level+1].0);
                        #[cfg(feature = "powers-of-2")]
                        let next_bucket = self.level_bucket($level+1, hash);

                        buckets_chosen[$level+1] = next_bucket;
                    }

                };

                let bucket_idx = bucket_num as usize;

                let bucket_filter = self.filter_window(bucket_idx);

                // The stop: an empty last byte, written only once the bucket fills, says nothing of
                // the chain sits deeper (`filtered_fill_is_contiguous`); a tombstone there is room only.
                macro_rules! record_or_stop {
                    () => {
                        if FIND_FIRST_EMPTY {
                            let scanned =
                                find_all_simd_u8_overread::<BUCKET_SIZE, SIMD_ARITY, false>(bucket_filter, EMPTY_FILTER);
                            let first_empty_found = if RECORD_FIRST {
                                pin_to_scalar(scanned)
                            } else {
                                scanned
                            };
                            if likely(first_empty_found & (1u32 << (BUCKET_SIZE - 1)) != 0) {
                                // The last byte never returns to empty, so nothing spilled past.
                                if RECORD_FIRST && unlikely(first_available_packed != u64::MAX) {
                                    return self.recorded_first_available(first_available_packed, key_filter);
                                }
                                // No over-read lane sits this low, so the bit is our bucket's.
                                return SearchResult::NotFoundFirstEmpty(
                                    bucket_num,
                                    unsafe { cttz_nonzero(first_empty_found) },
                                    $level,
                                    key_filter
                                );
                            }
                            // Room without the stop only happens once a delete holes a bucket;
                            // keep the first such spot, since only the stop rules out a duplicate.
                            if RECORD_FIRST {
                                let mut in_bucket_empties = first_empty_found;
                                if SIMD_ARITY > BUCKET_SIZE {
                                    in_bucket_empties &= u32::MAX >> (32 - BUCKET_SIZE);
                                }
                                // A deleted last slot reads `TOMBSTONE_FILTER`, never zero, so it
                                // is room here without being the stop.
                                let last_below_keys =
                                    last_slot_below_keys::<BUCKET_SIZE, SIMD_ARITY>(bucket_filter);
                                let room = in_bucket_empties
                                    | (last_below_keys as u32) << (BUCKET_SIZE - 1);
                                if unlikely(room != 0) {
                                    if first_available_packed == u64::MAX {
                                        first_available_packed =
                                            (($level as u64) << 32) | bucket_num as u64;
                                    }
                                }
                            }
                        }
                    };
                }

                // A lookup only asks whether the mask is zero, so it early-outs there.
                let mut key_filter_found = if FIND_FIRST_EMPTY {
                    find_all_simd_u8_overread::<BUCKET_SIZE, SIMD_ARITY, false>(bucket_filter, key_filter)
                } else {
                    find_all_simd_u8_overread::<BUCKET_SIZE, SIMD_ARITY, true>(bucket_filter, key_filter)
                };

                // `buckets_chosen` holds a level offset plus an index inside that level's buckets.
                let bucket_data = unsafe { self.buckets_data.get_unchecked(bucket_idx) };

                if (key_filter_found != 0) {
                    // Lanes past the last slot name the next bucket's, so they are masked off here.
                    if SIMD_ARITY > BUCKET_SIZE {
                        key_filter_found &= u32::MAX >> (32 - BUCKET_SIZE);
                    }
                    prefetch_l1(bucket_data);

                    // A neighbour-only match masks to nothing, leaving no candidate here.
                    if !(SIMD_ARITY > BUCKET_SIZE && key_filter_found == 0) {
                        if FIND_FIRST_EMPTY {
                            loop {
                                let index = unsafe { cttz_nonzero(key_filter_found) };

                                // In bounds: the mask still names a lane of this bucket.
                                let (filtered_key, value) =
                                    unsafe { &bucket_data.get_unchecked(index as usize) };

                                if likely(index < BUCKET_SIZE as u32 && fast_eq(filtered_key, key)) {
                                    if GET_VALUE {
                                        return SearchResult::FoundValue(value);
                                    }
                                    return SearchResult::Found(bucket_num, index);
                                }

                                key_filter_found &= key_filter_found - 1;
                                if likely(key_filter_found == 0) { break; }
                            }
                        } else {
                            // A lookup's first candidate comes off a min-index reduction. Two
                            // loops on purpose: one loop picking its first index by mode grows both walks.
                            let mut index =
                                find_first_simd_u8_overread::<SIMD_ARITY>(bucket_filter, key_filter);
                            loop {
                                // In bounds: the mask still names a lane of this bucket.
                                let (filtered_key, value) =
                                    unsafe { &bucket_data.get_unchecked(index as usize) };

                                if likely(index < BUCKET_SIZE as u32 && fast_eq(filtered_key, key)) {
                                    if GET_VALUE {
                                        return SearchResult::FoundValue(value);
                                    }
                                    return SearchResult::Found(bucket_num, index);
                                }

                                key_filter_found &= key_filter_found - 1;
                                if likely(key_filter_found == 0) { break; }
                                index = unsafe { cttz_nonzero(key_filter_found) };
                            }
                        }
                    }

                    read_next_meta();
                    record_or_stop!();
                }
                else {
                    // Not hoisted past the `if`: merging the two tails re-lays out the whole walk.
                    read_next_meta();
                    record_or_stop!();
                }

                // The frontier, once the level is done: nothing sits at or past `first_empty_level`.
                // A lookup usually ends at level 0, so that answer is its warm path and every
                // deeper level lies cold; anywhere else reaching the frontier is the rare case.
                if $level == 0 && !FIND_FIRST_EMPTY {
                    if likely($level + 1 == self.first_empty_level) {
                        return SearchResult::NotFound;
                    }
                } else if unlikely(
                    $level + 1 == self.walk_frontier::<FIND_FIRST_EMPTY, RECORD_FIRST>(),
                ) {
                    if RECORD_FIRST
                        && FIND_FIRST_EMPTY
                        && unlikely(first_available_packed != u64::MAX)
                    {
                        return self.recorded_first_available(first_available_packed, key_filter);
                    }
                    return SearchResult::NotFound;
                }
        }
        }

        // Unrolled walk; `L < MAX_LEVELS` const-folds a shorter instantiation's tail levels.
        seq! {L in 0..50 {
            if L < MAX_LEVELS {
                loop_body!(L);
            }
        }};
        // The last level's frontier answer again, as the walk runs off the far end only with the
        // frontier at MAX_LEVELS. Folding it into that level's check re-lays out a lookup walk.
        if RECORD_FIRST && FIND_FIRST_EMPTY && unlikely(first_available_packed != u64::MAX) {
            return self.recorded_first_available(first_available_packed, key_filter);
        }
        SearchResult::NotFound
    }

    /// Takes `key` out of the table and hands back the value it held, or `None` when the table
    /// does not hold it.
    ///
    /// The freed slot's filter byte is the whole record of the delete: `EMPTY_FILTER` anywhere
    /// but a bucket's last slot, and `TOMBSTONE_FILTER` in a last slot, where zero would say the
    /// bucket never filled and stop a probe short of the chain behind it. Zeroing
    /// `plain_insert_frontier` is the other half: it puts the inserts on the recording walk,
    /// which is the only one that offers the room this leaves.
    #[inline(never)]
    pub fn remove(&mut self, key: &[u8; KEY_LEN]) -> Option<[u8; VAL_LEN]> {
        let SearchResult::Found(bucket_num, index) = self.is_present::<false, false>(key) else {
            return None;
        };
        let (bucket_num, index) = (bucket_num as usize, index as usize);

        // `Found` names the bucket and the slot the walk just compared this key in.
        let value = unsafe {
            self.buckets_data
                .get_unchecked(bucket_num)
                .get_unchecked(index)
                .1
        };

        // The key and value bytes stay as they are: the filter byte alone guards the slot, since
        // neither byte it can hold is ever a key's, so a wipe would dirty a second line for
        // nothing. The bytes are unreachable until the next insert writes over them.
        *unsafe {
            self.filters
                .get_unchecked_mut(bucket_num)
                .get_unchecked_mut(index)
        } = if index == BUCKET_SIZE - 1 {
            TOMBSTONE_FILTER
        } else {
            EMPTY_FILTER
        };
        self.plain_insert_frontier = 0;

        Some(value)
    }

    /// The value stored under `key`, or `None` when the walk does not find it.
    #[cfg_attr(not(feature = "profiling"), inline(always))]
    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn get(&self, key: &[u8; KEY_LEN]) -> Option<&[u8; VAL_LEN]> {
        use SearchResult::*;

        if let FoundValue(value) = self.is_present::<true, false>(key) {
            Some(value)
        } else {
            None
        }
    }
}
