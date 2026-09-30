// The plain table, `MultiTable`, `include!`d into lib.rs so its items keep crate-root paths.

/// What a [`MultiTable`] search found; `GET_VALUE` picks between the `Present` and `Found` arms.
#[derive(PartialEq)]
pub enum PlainSearchResult<'a, const VAL_LEN: usize> {
    /// Not in the table, and no visited bucket had room.
    Absent,
    /// Not in the table; the first bucket with a free slot, that slot's index and its level.
    AbsentFirstEmpty(BucketNumber, KeyIndex, Level),
    /// In the table. Only produced when `GET_VALUE` is off.
    Present,
    /// In the table, with its value. Only produced when `GET_VALUE` is on.
    Found(&'a [u8; VAL_LEN]),
}
/// The plain table: a key takes the first bucket with room as it walks the level cascade.
/// `CONFIG_OK` holds `SIMD_ARITY`, one of 8/16/32/64, to `BUCKET_SIZE ..= 2 * BUCKET_SIZE`.
pub struct MultiTable<
    const KEY_LEN: usize,
    const VAL_LEN: usize,
    const MAX_LEVELS: usize = 8,
    H = HasherBuilder,
    const BUCKET_SIZE: usize = DEFAULT_BUCKET_SIZE,
    const SIMD_ARITY: usize = DEFAULT_SIMD_ARITY,
> {
    levels_meta: [(BucketCnts, LevelOffset); MAX_LEVELS],
    first_level_cnts: BucketCnts,
    /// How deep a lookup may walk: nothing was ever stored at or past this level.
    first_empty_level: usize,
    /// The bucket storage itself, public so a caller can touch it (the benchmark's pre-touch).
    pub bucket_data: BucketStore<Bucket<KEY_LEN, VAL_LEN, BUCKET_SIZE>>,
    next_spot: Vec<u8>,
    hasher_builder: H,
}

impl<
        const KEY_LEN: usize,
        const VAL_LEN: usize,
        const MAX_LEVELS: usize,
        const BUCKET_SIZE: usize,
        const SIMD_ARITY: usize,
    > DeepSizeOf
    for MultiTable<KEY_LEN, VAL_LEN, MAX_LEVELS, HasherBuilder, BUCKET_SIZE, SIMD_ARITY>
{
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.bucket_data.deep_size_of_children(context)
            + self.next_spot.deep_size_of_children(context)
    }
}

impl<
        const KEY_LEN: usize,
        const VAL_LEN: usize,
        const MAX_LEVELS: usize,
        const BUCKET_SIZE: usize,
        const SIMD_ARITY: usize,
    > MultiTable<KEY_LEN, VAL_LEN, MAX_LEVELS, HasherBuilder, BUCKET_SIZE, SIMD_ARITY>
{
    /// The struct's `BUCKET_SIZE`/`SIMD_ARITY` contract; referenced from every constructor.
    const CONFIG_OK: () = assert!(
        BUCKET_SIZE <= SIMD_ARITY
            && SIMD_ARITY <= 2 * BUCKET_SIZE
            && matches!(SIMD_ARITY, 8 | 16 | 32 | 64),
        "SIMD_ARITY must be one of 8/16/32/64 and satisfy BUCKET_SIZE <= SIMD_ARITY <= 2 * BUCKET_SIZE"
    );

    /// A cascade with `bucket_cnts` buckets per level, level 0 first, zeroed and empty.
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

        let level_offset: Vec<u32> = once(&0u32)
            .chain(bucket_cnts.iter().take(bucket_cnts.len() - 1))
            .scan(0, |sum, &x| {
                *sum += x;
                Some(*sum)
            })
            .collect();

        let total_cnt = bucket_cnts.iter().sum::<u32>() as usize;

        let first_level_cnts = bucket_cnts[0];
        let levels_meta: Vec<_> = bucket_cnts.into_iter().zip(level_offset).collect();

        let table = Self {
            first_level_cnts,
            first_empty_level: 1,
            levels_meta: std::array::from_fn(|i|
                // Level 0 sits in `first_level_cnts`, so this array starts at level 1.
                levels_meta.get(i+1).cloned().unwrap_or_default()),
            // SAFETY: a bucket is byte arrays throughout, so all-zero is the empty bucket.
            bucket_data: unsafe { BucketStore::new_zeroed(total_cnt) },
            next_spot: vec![0u8; total_cnt],
            hasher_builder: HasherBuilder::default(),
        };

        advise_hugepages(&table.bucket_data);
        advise_hugepages(&table.next_spot);

        table
    }

    /// The paper's levels computation, the same under every variant here. `utilization` is
    /// payload bytes over bytes allocated, which fixes the slot load factor the sizing takes.
    pub fn with_capacity(capacity: u64, utilization: f64) -> Self {
        let () = Self::CONFIG_OK;

        assert!(
            utilization > 0.0 && utilization < 1.0,
            "The utilization must lie in (0, 1), got {utilization}"
        );

        let meta_size_const = std::mem::size_of::<Self>();
        let pair_size = KEY_LEN + VAL_LEN;

        // The bucket budget the load factor derives from, once the table's own bytes are paid for.
        let payload = (capacity as usize * pair_size) as f64 - meta_size_const as f64 * utilization;
        assert!(
            payload > 0.0,
            "A capacity of {capacity} is too small at utilization {utilization}: the table's own \
             {meta_size_const} bytes take the whole budget"
        );

        let a = (capacity as f64 * (1 + pair_size * BUCKET_SIZE) as f64 * utilization)
            / (BUCKET_SIZE as f64 * payload);
        let a = f64::min(a, 0.999999);

        let bucket_cnts = sizing::bucket_cnts(capacity, BUCKET_SIZE as u32, a, MAX_LEVELS);
        Self::new_from_cnts(bucket_cnts)
    }

    /// Buckets across the whole cascade, every level together.
    pub fn total_buckets(&self) -> usize {
        self.bucket_data.len()
    }

    /// Buckets per level, level 0 first, read off the level table.
    pub fn level_bucket_cnts(&self) -> Vec<usize> {
        once(self.first_level_cnts)
            .chain(self.levels_meta.iter().map(|&(cnt, _)| cnt))
            .map(|cnt| cnt as usize)
            .take_while(|&cnt| cnt != 0)
            .collect()
    }

    /// Keys held in `bucket_idx`, for occupancy measurement outside the crate.
    pub fn bucket_fill(&self, bucket_idx: usize) -> usize {
        self.next_spot[bucket_idx] as usize
    }

    /// Adds a pair the table does not hold yet; `Err` on a duplicate or a cascade with no room.
    #[cfg_attr(not(feature = "profiling"), inline)]
    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn insert(&mut self, key: [u8; KEY_LEN], value: [u8; VAL_LEN]) -> Result<(), &str> {
        let (bucket_num, next_spot, level) = match self.is_present::<false, true>(&key) {
            PlainSearchResult::AbsentFirstEmpty(bucket_num, next_spot, level) => {
                (bucket_num as usize, next_spot as usize, level)
            }
            PlainSearchResult::Absent => return self.grow_search_depth_and_insert(key, value),
            _ => return Err("Key already present in the multitable"),
        };

        // The probe rested on the frontier, so widen it unless the cascade has no such level.
        if unlikely(level >= self.first_empty_level) {
            if self.level_cnt(level) == 0 {
                return Err("No space in the bucket, even on the last level");
            }
            self.first_empty_level = level + 1;
        }

        // `is_present` only reports a bucket it dereferenced and a slot it found free in it.
        let bucket = unsafe { self.bucket_data.get_unchecked_mut(bucket_num) };
        *unsafe { self.next_spot.get_unchecked_mut(bucket_num) } += 1;
        bucket.keys[next_spot] = key;
        bucket.values[next_spot] = value;

        Ok(())
    }

    /// Buckets on `level`; zero means the cascade has no such level.
    #[inline(always)]
    fn level_cnt(&self, level: usize) -> BucketCnts {
        if level == 0 {
            self.first_level_cnts
        } else {
            // Every caller reaches here with 1 <= level < MAX_LEVELS, so level - 1 is an index.
            unsafe { self.levels_meta.get_unchecked(level - 1) }.0
        }
    }

    /// Widens `first_empty_level` by one and retries, never past the first level the cascade lacks.
    #[cold]
    #[inline(never)]
    fn grow_search_depth_and_insert(
        &mut self,
        key: [u8; KEY_LEN],
        value: [u8; VAL_LEN],
    ) -> Result<(), &str> {
        if self.first_empty_level < MAX_LEVELS && self.level_cnt(self.first_empty_level) != 0 {
            self.first_empty_level += 1;
            return self.insert(key, value);
        }

        Err("No space in the bucket, even on the last level")
    }

    /// Walks the cascade for `key`: `FIND_FIRST_EMPTY` reports a free slot, `GET_VALUE` the value.
    #[cfg_attr(not(feature = "profiling"), inline)]
    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn is_present<const GET_VALUE: bool, const FIND_FIRST_EMPTY: bool>(
        &self,
        key: &[u8; KEY_LEN],
    ) -> PlainSearchResult<'_, VAL_LEN> {
        let next_hash_raw = hash_key_v3(&self.hasher_builder, key);

        let first_hash = next_hash_raw as u32;

        let level_bucket = |level: usize, key_hash: u32| {
            // Only called as `level_bucket($level + 1, ..)` under `$level + 1 < MAX_LEVELS`.
            let (bucket_cnts, level_offset) = *unsafe { self.levels_meta.get_unchecked(level - 1) };
            level_offset + map_rand_bytes(key_hash, bucket_cnts)
        };

        // Level 0's bucket up front; each level works out the next, so its address is ready.
        let mut buckets_chosen = [0u32; MAX_LEVELS];
        buckets_chosen[0] = map_rand_bytes(first_hash, self.first_level_cnts);

        macro_rules! probe_level {
            ($level:expr) => {
                // Only a lookup needs the frontier: insert stops at the first bucket with room.
                if !FIND_FIRST_EMPTY && unlikely($level == self.first_empty_level) {
                    return PlainSearchResult::Absent;
                }

                let bucket_num = buckets_chosen[$level];
                // `buckets_chosen` holds a level offset plus an index inside that level, so it
                // addresses a bucket of the one allocation, which `next_spot` matches in length.
                let bucket = unsafe { self.bucket_data.get_unchecked(bucket_num as usize) };
                let next_spot =
                    *unsafe { self.next_spot.get_unchecked(bucket_num as usize) } as usize;
                let has_room = next_spot != BUCKET_SIZE;

                if $level + 1 < MAX_LEVELS {
                    buckets_chosen[$level + 1] =
                        level_bucket($level + 1, level_hash_selector(next_hash_raw, $level + 1));
                }

                if FIND_FIRST_EMPTY && unlikely(next_spot == 0) {
                    return PlainSearchResult::AbsentFirstEmpty(bucket_num, 0, $level);
                }

                let key_index_found = find_simd::<_, BUCKET_SIZE>(&bucket.keys, key, next_spot);

                if likely(key_index_found.is_some()) {
                    let index = key_index_found.unwrap();
                    if GET_VALUE {
                        // The key scan only looks at the first `next_spot` slots and only
                        // returns an index below that, so it names a slot of this bucket.
                        return PlainSearchResult::Found(unsafe {
                            bucket.values.get_unchecked(index)
                        });
                    }
                    return PlainSearchResult::Present;
                }

                // Stopping at the first bucket with room needs `next_spot` never to decrease.
                if FIND_FIRST_EMPTY && has_room {
                    return PlainSearchResult::AbsentFirstEmpty(
                        bucket_num,
                        next_spot as KeyIndex,
                        $level,
                    );
                }
            };
        }

        // Unrolled so each `buckets_chosen` index is a constant and the array stays in
        // registers; the allow covers the last level's lookahead, which nothing reads.
        #[allow(unused_assignments)]
        if MAX_LEVELS <= 16 {
            seq! {L in 0..16 {
                if L < MAX_LEVELS {
                    probe_level!(L);
                }
            }};
        } else {
            seq! {L in 0..50 {
                if L < MAX_LEVELS {
                    probe_level!(L);
                }
            }};
        }

        PlainSearchResult::Absent
    }

    /// The value stored under `key`, or `None` when the walk does not find it.
    #[cfg_attr(not(feature = "profiling"), inline)]
    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn get(&self, key: &[u8; KEY_LEN]) -> Option<&[u8; VAL_LEN]> {
        match self.is_present::<true, false>(key) {
            PlainSearchResult::Found(value) => Some(value),
            _ => None,
        }
    }
}
