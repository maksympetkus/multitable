#[cfg(test)]
mod paper_sizing_tests {
    use crate::*;

    // Large enough to build a real multi-level cascade under the paper
    // sizing, small enough that the round trip stays well under a second.
    const N: u64 = 50_000;
    const UTILIZATION: f64 = 0.75;

    #[test]
    fn multi_table_paper_sizing_round_trips() {
        let mut table = MultiTable::<4, 4>::with_capacity(N, UTILIZATION);

        for i in 0..N as u32 {
            let key = i.to_le_bytes();
            table
                .insert(key, key)
                .unwrap_or_else(|e| panic!("insert {i} failed: {e}"));
        }
        for i in 0..N as u32 {
            let key = i.to_le_bytes();
            assert!(table.get(&key).is_some(), "missing key {i}");
        }
        for i in (N as u32)..(N as u32 + 1_000) {
            let key = i.to_le_bytes();
            assert!(
                table.get(&key).is_none(),
                "disjoint key {i} unexpectedly present"
            );
        }

        let paper_load = N as f64 / (table.total_buckets() * DEFAULT_BUCKET_SIZE) as f64;
        println!("MultiTable load factor: paper = {paper_load:.4}");
        assert!(paper_load > 0.0 && paper_load < 1.0);
    }

    /// `powers-of-2` rounds the cascade to the grid here; `first-power` rounds only level one.
    #[test]
    fn multi_table_filtered_simple_paper_sizing_round_trips() {
        let mut table = MultiTableFiltered::<4, 4>::with_capacity(N, UTILIZATION);

        for i in 0..N as u32 {
            let key = i.to_le_bytes();
            table
                .insert(key, key)
                .unwrap_or_else(|e| panic!("insert {i} failed: {e}"));
        }
        for i in 0..N as u32 {
            let key = i.to_le_bytes();
            assert!(table.get(&key).is_some(), "missing key {i}");
        }
        for i in (N as u32)..(N as u32 + 1_000) {
            let key = i.to_le_bytes();
            assert!(
                table.get(&key).is_none(),
                "disjoint key {i} unexpectedly present"
            );
        }

        let paper_load = N as f64 / (table.total_buckets() * DEFAULT_FILTERED_BUCKET_SIZE) as f64;
        println!("MultiTableFiltered load factor: paper = {paper_load:.4}");
        assert!(paper_load > 0.0 && paper_load < 1.0);
    }

    #[cfg(feature = "first-power")]
    #[test]
    fn the_paper_sizing_rounds_only_the_first_level_under_first_power() {
        let table = MultiTableFiltered::<4, 4>::with_capacity(N, UTILIZATION);
        let cnts = table.level_bucket_cnts();
        assert!(
            cnts[0].is_power_of_two(),
            "paper sizing starts with {} buckets, not a power of two: {cnts:?}",
            cnts[0]
        );
        assert!(
            cnts[1..].iter().any(|n| !n.is_power_of_two()),
            "every level is a power of two, so nothing was left free: {cnts:?}"
        );
    }
}

/// Coverage for `MultiTableFiltered::is_present`'s deep-level `loop_body!` path
/// (`MAX_LEVELS > 8`).
#[cfg(test)]
mod deep_levels_tests {
    use crate::*;

    const KEY_LEN: usize = 8;
    const VAL_LEN: usize = 8;

    /// A bijective scramble of `i` (an odd multiplier makes it a `u64` permutation), so
    /// disjoint `i` ranges give disjoint keys.
    fn key_of(i: u64) -> [u8; KEY_LEN] {
        i.wrapping_mul(0x9E37_79B9_7F4A_7C15).to_ne_bytes()
    }

    #[cfg(feature = "agile")]
    fn geometric_cnts(levels: usize, first: u32, ratio: f64) -> Vec<u32> {
        let mut cnts = Vec::with_capacity(levels);
        let mut cur = first as f64;
        for _ in 0..levels {
            cnts.push((cur.round() as u32).max(1));
            cur *= ratio;
        }
        cnts
    }

    /// Keys held per level, counted from filter bytes: a stored key's byte is never
    /// `EMPTY_FILTER`.
    fn level_occupancy<const MAX_LEVELS: usize>(
        table: &MultiTableFiltered<KEY_LEN, VAL_LEN, MAX_LEVELS, HasherBuilder>,
    ) -> Vec<usize> {
        (0..MAX_LEVELS)
            .map(|level| {
                let (width, offset) = table.levels_meta[level];
                let cnt = level_bucket_cnt(width);
                (offset as usize..(offset + cnt) as usize)
                    .map(|bucket| {
                        table.filters[bucket]
                            .iter()
                            .filter(|&&f| f != EMPTY_FILTER)
                            .count()
                    })
                    .sum::<usize>()
            })
            .collect()
    }

    fn slots(cnts: &[u32]) -> u64 {
        cnts.iter().sum::<u32>() as u64 * DEFAULT_FILTERED_BUCKET_SIZE as u64
    }

    fn fill_and_check<const MAX_LEVELS: usize>(
        table: &mut MultiTableFiltered<KEY_LEN, VAL_LEN, MAX_LEVELS, HasherBuilder>,
        keys: u64,
    ) -> Vec<usize> {
        for i in 0..keys {
            let key = key_of(i);
            table
                .insert(key, key)
                .unwrap_or_else(|e| panic!("insert {i} of {keys} failed at {MAX_LEVELS} levels: {e}"));
        }

        for i in 0..keys {
            let key = key_of(i);
            assert_eq!(
                table.get(&key),
                Some(&key),
                "key {i} of {keys} missing at {MAX_LEVELS} levels"
            );
        }

        for i in keys..keys + 1_000 {
            let key = key_of(i);
            assert!(
                table.get(&key).is_none(),
                "never-inserted key {i} reported present at {MAX_LEVELS} levels"
            );
        }

        level_occupancy(table)
    }

    fn round_trip<const MAX_LEVELS: usize>(cnts: Vec<u32>, keys: u64) -> Vec<usize> {
        let capacity = slots(&cnts);

        let mut table =
            MultiTableFiltered::<KEY_LEN, VAL_LEN, MAX_LEVELS, HasherBuilder>::new_from_cnts(
                cnts,
            );

        let occupancy = fill_and_check(&mut table, keys);
        println!("{MAX_LEVELS} levels, {keys} keys in {capacity} slots: {occupancy:?}");
        occupancy
    }

    /// Every level must hold a power-of-two bucket count so `powers-of-2`'s shift-based probe
    /// works, and the shift-derived counts must still cover every allocated bucket.
    #[cfg(feature = "powers-of-2")]
    fn assert_levels_are_powers_of_two<const MAX_LEVELS: usize>(
        table: &MultiTableFiltered<KEY_LEN, VAL_LEN, MAX_LEVELS, HasherBuilder>,
    ) {
        let cnts = table.level_bucket_cnts();
        for (level, &cnt) in cnts.iter().enumerate() {
            assert!(
                cnt.is_power_of_two(),
                "level {level} holds {cnt} buckets, which is not a power of two"
            );
        }
        assert_eq!(
            cnts.iter().sum::<usize>(),
            table.total_buckets(),
            "the per-level shifts {cnts:?} lose buckets against the allocation"
        );
    }

    /// Nine levels: the first `MAX_LEVELS` that leaves the unrolled block and
    /// lands in the loop.
    #[cfg(feature = "agile")]
    #[test]
    fn round_trips_at_nine_levels() {
        let cnts = geometric_cnts(9, 400, 0.72);
        let keys = slots(&cnts) * 95 / 100;
        let occupancy = round_trip::<9>(cnts, keys);

        let populated = occupancy.iter().filter(|&&n| n > 0).count();
        assert!(
            populated >= 8,
            "only {populated} of nine levels hold keys: {occupancy:?}"
        );
    }

    /// Nineteen levels with the exact cascade `examples/perfect.rs` uses, so the
    /// test and the example cover the same shape.
    #[cfg(feature = "agile")]
    #[test]
    fn round_trips_at_nineteen_levels_like_perfect() {
        let cnts = vec![
            277, 203, 148, 108, 78, 56, 39, 28, 20, 14, 9, 7, 4, 3, 2, 1, 1, 1, 1,
        ];
        let keys = slots(&cnts) * 95 / 100;
        let occupancy = round_trip::<19>(cnts, keys);

        let populated = occupancy.iter().filter(|&&n| n > 0).count();
        assert!(
            populated > 8,
            "the point of this path is levels past the eighth; only {populated} levels hold keys: {occupancy:?}"
        );
    }

    /// Fifty levels: the ceiling `new_from_cnts` allows.
    #[cfg(feature = "agile")]
    #[test]
    fn round_trips_at_fifty_levels() {
        let cnts = geometric_cnts(50, 1_000, 0.75);
        let keys = slots(&cnts) * 95 / 100;
        let occupancy = round_trip::<50>(cnts, keys);

        let populated = occupancy.iter().filter(|&&n| n > 0).count();
        assert!(
            populated > 8,
            "only {populated} of fifty levels hold keys: {occupancy:?}"
        );
    }

    /// Eight levels, the unrolled `is_present` path rather than the fallback loop, loaded high
    /// enough that the cascade reaches level 4.
    #[cfg(feature = "agile")]
    #[test]
    fn round_trips_at_eight_levels_reaching_level_four() {
        let cnts = vec![277, 203, 148, 108, 78, 56, 39, 28];
        let keys = slots(&cnts) * 90 / 100;
        let occupancy = round_trip::<8>(cnts, keys);

        assert!(
            occupancy[4] > 0,
            "level 4 holds no keys even though the cascade was loaded to reach it: {occupancy:?}"
        );
    }

    /// Levels past the cascade's end all address bucket 0, so widening into them would file
    /// surplus keys under the wrong key's chain.
    #[test]
    fn a_short_cascade_never_widens_past_its_last_level() {
        let cnts = vec![8u32, 1];
        let levels = cnts.len();
        let mut table =
            MultiTableFiltered::<KEY_LEN, VAL_LEN, 8, HasherBuilder>::new_from_cnts(cnts);

        // `Full` here is expected: it signals space ran out, not a bug.
        let mut stored = Vec::new();
        for i in 0..400u64 {
            let key = key_of(i);
            match table.insert(key, key) {
                Ok(()) => stored.push(i),
                Err(InsertError::AlreadyPresent) => {}
                Err(InsertError::Full) => break,
            }
        }

        assert_eq!(
            table.first_empty_level, levels,
            "frontier reached level {} on a {levels}-level cascade",
            table.first_empty_level
        );
        assert!(!stored.is_empty(), "nothing was accepted at all");
        for &i in &stored {
            assert_eq!(table.get(&key_of(i)), Some(&key_of(i)), "key {i} lost");
        }
    }

    /// `powers-of-2` at the default depth, on the cascade its own `with_capacity` builds.
    #[cfg(feature = "powers-of-2")]
    #[test]
    fn round_trips_with_capacity_under_powers_of_2() {
        const N: u64 = 50_000;

        let mut table = MultiTableFiltered::<KEY_LEN, VAL_LEN>::with_capacity(N, 0.75);
        assert_levels_are_powers_of_two(&table);

        let occupancy = fill_and_check(&mut table, N);
        println!("powers-of-2 with_capacity({N}, 0.75): {occupancy:?}");

        let populated = occupancy.iter().filter(|&&n| n > 0).count();
        assert!(
            populated >= 3,
            "only {populated} levels hold keys, so the cascade is not being walked: {occupancy:?}"
        );
    }

    /// `powers-of-2` past the eighth level: the plain loop, not the unrolled path.
    #[cfg(feature = "powers-of-2")]
    #[test]
    fn round_trips_at_twelve_levels_under_powers_of_2() {
        let cnts = vec![
            2048, 2048, 1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64,
        ];
        let keys = slots(&cnts) * 95 / 100;
        let occupancy = round_trip::<12>(cnts, keys);

        let populated = occupancy.iter().filter(|&&n| n > 0).count();
        assert!(
            populated > 8,
            "the point of this path is levels past the eighth; only {populated} hold keys: {occupancy:?}"
        );
    }

    /// Eight levels under `powers-of-2`, loaded to need level 4. First eight
    /// counts of `round_trips_at_twelve_levels_under_powers_of_2`'s cascade.
    #[cfg(feature = "powers-of-2")]
    #[test]
    fn round_trips_at_eight_levels_reaching_level_four() {
        let cnts = vec![2048, 2048, 1024, 1024, 512, 512, 256, 256];
        let keys = slots(&cnts) * 95 / 100;
        let occupancy = round_trip::<8>(cnts, keys);

        assert!(
            occupancy[4] > 0,
            "level 4 holds no keys even though the cascade was loaded to reach it: {occupancy:?}"
        );
    }

    /// Checks the sizing function directly: every level is a power of two and the cascade
    /// covers its key budget. The depth cap (60) is generous since high loads round longer.
    #[cfg(feature = "powers-of-2")]
    #[test]
    fn every_sized_level_is_a_power_of_two() {
        for q in [256u64, 50_000, 1_000_000, 100_000_000] {
            for a in [0.4, 0.5, 0.75, 0.87, 0.95, 0.997] {
                let cnts =
                    bucket_cnts_for_utilization_powers(q, DEFAULT_BUCKET_SIZE as u32, a, 60);
                assert!(
                    cnts.iter().all(|c| c.is_power_of_two()),
                    "sizing for q = {q}, a = {a} has a level that is not a power of two: {cnts:?}"
                );
                assert!(
                    slots(&cnts) >= q,
                    "sizing for q = {q}, a = {a} holds only {} slots: {cnts:?}",
                    slots(&cnts)
                );
            }
        }
    }

    /// A one-bucket level is a real level, not the end of the cascade.
    #[cfg(feature = "powers-of-2")]
    #[test]
    fn a_single_bucket_last_level_is_reachable() {
        let cnts = vec![4u32, 1];
        let keys = slots(&cnts);
        let occupancy = round_trip::<8>(cnts, keys);

        assert!(
            occupancy[1] > 0,
            "the one-bucket last level took no keys at all: {occupancy:?}"
        );
    }

    /// `first-power` at the default depth, on the cascade its own `with_capacity` builds.
    #[cfg(feature = "first-power")]
    #[test]
    fn round_trips_with_capacity_under_first_power() {
        const N: u64 = 50_000;

        // 0.9, not the paper's 0.75: level 0 fills exactly, and from 0.9 level 1 holds fewer slots
        // than the leftover keys at 8 and 16 wide alike, so the walk has to reach level 2.
        const UTILIZATION: f64 = 0.9;

        let mut table = MultiTableFiltered::<KEY_LEN, VAL_LEN>::with_capacity(N, UTILIZATION);

        // The one level the shift addresses has to be a power of two; the rest
        // are ordinary counts, mapped the way `agile` maps them.
        assert!(
            table.levels_meta[0].0.is_power_of_two(),
            "level 0 holds {} buckets, which is not a power of two",
            table.levels_meta[0].0
        );

        let occupancy = fill_and_check(&mut table, N);
        println!("first-power with_capacity({N}, {UTILIZATION}): {occupancy:?}");

        let populated = occupancy.iter().filter(|&&n| n > 0).count();
        assert!(
            populated >= 3,
            "only {populated} levels hold keys, so the cascade is not being walked: {occupancy:?}"
        );
    }

    /// `first-power` on the deep-level loop, with the other deep tests'
    /// nineteen-level cascade, first level rounded to a power of two.
    #[cfg(feature = "first-power")]
    #[test]
    fn round_trips_at_nineteen_levels_under_first_power() {
        let cnts = vec![
            256, 203, 148, 108, 78, 56, 39, 28, 20, 14, 9, 7, 4, 3, 2, 1, 1, 1, 1,
        ];
        let keys = slots(&cnts) * 95 / 100;
        let occupancy = round_trip::<19>(cnts, keys);

        let populated = occupancy.iter().filter(|&&n| n > 0).count();
        assert!(
            populated > 8,
            "the point of this path is levels past the eighth; only {populated} hold keys: {occupancy:?}"
        );
    }

    /// Eight levels under `first-power`, loaded to need level 4. Same
    /// cascade prefix as the nineteen-level test above.
    #[cfg(feature = "first-power")]
    #[test]
    fn round_trips_at_eight_levels_reaching_level_four() {
        let cnts = vec![256, 203, 148, 108, 78, 56, 39, 28];
        let keys = slots(&cnts) * 90 / 100;
        let occupancy = round_trip::<8>(cnts, keys);

        assert!(
            occupancy[4] > 0,
            "level 4 holds no keys even though the cascade was loaded to reach it: {occupancy:?}"
        );
    }

    /// What `first-power`'s sizing promises: only the first level is a
    /// power of two, which is why only level 0 is probed with a shift.
    #[cfg(feature = "first-power")]
    #[test]
    fn only_the_first_sized_level_is_a_power_of_two() {
        for q in [256u64, 50_000, 1_000_000, 100_000_000] {
            for a in [0.4, 0.5, 0.75, 0.87, 0.95, 0.997] {
                let cnts = bucket_cnts_for_utilization_first_power(
                    q,
                    DEFAULT_BUCKET_SIZE as u32,
                    a,
                    60,
                );
                assert!(
                    cnts[0].is_power_of_two(),
                    "sizing for q = {q}, a = {a} starts with {} buckets, not a power of two: {cnts:?}",
                    cnts[0]
                );
            }
        }
    }
}

/// The filter scan reads `SIMD_ARITY` bytes from a `BUCKET_SIZE`-byte bucket, so at 12/16
/// the top four lanes belong to the next bucket, not this one.
#[cfg(test)]
mod filter_overread_tests {
    use crate::*;

    const KEY_LEN: usize = 4;
    const VAL_LEN: usize = 4;
    const BUCKET_SIZE: usize = 12;
    const SIMD_ARITY: usize = 16;

    type Table =
        MultiTableFiltered<KEY_LEN, VAL_LEN, 8, HasherBuilder, BUCKET_SIZE, SIMD_ARITY>;

    /// Powers of two throughout, so the cascade stays legal under every variant feature
    /// (`powers-of-2` shifts every level, `first-power` only the first).
    const CNTS: [u32; 4] = [64, 16, 4, 1];

    /// A table that once held `key`, now wiped, plus the bucket and filter byte it used.
    /// `first_empty_level` stays 1, so a lookup only ever probes that bucket.
    fn wiped_table(key: [u8; KEY_LEN]) -> (Table, usize, u8) {
        let mut table = Table::new_from_cnts(CNTS.to_vec());
        table.insert(key, [0xAA; VAL_LEN]).expect("insert");

        let home = (0..table.total_buckets())
            .find(|&b| table.filters[b][0] != EMPTY_FILTER)
            .expect("the key landed somewhere");
        let filter = table.filters[home][0];

        assert_eq!(
            table.remove(&key),
            Some([0xAA; VAL_LEN]),
            "the key was not there to remove"
        );
        // The pair as well, which `remove` leaves alone: these tests plant their own bytes here.
        table.buckets_data[home][0] = ([0u8; KEY_LEN], [0u8; VAL_LEN]);
        assert_eq!(table.get(&key), None, "the wipe left the key findable");

        (table, home, filter)
    }

    /// A key whose level-0 bucket has a bucket after it to over-read into.
    fn key_with_a_next_bucket() -> ([u8; KEY_LEN], Table, usize, u8) {
        for i in 0u32.. {
            let key = i.to_ne_bytes();
            let (table, home, filter) = wiped_table(key);
            if home + 1 < table.total_buckets() {
                return (key, table, home, filter);
            }
        }
        unreachable!()
    }

    /// A filter byte alone, in the first over-read lane: the key compare has
    /// to reject it, and the slot it names is not this bucket's to read.
    #[test]
    fn a_matching_filter_byte_in_the_next_bucket_is_not_a_hit() {
        let (key, mut table, home, filter) = key_with_a_next_bucket();

        table.filters[home + 1][0] = filter;
        table.buckets_data[home + 1][0] = ([0xFF; KEY_LEN], [0xBB; VAL_LEN]);

        assert_eq!(
            table.get(&key),
            None,
            "a filter byte planted in the next bucket's slot 0 was read as this bucket's"
        );
    }

    /// The same lane, but with the searched key itself planted there. That bucket is off
    /// this key's search path (`first_empty_level` is 1), so a hit can only be the over-read.
    #[test]
    fn a_matching_key_in_the_next_bucket_is_not_a_hit() {
        let (key, mut table, home, filter) = key_with_a_next_bucket();

        table.filters[home + 1][0] = filter;
        table.buckets_data[home + 1][0] = (key, [0xBB; VAL_LEN]);

        assert_eq!(
            table.get(&key),
            None,
            "a key stored in the next bucket was found by scanning this one"
        );
    }

    /// A real hit in the bucket's last slot, adjacent to a matching over-read lane: masking
    /// the over-read away must not cost that last slot.
    #[test]
    fn a_next_bucket_lane_does_not_hide_a_hit_beside_it() {
        let (key, mut table, home, filter) = key_with_a_next_bucket();
        let last = BUCKET_SIZE - 1;

        table.filters[home][last] = filter;
        table.buckets_data[home][last] = (key, [0xAA; VAL_LEN]);
        table.filters[home + 1][0] = filter;
        table.buckets_data[home + 1][0] = ([0xFF; KEY_LEN], [0xBB; VAL_LEN]);

        assert_eq!(
            table.get(&key),
            Some(&[0xAA; VAL_LEN]),
            "masking the next bucket's lane away lost the slot right before it"
        );
    }
}

/// A filter collision inside one bucket: the first candidate the reduction names is some
/// other key, so only a walk past it reaches the real pair.
#[cfg(test)]
mod filter_collision_tests {
    use crate::*;

    fn walks_past_a_collision<const BUCKET_SIZE: usize, const SIMD_ARITY: usize>() {
        let mut table = MultiTableFiltered::<
            4,
            4,
            8,
            HasherBuilder,
            BUCKET_SIZE,
            SIMD_ARITY,
        >::new_from_cnts(vec![64, 16, 4, 1]);

        let key = 7u32.to_ne_bytes();
        table.insert(key, [0xAA; 4]).expect("insert");

        let home = (0..table.total_buckets())
            .find(|&b| table.filters[b][0] != EMPTY_FILTER)
            .expect("the key landed somewhere");
        let filter = table.filters[home][0];

        // Slot 0 keeps the filter byte under a different key; the real pair moves
        // one slot along, out of reach of the first candidate alone.
        table.buckets_data[home][0] = ([0xFF; 4], [0xBB; 4]);
        table.filters[home][1] = filter;
        table.buckets_data[home][1] = (key, [0xAA; 4]);

        assert_eq!(
            table.get(&key),
            Some(&[0xAA; 4]),
            "the search stopped at the colliding slot instead of walking on"
        );
    }

    #[test]
    fn a_bucket_wide_scan_walks_past_a_collision() {
        walks_past_a_collision::<8, 8>();
    }

    #[test]
    fn an_over_reading_scan_walks_past_a_collision() {
        walks_past_a_collision::<12, 16>();
    }
}

/// `fast_eq`'s >32-byte arm: u128 chunks over the length plus an
/// overlapping tail chunk for a non-multiple-of-16 length.
#[cfg(test)]
mod fast_eq_tests {
    use crate::utils::fast_eq;

    #[test]
    fn equal_keys_match_above_32() {
        let a = [7u8; 48];
        let b = [7u8; 48];
        assert!(fast_eq(&a, &b));

        let a = [7u8; 40];
        let b = [7u8; 40];
        assert!(fast_eq(&a, &b));
    }

    #[test]
    fn differs_at_first_byte() {
        let a = [0u8; 48];
        let mut b = a;
        b[0] = 1;
        assert!(!fast_eq(&a, &b));
    }

    #[test]
    fn differs_at_last_byte() {
        let a = [0u8; 48];
        let mut b = a;
        b[47] = 1;
        assert!(!fast_eq(&a, &b));
    }

    #[test]
    fn differs_only_in_tail_region_of_non_multiple_length() {
        // 40 isn't a multiple of 16, so the tail chunk [24..40) overlaps the second full chunk
        // [16..32); byte 39, past both full chunks, must still be caught.
        let a = [0u8; 40];
        let mut b = a;
        b[39] = 1;
        assert!(!fast_eq(&a, &b));

        // Same, but the only differing byte is 24, inside the overlap with the second full chunk.
        let mut b2 = a;
        b2[24] = 1;
        assert!(!fast_eq(&a, &b2));
    }
}

/// `find_simd` at every key width with a whole-key lane: must match a plain first-match
/// scan, held to `len`, whatever the bucket keys' overlap.
#[cfg(test)]
mod find_simd_tests {
    use crate::utils::{fast_eq, find_simd};

    /// The contract `find_simd` implements, spelled out scalar.
    fn reference<const KEY_LEN: usize, const BUCKET_LEN: usize>(
        data: &[[u8; KEY_LEN]; BUCKET_LEN],
        target: &[u8; KEY_LEN],
        len: usize,
    ) -> Option<usize> {
        data.iter()
            .position(|k| fast_eq(k, target))
            .filter(|&i| i < len)
    }

    /// Keys that pairwise share one half and differ in the other, so a scan
    /// comparing only half a key would report a match that isn't there.
    fn bucket<const KEY_LEN: usize, const BUCKET_LEN: usize>() -> [[u8; KEY_LEN]; BUCKET_LEN] {
        std::array::from_fn(|i| {
            let mut key = [0u8; KEY_LEN];
            let (first, last) = if i % 2 == 0 {
                (7, i as u8)
            } else {
                (i as u8, 7)
            };
            key[0] = first;
            key[KEY_LEN - 1] = last;
            key
        })
    }

    fn check<const KEY_LEN: usize, const BUCKET_LEN: usize>() {
        let data = bucket::<KEY_LEN, BUCKET_LEN>();

        for i in 0..BUCKET_LEN {
            for len in 0..=BUCKET_LEN {
                assert_eq!(
                    find_simd::<KEY_LEN, BUCKET_LEN>(&data, &data[i], len),
                    reference(&data, &data[i], len),
                    "key {i} of {BUCKET_LEN} at len {len}, width {KEY_LEN}"
                );
            }
        }

        // Absent targets: one unlike any key, and the all-zero empty slot.
        for target in [[200u8; KEY_LEN], [0u8; KEY_LEN]] {
            for len in 0..=BUCKET_LEN {
                assert_eq!(
                    find_simd::<KEY_LEN, BUCKET_LEN>(&data, &target, len),
                    None,
                    "absent target reported at len {len}, width {KEY_LEN}"
                );
            }
        }

        // A repeated key must report the first of the two, not the later one.
        if BUCKET_LEN >= 4 {
            let mut repeated = data;
            repeated[BUCKET_LEN - 1] = repeated[1];
            assert_eq!(
                find_simd::<KEY_LEN, BUCKET_LEN>(&repeated, &repeated[1], BUCKET_LEN),
                Some(1),
                "repeated key at width {KEY_LEN}"
            );
        }
    }

    #[test]
    fn whole_key_lanes_agree_with_a_scalar_scan() {
        // Widths 4/8/16 over a full bucket, a whole vector, and a bucket too short for one,
        // the last landing entirely in the scalar tail.
        check::<4, 8>();
        check::<8, 8>();
        check::<16, 8>();
        check::<4, 4>();
        check::<8, 4>();
        check::<16, 4>();
        check::<8, 2>();
        check::<16, 2>();
    }
}

/// Coverage for the plain `MultiTable` search path: the empty-slot check and the
/// `first_empty_level` frontier, both of which change what a search may skip.
#[cfg(test)]
mod multi_table_search_tests {
    use crate::*;
    use std::hint::black_box;
    use std::time::Instant;

    const KEY_LEN: usize = 8;
    const VAL_LEN: usize = 4;

    /// Distinct keys from a bijective scramble of `i`, as in `deep_levels_tests`.
    fn key_of(i: u64) -> [u8; KEY_LEN] {
        i.wrapping_mul(0x9E37_79B9_7F4A_7C15).to_ne_bytes()
    }

    fn value_of(i: u64) -> [u8; VAL_LEN] {
        (i as u32).to_ne_bytes()
    }

    fn geometric_cnts(levels: usize, first: u32, ratio: f64) -> Vec<u32> {
        let mut cnts = Vec::with_capacity(levels);
        let mut cur = first as f64;
        for _ in 0..levels {
            cnts.push((cur.round() as u32).max(1));
            cur *= ratio;
        }
        cnts
    }

    fn slots(cnts: &[u32]) -> u64 {
        cnts.iter().sum::<u32>() as u64 * DEFAULT_BUCKET_SIZE as u64
    }

    /// Keys held per level, from `next_spot` counters. Level 0 lives in `first_level_cnts`;
    /// later levels are `levels_meta[level - 1]`, shifted since level 0 sits outside that array.
    fn level_occupancy<const MAX_LEVELS: usize>(
        table: &MultiTable<KEY_LEN, VAL_LEN, MAX_LEVELS>,
    ) -> Vec<usize> {
        (0..MAX_LEVELS)
            .map(|level| {
                let (cnt, offset) = if level == 0 {
                    (table.first_level_cnts, 0)
                } else {
                    table.levels_meta[level - 1]
                };
                (offset as usize..(offset + cnt) as usize)
                    .map(|bucket| table.next_spot[bucket] as usize)
                    .sum::<usize>()
            })
            .collect()
    }

    /// On a cascade where every level has buckets, the deepest key sits at exactly
    /// `first_empty_level - 1`: the invariant the search cutoff relies on.
    fn assert_frontier_matches_occupancy<const MAX_LEVELS: usize>(
        table: &MultiTable<KEY_LEN, VAL_LEN, MAX_LEVELS>,
        occupancy: &[usize],
    ) {
        let deepest = occupancy
            .iter()
            .rposition(|&n| n > 0)
            .expect("no keys stored at all");
        assert_eq!(
            table.first_empty_level,
            deepest + 1,
            "frontier at {} but keys reach level {deepest}: {occupancy:?}",
            table.first_empty_level
        );
    }

    fn round_trip<const MAX_LEVELS: usize>(cnts: Vec<u32>, keys: u64) -> Vec<usize> {
        let capacity = slots(&cnts);
        let mut table = MultiTable::<KEY_LEN, VAL_LEN, MAX_LEVELS>::new_from_cnts(cnts);

        for i in 0..keys {
            table
                .insert(key_of(i), value_of(i))
                .unwrap_or_else(|e| panic!("insert {i} of {keys} failed at {MAX_LEVELS} levels: {e}"));
        }

        for i in 0..keys {
            assert_eq!(
                table.get(&key_of(i)),
                Some(&value_of(i)),
                "key {i} of {keys} missing at {MAX_LEVELS} levels"
            );
        }

        for i in keys..keys + 1_000 {
            assert!(
                table.get(&key_of(i)).is_none(),
                "never-inserted key {i} reported present at {MAX_LEVELS} levels"
            );
        }

        for i in (0..keys).step_by((keys / 50).max(1) as usize) {
            assert!(
                table.insert(key_of(i), value_of(i)).is_err(),
                "duplicate insert of key {i} was accepted at {MAX_LEVELS} levels"
            );
        }

        let occupancy = level_occupancy(&table);
        assert_frontier_matches_occupancy(&table, &occupancy);
        println!("{MAX_LEVELS} levels, {keys} keys in {capacity} slots: {occupancy:?}");
        occupancy
    }

    /// The default depth, at a load that spreads keys well down the
    /// cascade, widening the frontier repeatedly.
    #[test]
    fn round_trips_at_default_eight_levels() {
        let cnts = geometric_cnts(8, 400, 0.72);
        let keys = slots(&cnts) * 90 / 100;
        let occupancy = round_trip::<8>(cnts, keys);

        let populated = occupancy.iter().filter(|&&n| n > 0).count();
        assert!(
            populated >= 6,
            "only {populated} of eight levels hold keys, too shallow to exercise the cutoff: {occupancy:?}"
        );
    }

    /// A deep cascade, the nineteen-level shape `examples/perfect.rs` uses,
    /// crossing far more levels than the frontier starts out allowing.
    #[test]
    fn round_trips_at_nineteen_levels() {
        let cnts = vec![
            277, 203, 148, 108, 78, 56, 39, 28, 20, 14, 9, 7, 4, 3, 2, 1, 1, 1, 1,
        ];
        let keys = slots(&cnts) * 95 / 100;
        let occupancy = round_trip::<19>(cnts, keys);

        let populated = occupancy.iter().filter(|&&n| n > 0).count();
        assert!(
            populated > 8,
            "the point of this case is levels past the eighth; only {populated} hold keys: {occupancy:?}"
        );
    }

    /// The cascade the crate actually builds for a key budget, at the default
    /// depth: the realistic shape, as opposed to the hand-made cascades above.
    #[test]
    fn round_trips_with_capacity_at_default_levels() {
        const N: u64 = 50_000;
        // 0.85, not the paper's 0.75, so the frontier-movement assertion below stays meaningful.
        const UTILIZATION: f64 = 0.85;

        let mut table = MultiTable::<KEY_LEN, VAL_LEN>::with_capacity(N, UTILIZATION);

        for i in 0..N {
            table
                .insert(key_of(i), value_of(i))
                .unwrap_or_else(|e| panic!("insert {i} of {N} failed: {e}"));
        }
        for i in 0..N {
            assert_eq!(table.get(&key_of(i)), Some(&value_of(i)), "missing key {i}");
        }
        for i in N..N + 1_000 {
            assert!(
                table.get(&key_of(i)).is_none(),
                "never-inserted key {i} reported present"
            );
        }

        let occupancy = level_occupancy(&table);
        assert_frontier_matches_occupancy(&table, &occupancy);

        let populated = occupancy.iter().filter(|&&n| n > 0).count();
        println!("with_capacity({N}, {UTILIZATION}): {occupancy:?}");
        // At bucket size >= 32 nothing spills at this count, so the frontier staying put is
        // the correct outcome here, not a failure.
        assert!(
            populated >= 2 || DEFAULT_BUCKET_SIZE >= 32,
            "nothing spilled past level 0, so the frontier never had to move: {occupancy:?}"
        );
    }

    /// The frontier must stop at the cascade's last level, since absent levels all address
    /// bucket 0, which lets the search drop its zero-count check.
    #[test]
    fn a_short_cascade_never_widens_past_its_last_level() {
        let cnts = vec![8u32, 1];
        let levels = cnts.len();
        let capacity = slots(&cnts);
        let mut table = MultiTable::<KEY_LEN, VAL_LEN, 8>::new_from_cnts(cnts);

        let stored: Vec<u64> = (0..capacity * 4)
            .filter(|&i| table.insert(key_of(i), value_of(i)).is_ok())
            .collect();

        assert_eq!(
            table.first_empty_level, levels,
            "frontier reached level {} on a {levels}-level cascade",
            table.first_empty_level
        );
        assert!(
            stored.len() as u64 <= capacity,
            "{} keys accepted into {capacity} slots",
            stored.len()
        );
        for &i in &stored {
            assert_eq!(table.get(&key_of(i)), Some(&value_of(i)), "key {i} lost");
        }
    }

    #[test]
    fn round_trips_with_a_non_default_val_len() {
        const N: u64 = 20_000;
        const KEY_LEN: usize = 8;
        const VAL_LEN: usize = 16;

        fn key_of(i: u64) -> [u8; KEY_LEN] {
            i.wrapping_mul(0x9E37_79B9_7F4A_7C15).to_ne_bytes()
        }

        fn value_of(i: u64) -> [u8; VAL_LEN] {
            let mut value = [0u8; VAL_LEN];
            value[0..8].copy_from_slice(&i.to_ne_bytes());
            value[8..16].copy_from_slice(&(!i).to_ne_bytes());
            value
        }

        let mut table = MultiTable::<KEY_LEN, VAL_LEN>::with_capacity(N, 0.75);

        for i in 0..N {
            table
                .insert(key_of(i), value_of(i))
                .unwrap_or_else(|e| panic!("insert {i} of {N} failed: {e}"));
        }
        for i in 0..N {
            assert_eq!(table.get(&key_of(i)), Some(&value_of(i)), "missing key {i}");
        }
        for i in N..N + 1_000 {
            assert!(
                table.get(&key_of(i)).is_none(),
                "never-inserted key {i} reported present"
            );
        }
    }

    /// Times insert, a hitting lookup, and a missing lookup (the one that
    /// walks every level it's allowed to).
    fn time_search<const MAX_LEVELS: usize>(
        label: &str,
        mut table: MultiTable<KEY_LEN, VAL_LEN, MAX_LEVELS>,
        keys: u64,
    ) {
        let started = Instant::now();
        for i in 0..keys {
            table
                .insert(key_of(i), value_of(i))
                .unwrap_or_else(|e| panic!("insert {i} failed: {e}"));
        }
        let insert_ns = started.elapsed().as_nanos() as f64 / keys as f64;

        let started = Instant::now();
        let mut hits = 0u64;
        for i in 0..keys {
            hits += black_box(table.get(&key_of(i))).is_some() as u64;
        }
        let hit_ns = started.elapsed().as_nanos() as f64 / keys as f64;
        assert_eq!(hits, keys);

        let started = Instant::now();
        let mut misses = 0u64;
        for i in keys..2 * keys {
            misses += black_box(table.get(&key_of(i))).is_none() as u64;
        }
        let miss_ns = started.elapsed().as_nanos() as f64 / keys as f64;
        assert_eq!(misses, keys);

        println!(
            "{label}: {keys} keys, insert {insert_ns:.1} ns/op, get hit {hit_ns:.1} ns/op, get miss {miss_ns:.1} ns/op"
        );
    }

    /// Coarse wall-clock numbers for the search path, since criterion doesn't cover `MultiTable`.
    /// Run: `cargo +nightly test --release --features agile --lib -- --ignored --nocapture timing`.
    #[test]
    #[ignore]
    fn timing_indication() {
        const N: u64 = 2_000_000;

        // The realistic shape: what `with_capacity` builds for a key budget,
        // which is only a couple of levels deep.
        time_search(
            "with_capacity(2M, 0.75)",
            MultiTable::<KEY_LEN, VAL_LEN, 8>::with_capacity(N, 0.75),
            N,
        );

        // A deep cascade at high load, where a search crosses many levels and
        // the frontier has something to cut off.
        let cnts = geometric_cnts(19, 70_000, 0.72);
        let keys = slots(&cnts) * 90 / 100;
        time_search(
            "19 levels at 90% load",
            MultiTable::<KEY_LEN, VAL_LEN, 19>::new_from_cnts(cnts),
            keys,
        );
    }
}

/// The shape the walk's stop rests on: writes take the lowest empty slot and a delete never
/// zeroes a last byte, so an empty last slot means the bucket never filled. Without deletes the
/// occupied slots are then a prefix, which is what this checks; a compacting delete would break it.
#[cfg(test)]
mod fill_shape {
    use crate::*;

    #[test]
    fn filtered_fill_is_contiguous() {
        for lf in [0.50f64, 0.77, 0.85] {
            let n: u64 = 400_000;
            let mut table = MultiTableFiltered::<4, 4, 16>::with_capacity(n, lf);
            // splitmix64 keys: a Weyl sequence would be spread so evenly by a
            // multiplicative hash that no bucket ever fills, testing nothing.
            let mut state: u64 = 0x243f_6a88_85a3_08d3;
            for _ in 0..n {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                let key = ((z ^ (z >> 31)) as u32).to_ne_bytes();
                let _ = table.insert(key, key);
            }

            let last = DEFAULT_FILTERED_BUCKET_SIZE - 1;
            for filters in table.filters.iter().take(table.total_buckets()) {
                let highest_written = match filters.iter().rposition(|&b| b != EMPTY_FILTER) {
                    Some(highest) => highest,
                    None => continue,
                };
                assert!(
                    filters[..highest_written]
                        .iter()
                        .all(|&b| b != EMPTY_FILTER),
                    "lf {lf}: a written slot sits above an empty one, so the \
                     occupied slots are no longer a prefix: {filters:?}"
                );
                assert!(
                    filters[last] == EMPTY_FILTER || highest_written == last,
                    "lf {lf}: bucket is stepped over yet still has room: {filters:?}"
                );
            }
        }
    }
}

/// The recording insert walk must find the shallowest available slot, not just the first
/// bucket with room; state below is poked by hand since insert-only fills can't tell them apart.
#[cfg(test)]
mod first_available_tests {
    use crate::*;

    const KEY_LEN: usize = 8;
    const VAL_LEN: usize = 4;
    const B: usize = DEFAULT_FILTERED_BUCKET_SIZE;

    /// Four levels, not three: the variant features seed four levels'
    /// buckets up front, so a shallower cascade would index past its array.
    type Table = MultiTableFiltered<KEY_LEN, VAL_LEN, 4>;

    /// Distinct keys from a bijective scramble of `i`, as elsewhere.
    fn key_of(i: u64) -> [u8; KEY_LEN] {
        i.wrapping_mul(0x9E37_79B9_7F4A_7C15).to_ne_bytes()
    }

    fn value_of(i: u64) -> [u8; VAL_LEN] {
        (i as u32).to_ne_bytes()
    }

    /// One bucket per level, so every key's chain is 0 -> 1 -> 2 -> 3 under all three variant
    /// features; key `i` lands in bucket `i / B`, slot `i % B`.
    fn chain_table(filled: u64) -> Table {
        let mut table = Table::new_from_cnts(vec![1, 1, 1, 1]);
        for i in 0..filled {
            table
                .insert(key_of(i), value_of(i))
                .unwrap_or_else(|e| panic!("fill insert {i} failed: {e}"));
        }
        table
    }

    /// A slot to hole: mid-bucket, and never the last one, whose byte is the
    /// overflow signal a delete must not clear.
    fn hole_slot() -> usize {
        (B / 2).min(B.saturating_sub(2))
    }

    fn position_of(table: &Table, key: &[u8; KEY_LEN]) -> (u32, u32) {
        match table.is_present::<false, false>(key) {
            SearchResult::Found(bucket, slot) => (bucket, slot),
            _ => panic!("key not found"),
        }
    }

    /// Holes bucket 0, checks the probe reports that hole rather than deeper room, inserts,
    /// and checks the key landed there and everything else still round-trips.
    fn check_hole_is_reused(mut table: Table, filled: u64) {
        let hole = hole_slot();
        // Key `hole` sits at (0, hole): mid-bucket, so the byte goes back to empty.
        assert_eq!(
            table.remove(&key_of(hole as u64)),
            Some(value_of(hole as u64)),
            "the fill key at the slot to hole was not there"
        );

        let (key, value) = (key_of(1_000), value_of(1_000));
        match table.is_present_recording::<false, true>(&key) {
            SearchResult::NotFoundFirstEmpty(bucket, slot, level, _) => assert_eq!(
                (bucket, slot, level),
                (0, hole as u32, 0),
                "probe passed over the shallowest available slot"
            ),
            _ => panic!("probe found no room at all"),
        }

        table
            .insert(key, value)
            .expect("insert into the holed bucket");
        assert_eq!(
            position_of(&table, &key),
            (0, hole as u32),
            "insert landed past the shallowest available slot"
        );
        assert_eq!(table.get(&key), Some(&value));

        assert_eq!(table.get(&key_of(hole as u64)), None);
        for i in (0..filled).filter(|&i| i != hole as u64) {
            assert_eq!(
                table.get(&key_of(i)),
                Some(&value_of(i)),
                "fill key {i} lost around the reused hole"
            );
        }
    }

    /// Bucket 0 overflowed, then its last slot deleted: the tombstone keeps the overflowed keys
    /// reachable, the non-recording probe still walks past it, and `insert` lands on it.
    #[test]
    fn a_tombstoned_last_slot_keeps_the_overflow_signal_and_insert_reuses_it() {
        let filled = B as u64 + 1;
        let last = B - 1;
        // The byte alone first, on a twin table: the plain probe has to walk past a tombstone
        // rather than stop on it, which is only visible while the delete is unannounced, since
        // a `remove` hands the inserts over and that probe then gives up at level 0.
        let mut byte_only = chain_table(filled);
        byte_only.filters[0][last] = TOMBSTONE_FILTER;

        assert_eq!(byte_only.get(&key_of(last as u64)), None);
        assert_eq!(
            position_of(&byte_only, &key_of(B as u64)),
            (1, 0),
            "the overflowed key was lost behind the tombstone"
        );
        for i in (0..filled).filter(|&i| i != last as u64) {
            assert_eq!(byte_only.get(&key_of(i)), Some(&value_of(i)));
        }

        // The non-recording probe stops only on a zero last byte, so it walks on to bucket 1.
        let (key, value) = (key_of(1_000), value_of(1_000));
        match byte_only.is_present::<false, true>(&key) {
            SearchResult::NotFoundFirstEmpty(bucket, slot, level, _) => assert_eq!(
                (bucket, slot, level),
                (1, 1, 1),
                "the non-recording probe offered the tombstone"
            ),
            _ => panic!("the non-recording probe found no room"),
        }

        // The whole delete now, byte and announcement together, on a table of the same fill.
        let mut table = chain_table(filled);
        assert_eq!(
            table.remove(&key_of(last as u64)),
            Some(value_of(last as u64)),
            "the fill key in the last slot was not there"
        );
        assert_eq!(table.filters[0][last], TOMBSTONE_FILTER);
        // `insert` goes through the recording walk, which offers the tombstoned last slot.
        match table.is_present_recording::<false, true>(&key) {
            SearchResult::NotFoundFirstEmpty(bucket, slot, level, _) => assert_eq!(
                (bucket, slot, level),
                (0, last as u32, 0),
                "the recording walk passed over the tombstone"
            ),
            _ => panic!("the recording walk found no room"),
        }
        table.insert(key, value).expect("insert onto the tombstone");
        assert_eq!(position_of(&table, &key), (0, last as u32));
        assert_eq!(table.get(&key), Some(&value));
        assert_eq!(position_of(&table, &key_of(B as u64)), (1, 0));

        // The slot is live again, so the next key goes past the bucket as before.
        let (key, value) = (key_of(1_001), value_of(1_001));
        table
            .insert(key, value)
            .expect("insert past the refilled bucket");
        assert_eq!(position_of(&table, &key), (1, 1));
        assert_eq!(table.get(&key), Some(&value));
        assert_eq!(table.insert(key, value), Err(InsertError::AlreadyPresent));
    }

    /// Bucket 0 with a cleared mid-bucket slot and a tombstoned last one: the recording walk
    /// takes the lower hole first.
    #[test]
    fn a_mid_bucket_hole_is_taken_before_a_tombstoned_last_slot() {
        let filled = B as u64 + 1;
        let mut table = chain_table(filled);
        let hole = hole_slot();
        let last = B - 1;
        assert_eq!(
            table.remove(&key_of(hole as u64)),
            Some(value_of(hole as u64))
        );
        assert_eq!(
            table.remove(&key_of(last as u64)),
            Some(value_of(last as u64))
        );

        let (key, value) = (key_of(1_000), value_of(1_000));
        match table.is_present_recording::<false, true>(&key) {
            SearchResult::NotFoundFirstEmpty(bucket, slot, level, _) => assert_eq!(
                (bucket, slot, level),
                (0, hole as u32, 0),
                "the recording walk skipped the lower hole"
            ),
            _ => panic!("the recording walk found no room"),
        }
        table.insert(key, value).expect("insert into the hole");
        assert_eq!(position_of(&table, &key), (0, hole as u32));
        assert_eq!(table.get(&key), Some(&value));
        assert_eq!(
            table.filters[0][last], TOMBSTONE_FILTER,
            "the tombstone was written over"
        );
        assert_eq!(position_of(&table, &key_of(B as u64)), (1, 0));
    }

    /// Bucket 0 full but holed, bucket 1 with room: the walk must return the recorded hole,
    /// not the level where it stopped.
    #[test]
    fn insert_reuses_a_hole_over_a_deeper_bucket_with_room() {
        let filled = B as u64 + 1;
        check_hole_is_reused(chain_table(filled), filled);
    }

    /// Buckets 0 and 1 full, bucket 0 holed: the recorded hole must survive to the walk's far
    /// end (the frontier exit, or the first virgin level under absent-stop) without being replaced.
    #[test]
    fn insert_reuses_a_hole_when_every_deeper_bucket_overflowed() {
        let filled = 2 * B as u64;
        check_hole_is_reused(chain_table(filled), filled);
    }

    /// Zeroing `plain_insert_frontier` is what puts the inserts on the recording walk, so a
    /// hole punched behind its back must read as no hole at all: the plain walk takes the first
    /// bucket with room, the same slot it took before. `check_hole_is_reused` is this test's
    /// twin, with the delete announced the way `remove` announces it.
    #[test]
    fn a_hole_no_delete_announced_is_not_reused() {
        let filled = B as u64 + 1;
        let mut table = chain_table(filled);
        let hole = hole_slot();
        // The byte on its own, deliberately not through `remove`, which announces it.
        table.filters[0][hole] = EMPTY_FILTER;

        let (key, value) = (key_of(1_000), value_of(1_000));
        table
            .insert(key, value)
            .expect("insert past the holed bucket");
        assert_eq!(
            position_of(&table, &key),
            (1, 1),
            "the insert took a hole no delete had announced"
        );
        assert_eq!(table.get(&key), Some(&value));
        assert_eq!(
            table.filters[0][hole], EMPTY_FILTER,
            "the hole was written over"
        );
    }

    /// The over-read (16 lanes over 8-slot buckets) covers the next bucket, which on level 0
    /// is another key's home: a hole there must not be recorded as room for this key.
    #[test]
    fn an_over_read_neighbours_hole_is_not_recorded_as_room() {
        type Narrow = MultiTableFiltered<KEY_LEN, VAL_LEN, 4, HasherBuilder, 8, 16>;

        /// The level-0 bucket a key homes in, while that bucket still has room.
        fn home_bucket(table: &Narrow, key: &[u8; KEY_LEN]) -> Option<u32> {
            match table.is_present::<false, true>(key) {
                SearchResult::NotFoundFirstEmpty(bucket, _, 0, _) => Some(bucket),
                _ => None,
            }
        }

        let mut table = Narrow::new_from_cnts(vec![2, 1, 1, 1]);

        let i = (1_000u64..1_100)
            .find(|&i| home_bucket(&table, &key_of(i)) == Some(0))
            .expect("some key homes on level 0's bucket 0");
        let (key, value) = (key_of(i), value_of(i));

        // Bucket 1 is filled with real keys, so the hole in it can be left by a real `remove`.
        let mut neighbours = Vec::new();
        for j in 0..1_000u64 {
            if neighbours.len() == 8 {
                break;
            }
            if home_bucket(&table, &key_of(j)) == Some(1) {
                table
                    .insert(key_of(j), value_of(j))
                    .expect("fill level 0's bucket 1");
                neighbours.push(j);
            }
        }
        assert_eq!(neighbours.len(), 8, "level 0's bucket 1 never filled up");

        // Bucket 0 full, fabricated: its keys are all-zero and match no probed key, so only its
        // filter bytes matter here.
        table.filters[0] = [MIN_KEY_FILTER + 5; 8];
        assert!(
            matches!(
                table.is_present::<false, false>(&key_of(neighbours[3])),
                SearchResult::Found(1, 3)
            ),
            "the neighbour's fourth key is not in the slot this holes"
        );
        assert_eq!(
            table.remove(&key_of(neighbours[3])),
            Some(value_of(neighbours[3]))
        );
        assert_eq!(table.filters[1][3], EMPTY_FILTER);

        table
            .insert(key, value)
            .expect("insert past the full level 0");
        assert!(
            matches!(
                table.is_present::<false, false>(&key),
                SearchResult::Found(2, 0)
            ),
            "key landed off its own chain: level 1's bucket was the first with room"
        );
        assert_eq!(table.get(&key), Some(&value));
    }
}

/// `remove`'s whole contract: what it hands back, what it writes into the slot it frees, and
/// that the room it leaves is offered again without anything else in the cascade moving.
#[cfg(test)]
mod remove_tests {
    use crate::*;

    const KEY_LEN: usize = 8;
    const VAL_LEN: usize = 4;
    const B: usize = DEFAULT_FILTERED_BUCKET_SIZE;

    /// Four levels, as in `first_available_tests`: the variant features seed four levels'
    /// buckets up front, so a shallower cascade would index past its array.
    type Table = MultiTableFiltered<KEY_LEN, VAL_LEN, 4>;
    /// The default depth, for the cases that want the cascade `with_capacity` builds.
    type Cascade = MultiTableFiltered<KEY_LEN, VAL_LEN>;

    /// Distinct keys from a bijective scramble of `i`, as elsewhere.
    fn key_of(i: u64) -> [u8; KEY_LEN] {
        i.wrapping_mul(0x9E37_79B9_7F4A_7C15).to_ne_bytes()
    }

    fn value_of(i: u64) -> [u8; VAL_LEN] {
        (i as u32).to_ne_bytes()
    }

    /// splitmix64, so the churn below draws the same third on every run.
    fn scramble(i: u64) -> u64 {
        let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// One bucket per level, so every key's chain is 0 -> 1 -> 2 -> 3 under all three variant
    /// features; key `i` lands in bucket `i / B`, slot `i % B`.
    fn chain_table(filled: u64) -> Table {
        let mut table = Table::new_from_cnts(vec![1, 1, 1, 1]);
        for i in 0..filled {
            table
                .insert(key_of(i), value_of(i))
                .unwrap_or_else(|e| panic!("fill insert {i} failed: {e}"));
        }
        table
    }

    fn position_of(table: &Table, key: &[u8; KEY_LEN]) -> (u32, u32) {
        match table.is_present::<false, false>(key) {
            SearchResult::Found(bucket, slot) => (bucket, slot),
            _ => panic!("key not found"),
        }
    }

    /// A slot to hole: mid-bucket, and never the last one, whose byte carries the stop.
    fn hole_slot() -> usize {
        (B / 2).min(B.saturating_sub(2))
    }

    /// The value comes back, the key stops reading, a second `remove` finds nothing and every
    /// other key is where it was. The table keeps no key count, so there is none to check.
    #[test]
    fn remove_hands_back_the_value_and_leaves_the_rest_alone() {
        const N: u64 = 20_000;

        let mut table = Cascade::with_capacity(N, 0.8);
        for i in 0..N {
            table
                .insert(key_of(i), value_of(i))
                .unwrap_or_else(|e| panic!("insert {i} of {N} failed: {e}"));
        }

        for i in (0..N).step_by(97) {
            assert_eq!(
                table.remove(&key_of(i)),
                Some(value_of(i)),
                "remove of key {i} did not hand back its value"
            );
            assert_eq!(
                table.get(&key_of(i)),
                None,
                "key {i} still reads after remove"
            );
            assert_eq!(
                table.remove(&key_of(i)),
                None,
                "the second remove of key {i} found something"
            );
        }

        for i in (0..N).filter(|i| !i.is_multiple_of(97)) {
            assert_eq!(
                table.get(&key_of(i)),
                Some(&value_of(i)),
                "key {i} was lost around a remove"
            );
        }
        assert_eq!(
            table.remove(&key_of(N + 1)),
            None,
            "remove found a key that was never inserted"
        );
    }

    /// A mid-bucket slot goes back to `EMPTY_FILTER`, and the next key that hashes into the
    /// bucket takes it, filter byte and all.
    #[test]
    fn a_removed_mid_bucket_slot_is_empty_and_taken_again() {
        let filled = B as u64 + 1;
        let mut table = chain_table(filled);
        let hole = hole_slot();

        assert_eq!(position_of(&table, &key_of(hole as u64)), (0, hole as u32));
        assert_eq!(
            table.remove(&key_of(hole as u64)),
            Some(value_of(hole as u64))
        );
        assert_eq!(
            table.filters[0][hole], EMPTY_FILTER,
            "a mid-bucket delete left something in the filter byte"
        );

        let (key, value) = (key_of(1_000), value_of(1_000));
        let key_filter = match table.is_present_recording::<false, true>(&key) {
            SearchResult::NotFoundFirstEmpty(0, slot, 0, key_filter) if slot == hole as u32 => {
                key_filter
            }
            _ => panic!("the recording walk passed over the freed slot"),
        };
        assert!(
            key_filter >= MIN_KEY_FILTER,
            "a key's byte reads as deleted"
        );

        table
            .insert(key, value)
            .expect("insert into the freed slot");
        assert_eq!(position_of(&table, &key), (0, hole as u32));
        assert_eq!(table.get(&key), Some(&value));
        assert_eq!(
            table.filters[0][hole], key_filter,
            "the reused slot did not take the new key's filter byte"
        );
        for i in (0..filled).filter(|&i| i != hole as u64) {
            assert_eq!(
                table.get(&key_of(i)),
                Some(&value_of(i)),
                "fill key {i} lost"
            );
        }
    }

    /// A deleted last slot keeps the byte that says the bucket once filled, so the chain behind
    /// it is still walked: the deeper copy of a key still refuses a duplicate and still reads.
    #[test]
    fn a_removed_last_slot_tombstones_and_the_chain_past_it_is_still_searched() {
        let filled = B as u64 + 1;
        let mut table = chain_table(filled);
        let last = B - 1;
        // Key `last` is bucket 0's last slot; key `B` is the one that overflowed into bucket 1.
        let deep = B as u64;
        assert_eq!(position_of(&table, &key_of(last as u64)), (0, last as u32));
        assert_eq!(position_of(&table, &key_of(deep)), (1, 0));

        assert_eq!(
            table.remove(&key_of(last as u64)),
            Some(value_of(last as u64))
        );
        assert_eq!(
            table.filters[0][last], TOMBSTONE_FILTER,
            "a deleted last slot went back to empty"
        );
        assert_eq!(table.get(&key_of(last as u64)), None);

        assert_eq!(
            table.insert(key_of(deep), value_of(deep)),
            Err(InsertError::AlreadyPresent),
            "the tombstone stopped the walk short of the deeper copy of the key"
        );
        assert_eq!(table.get(&key_of(deep)), Some(&value_of(deep)));

        // A fresh key on the same chain lands on the tombstone.
        let (key, value) = (key_of(1_000), value_of(1_000));
        table.insert(key, value).expect("insert onto the tombstone");
        assert_eq!(position_of(&table, &key), (0, last as u32));
        assert_eq!(table.get(&key), Some(&value));
        assert_eq!(
            table.get(&key_of(deep)),
            Some(&value_of(deep)),
            "refilling the tombstoned slot lost the key behind it"
        );
    }

    /// The plain walk's mirror of the frontier: equal to it until the first `remove`, and zero
    /// from then on, across further inserts and every widening of `first_empty_level`.
    #[test]
    fn the_first_remove_hands_the_inserts_over_for_good() {
        let filled = B as u64 + 1;
        let mut table = chain_table(filled);
        assert!(
            table.first_empty_level >= 2,
            "the fill never widened the frontier, so there is nothing to mirror"
        );
        assert_eq!(
            table.plain_insert_frontier as usize, table.first_empty_level,
            "the mirror drifted from the frontier before any delete"
        );

        assert!(table.remove(&key_of(0)).is_some());
        assert_eq!(
            table.plain_insert_frontier, 0,
            "the delete did not hand the inserts over"
        );

        // Enough keys to take the freed slot, fill the level the frontier already covers and
        // widen it once more; the mirror stays zero throughout.
        let widened_at = table.first_empty_level;
        for i in 1_000..1_000 + 2 * B as u64 - 2 {
            table
                .insert(key_of(i), value_of(i))
                .unwrap_or_else(|e| panic!("insert {i} after the delete failed: {e}"));
            assert_eq!(
                table.plain_insert_frontier, 0,
                "insert {i} put the mirror back"
            );
        }
        assert!(
            table.first_empty_level > widened_at,
            "the frontier never widened after the delete, so nothing was proven"
        );
        assert_eq!(table.plain_insert_frontier, 0);
    }

    /// Churn at load: a third of the keys out and as many fresh ones in, round after round,
    /// with every live key still reading its value, every dead one gone, exactly one filter
    /// byte live per live key, and a cascade that settles instead of marching deeper.
    #[test]
    fn churn_at_load_keeps_every_live_key_and_holds_the_frontier() {
        const N: u64 = 20_000;
        const UTILIZATION: f64 = 0.8;
        const ROUNDS: u64 = 16;
        /// Levels of slack over the no-delete fill. Room a delete leaves only serves keys whose
        /// chain crosses that bucket, so churn strands some keys deeper than a fresh fill puts
        /// them; two levels is what the widths and variants here actually take, and the
        /// settling check below is the sharper half of this: past the halfway round the
        /// frontier stops moving, where leaked room would push it deeper every round until the
        /// cascade ran out of space.
        const SLACK: usize = 2;

        let mut reference = Cascade::with_capacity(N, UTILIZATION);
        for i in 0..N {
            reference
                .insert(key_of(i), value_of(i))
                .unwrap_or_else(|e| panic!("reference insert {i} failed: {e}"));
        }
        let bound = reference.first_empty_level + SLACK;

        let mut table = Cascade::with_capacity(N, UTILIZATION);
        for i in 0..N {
            table
                .insert(key_of(i), value_of(i))
                .unwrap_or_else(|e| panic!("insert {i} of {N} failed: {e}"));
        }

        let mut live: Vec<u64> = (0..N).collect();
        let mut dead: Vec<u64> = Vec::new();
        let mut next = N;
        let mut settled = 0;

        for round in 0..ROUNDS {
            let (kept, removed): (Vec<u64>, Vec<u64>) = live
                .iter()
                .copied()
                .partition(|&i| !scramble(i ^ (round << 40)).is_multiple_of(3));
            for &i in &removed {
                assert_eq!(
                    table.remove(&key_of(i)),
                    Some(value_of(i)),
                    "round {round}: remove of live key {i} came back empty"
                );
            }

            live = kept;
            for _ in 0..removed.len() {
                table
                    .insert(key_of(next), value_of(next))
                    .unwrap_or_else(|e| panic!("round {round}: refill insert {next} failed: {e}"));
                live.push(next);
                next += 1;
            }
            dead.extend(removed);

            for &i in &live {
                assert_eq!(
                    table.get(&key_of(i)),
                    Some(&value_of(i)),
                    "round {round}: live key {i} lost"
                );
            }
            for &i in &dead {
                assert_eq!(
                    table.get(&key_of(i)),
                    None,
                    "round {round}: removed key {i} still reads"
                );
            }
            // One live filter byte per live key: a delete frees its slot and no other.
            let live_bytes: usize = table
                .filters
                .iter()
                .take(table.total_buckets())
                .map(|bucket| bucket.iter().filter(|&&f| f >= MIN_KEY_FILTER).count())
                .sum();
            assert_eq!(
                live_bytes,
                live.len(),
                "round {round}: {live_bytes} filter bytes live against {} keys",
                live.len()
            );

            assert!(
                table.first_empty_level <= bound,
                "round {round}: the cascade reached level {} against a fill that stops at {}",
                table.first_empty_level,
                bound - SLACK
            );
            if round == ROUNDS / 2 {
                settled = table.first_empty_level;
            }
            if round > ROUNDS / 2 {
                assert_eq!(
                    table.first_empty_level, settled,
                    "round {round}: the cascade is still deepening halfway through the churn"
                );
            }
        }
        println!(
            "churn: {ROUNDS} rounds over {N} keys, frontier {} against {} without deletes",
            table.first_empty_level,
            bound - SLACK
        );
    }

    /// Keys held per level, counted from filter bytes, for the churn below to report its shape.
    fn level_occupancy(table: &Cascade) -> Vec<usize> {
        table
            .levels_meta
            .iter()
            .map(|&(width, offset)| (level_bucket_cnt(width), offset))
            .take_while(|&(cnt, _)| cnt != 0)
            .map(|(cnt, offset)| {
                (offset as usize..(offset + cnt) as usize)
                    .map(|bucket| {
                        table.filters[bucket]
                            .iter()
                            .filter(|&&f| f >= MIN_KEY_FILTER)
                            .count()
                    })
                    .sum::<usize>()
            })
            .collect()
    }

    /// A table filled to its sizing target, then one key in twenty out and as many fresh ones
    /// in, round after round: every insert should find room, live keys should read and removed
    /// ones should not. The inserts do not all find room, which is why this is ignored. Room a
    /// delete frees at level 0 is taken again only by a newcomer whose own level-0 bucket it
    /// sits in, so a round's reuse falls short of its arrivals wherever holes are scarce, and
    /// level 0 drains until its holes are plentiful enough for the reuse to match what the
    /// deletes take out. That equilibrium keeps more keys below level 0 than a fresh fill
    /// does, and the tail `with_capacity` builds at 0.8 is sized for the fresh fill's spill:
    /// 2224 slots past level 0 for 20000 keys at bucket size 16, 4144 at 8. Under agile the
    /// cascade runs out at round 35 (1.8 turnovers) at bucket size 16, with level 0 drained
    /// from 19969 keys to 17415, and at round 13 at bucket size 8; the sizing variants run out
    /// between rounds 9 and 19. The smaller the churn fraction, the scarcer the holes each
    /// round and the deeper the equilibrium: a third out per round, as the test above churns,
    /// settles without overflowing where a twentieth does not.
    #[test]
    #[ignore = "5% churn at capacity drains level 0 into tail levels sized for a fresh fill; the cascade runs out of room within two turnovers (round 35 at bucket size 16, round 13 at 8)"]
    fn churn_at_capacity_by_a_twentieth_a_round_never_runs_out_of_room() {
        const N: u64 = 20_000;
        const UTILIZATION: f64 = 0.8;
        /// Three turnovers of the table.
        const ROUNDS: u64 = 60;
        const ONE_IN: u64 = 20;

        let mut table = Cascade::with_capacity(N, UTILIZATION);
        for i in 0..N {
            table
                .insert(key_of(i), value_of(i))
                .unwrap_or_else(|e| panic!("insert {i} of {N} failed: {e}"));
        }
        let fresh = level_occupancy(&table);

        let mut live: Vec<u64> = (0..N).collect();
        let mut dead: Vec<u64> = Vec::new();
        let mut next = N;

        for round in 0..ROUNDS {
            let (kept, removed): (Vec<u64>, Vec<u64>) = live
                .iter()
                .copied()
                .partition(|&i| !scramble(i ^ (round << 40)).is_multiple_of(ONE_IN));
            for &i in &removed {
                assert_eq!(
                    table.remove(&key_of(i)),
                    Some(value_of(i)),
                    "round {round}: remove of live key {i} came back empty"
                );
            }
            live = kept;

            // A full cascade is the failure under test, so `Full` is reported with the round
            // and the shape it happened at; any other refusal is a bug of its own.
            let mut refused = None;
            let mut full = false;
            for _ in 0..removed.len() {
                let (key, value) = (key_of(next), value_of(next));
                match table.insert(key, value) {
                    Ok(()) => {
                        live.push(next);
                        next += 1;
                    }
                    Err(InsertError::Full) => {
                        full = true;
                        break;
                    }
                    Err(e) => {
                        refused = Some((next, e));
                        break;
                    }
                }
            }
            if let Some((i, e)) = refused {
                panic!("round {round}: refill insert {i} failed: {e}");
            }
            assert!(
                !full,
                "round {round}: the cascade ran out of room after {:.2} turnovers of {N} keys, \
                 holding {:?} per level against {fresh:?} after the fill",
                (round + 1) as f64 / ONE_IN as f64,
                level_occupancy(&table)
            );
            dead.extend(removed);

            for &i in &live {
                assert_eq!(
                    table.get(&key_of(i)),
                    Some(&value_of(i)),
                    "round {round}: live key {i} lost"
                );
            }
            for &i in &dead {
                assert_eq!(
                    table.get(&key_of(i)),
                    None,
                    "round {round}: removed key {i} still reads"
                );
            }
        }
        println!(
            "churn: {ROUNDS} rounds of one in {ONE_IN} over {N} keys, holding {:?} per level \
             against {fresh:?} after the fill",
            level_occupancy(&table)
        );
    }

    /// A bucket that never filled: its last byte is still zero, so `remove` leaves a plain hole
    /// and the next insert takes that lane rather than the tail.
    #[test]
    fn a_hole_in_a_never_filled_bucket_is_taken_before_the_tail() {
        let mut table = chain_table(2);
        let last = B - 1;
        assert_eq!(
            table.filters[0][last], EMPTY_FILTER,
            "the bucket filled up, so it is not the case under test"
        );

        assert_eq!(table.remove(&key_of(0)), Some(value_of(0)));
        assert_eq!(table.filters[0][0], EMPTY_FILTER);
        assert_eq!(
            table.filters[0][last], EMPTY_FILTER,
            "the delete wrote a tombstone into a bucket that never filled"
        );

        let (key, value) = (key_of(1_000), value_of(1_000));
        table
            .insert(key, value)
            .expect("insert into the freed lane");
        assert_eq!(
            position_of(&table, &key),
            (0, 0),
            "the insert took the tail rather than the hole"
        );
        assert_eq!(table.get(&key), Some(&value));
        assert_eq!(table.get(&key_of(1)), Some(&value_of(1)));
    }
}

/// `upsert` on a cascade it has to grow itself, and over keys it already holds.
#[cfg(test)]
mod upsert_tests {
    use crate::*;

    const KEY_LEN: usize = 4;
    const VAL_LEN: usize = 4;
    const CAPACITY: u64 = 4096;
    const UTILIZATION: f64 = 0.8;
    /// Far past any `i` the test uses, so a second pass writes a value the first never wrote.
    const SECOND_PASS: u64 = 1_000_000;

    type Table = MultiTableFiltered<KEY_LEN, VAL_LEN>;

    /// Distinct keys from a bijective scramble of `i`, as elsewhere.
    fn key_of(i: u64) -> [u8; KEY_LEN] {
        (i as u32).wrapping_mul(0x9E37_79B9).to_ne_bytes()
    }

    fn value_of(i: u64) -> [u8; VAL_LEN] {
        (i as u32).to_ne_bytes()
    }

    /// Keys `insert` gets in before the cascade first answers `Full`.
    fn insert_capacity() -> u64 {
        let mut table = Table::with_capacity(CAPACITY, UTILIZATION);
        let mut filled = 0;
        while table.insert(key_of(filled), value_of(filled)).is_ok() {
            filled += 1;
        }
        filled
    }

    #[test]
    fn upsert_fills_like_insert_and_then_overwrites_in_place() {
        let filled = insert_capacity();

        assert!(filled > 0, "the reference fill took no keys at all");

        let mut table = Table::with_capacity(CAPACITY, UTILIZATION);
        for i in 0..filled {
            table
                .upsert(key_of(i), value_of(i))
                .unwrap_or_else(|e| panic!("upsert {i} of {filled} failed: {e}"));
        }
        for i in 0..filled {
            assert_eq!(
                table.get(&key_of(i)),
                Some(&value_of(i)),
                "key {i} is missing after a fill `insert` manages"
            );
        }

        for i in 0..filled {
            table
                .upsert(key_of(i), value_of(i + SECOND_PASS))
                .unwrap_or_else(|e| panic!("overwriting upsert {i} failed: {e}"));
        }
        for i in 0..filled {
            assert_eq!(
                table.get(&key_of(i)),
                Some(&value_of(i + SECOND_PASS)),
                "key {i} kept its first value"
            );
        }

        // A fresh key on a full table, refused by `grow_search_depth_and_insert`.
        assert_eq!(
            table.upsert(key_of(filled), value_of(filled)),
            Err(InsertError::Full),
            "upsert took a key the full cascade has no room for"
        );
        assert_eq!(table.get(&key_of(filled)), None);
    }

    /// A cascade far smaller than the keys offered: each refusal is `Full`, writes nothing and
    /// leaves the frontier alone, and every key accepted before or after it still reads back.
    #[test]
    fn a_full_cascade_answers_full_and_keeps_every_earlier_key() {
        let mut table = Table::new_from_cnts(vec![4, 2, 1]);
        let slots = table.total_buckets() * DEFAULT_FILTERED_BUCKET_SIZE;

        let mut stored = Vec::new();
        let mut refused = Vec::new();
        for i in 0..(4 * slots) as u64 {
            let filters = table.filters.clone();
            let frontier = (table.first_empty_level, table.plain_insert_frontier);
            match table.insert(key_of(i), value_of(i)) {
                Ok(()) => stored.push(i),
                Err(e) => {
                    assert_eq!(e, InsertError::Full, "key {i} refused for the wrong reason");
                    assert!(
                        table.filters == filters,
                        "the refused insert of key {i} wrote"
                    );
                    assert_eq!(
                        (table.first_empty_level, table.plain_insert_frontier),
                        frontier,
                        "the refused insert of key {i} moved the frontier"
                    );
                    refused.push(i);
                }
            }
        }

        assert!(!refused.is_empty(), "{} keys fit {slots} slots", 4 * slots);
        assert!(stored.len() <= slots);
        assert_eq!(
            table.first_empty_level, 3,
            "the fill never reached the last level"
        );
        for &i in &stored {
            assert_eq!(table.get(&key_of(i)), Some(&value_of(i)), "key {i} lost");
        }
        for &i in &refused {
            assert_eq!(table.get(&key_of(i)), None, "refused key {i} reads");
            assert_eq!(
                table.upsert(key_of(i), value_of(i)),
                Err(InsertError::Full),
                "upsert found room for key {i} that insert did not"
            );
        }
        assert_eq!(
            table.insert(key_of(stored[0]), value_of(stored[0])),
            Err(InsertError::AlreadyPresent)
        );
    }
}
