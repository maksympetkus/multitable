//! Level sizing: each level is sized from an estimated overflow quantile, not a proven bound.

pub mod driver {

    use super::search::Evaluator;

    const fn parse_ratio(s: &str) -> f64 {
        let b = s.as_bytes();
        let mut mantissa = 0u64;
        let mut divisor = 1.0f64;
        let mut seen_dot = false;
        let mut digits = 0;
        let mut i = 0;
        while i < b.len() {
            let c = b[i];
            if c == b'.' {
                assert!(
                    !seen_dot && digits > 0 && i + 1 < b.len(),
                    "MT_PAPER_RATIO must be digits with at most one interior dot, like 1.01"
                );
                seen_dot = true;
            } else {
                assert!(
                    c.is_ascii_digit(),
                    "MT_PAPER_RATIO must be digits with at most one interior dot, like 1.01"
                );
                // 15 digits keeps both mantissa and divisor exactly representable in f64.
                assert!(digits < 15, "MT_PAPER_RATIO has more than 15 digits");
                mantissa = mantissa * 10 + (c - b'0') as u64;
                if seen_dot {
                    divisor *= 10.0;
                }
                digits += 1;
            }
            i += 1;
        }
        assert!(digits > 0, "MT_PAPER_RATIO is empty");
        let ratio = mantissa as f64 / divisor;
        assert!(ratio >= 1.0, "MT_PAPER_RATIO must be at least 1");
        ratio
    }

    /// Default `ratio`: `MT_PAPER_RATIO` at build time, else 1.01, the paper's ratio.
    pub const PAPER_RATIO: f64 = match option_env!("MT_PAPER_RATIO") {
        Some(s) => parse_ratio(s),
        None => 1.01,
    };

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Params {
        /// Slots per bucket.
        pub s: u64,
        /// Target average load factor, keys per slot, in `(0, 1]`.
        pub alpha: f64,
        /// Per-level failure budget.
        pub delta: f64,
        /// Progressive overfill factor for the upper levels; `1.0` disables it.
        pub ratio: f64,
        /// Additive load boost applied to every retarget.
        pub a_increment: f64,
        /// Ask the Chernoff slack bound for a second opinion on each level's bucket count.
        pub use_bound: bool,
        /// Run the full bound search instead of the single-probe shortcut.
        pub strict_bound_search: bool,
        pub max_levels: usize,
        /// Round levels down to a power of two, tolerating this much overshoot; `None` disables it.
        pub pow2_tolerance: Option<f64>,
        /// Round only the first level; inert without `pow2_tolerance`.
        pub pow2_first_level_only: bool,
    }

    impl Default for Params {
        fn default() -> Self {
            Self {
                s: 8,
                alpha: 0.9,
                delta: 1.0 / 128.0,
                ratio: PAPER_RATIO,
                a_increment: 0.00,
                use_bound: true,
                strict_bound_search: false,
                max_levels: 100,
                pow2_tolerance: None,
                pow2_first_level_only: false,
            }
        }
    }

    /// Which rule produced a level's bucket count.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Rule {
        First,
        GeneralQuantile,
        GeneralBound,
        /// A level that absorbs all but at most `s` keys.
        Truncation,
        SingleBucket,
        /// The single bucket that closes a truncated cascade.
        FinalBucket,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Level {
        pub buckets: u64,
        pub keys_in: u64,
        pub target_load: Option<f64>,
        /// Keys planned to cascade to the next level.
        pub residue: u64,
        pub rule: Rule,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct Sizing {
        /// Bucket count per level, top level first.
        pub sizes: Vec<u64>,
        pub levels: Vec<Level>,
        pub buckets: u64,
        /// Total slots.
        pub capacity: u64,
        /// Realized load factor, `q / capacity`.
        pub load: f64,
        /// Bucket budget the driver was working against, `ceil(q / (s alpha))`.
        pub budget: u64,
    }

    impl Sizing {
        /// Number of levels, which is also the probe count of a negative lookup.
        pub fn depth(&self) -> usize {
            self.sizes.len()
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum SizingError {
        BadParams(&'static str),
        FirstLevelExceedsBudget {
            buckets: u64,
            budget: u64,
        },
        TooManyLevels {
            sizes: Vec<u64>,
        },
    }

    impl std::fmt::Display for SizingError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                SizingError::BadParams(m) => write!(f, "bad parameters: {m}"),
                SizingError::FirstLevelExceedsBudget { buckets, budget } => {
                    write!(f, "first level wants {buckets} buckets, budget is {budget}")
                }
                SizingError::TooManyLevels { sizes } => {
                    write!(f, "cascade did not terminate after {} levels", sizes.len())
                }
            }
        }
    }

    impl std::error::Error for SizingError {}

    /// Slots per bucket past which the `O(s)` tail and `O(s^2)` Panjer head cost too much.
    const MAX_SLOTS_PER_BUCKET: u64 = 1 << 20;

    fn validate(params: &Params) -> Result<(), SizingError> {
        if params.s == 0 {
            return Err(SizingError::BadParams("s must be at least 1"));
        }
        if params.s > MAX_SLOTS_PER_BUCKET {
            return Err(SizingError::BadParams("s exceeds 2^20 slots per bucket"));
        }
        if !(params.alpha > 0.0 && params.alpha <= 1.0) {
            return Err(SizingError::BadParams("alpha must lie in (0, 1]"));
        }
        if !(params.delta > 0.0 && params.delta < 1.0) {
            return Err(SizingError::BadParams("delta must lie in (0, 1)"));
        }
        if params.ratio < 1.0 {
            return Err(SizingError::BadParams("ratio must be at least 1"));
        }
        if params
            .pow2_tolerance
            .is_some_and(|t| !(0.0..1.0).contains(&t))
        {
            return Err(SizingError::BadParams("pow2_tolerance must lie in [0, 1)"));
        }
        Ok(())
    }

    /// Largest power of two at or below `n * (1 + tol)`, so a level never overruns its budget.
    fn pow2_at_or_below(n: u64, tol: f64) -> u64 {
        let ceiling = ((n as f64) * (1.0 + tol)).floor().max(1.0);
        1u64 << (ceiling.min(u64::MAX as f64) as u64).ilog2()
    }

    pub fn level_sizes(q: u64, params: &Params) -> Result<Sizing, SizingError> {
        validate(params)?;
        let mut ev = Evaluator::new(params.s, params.delta);
        level_sizes_with(q, params, &mut ev)
    }

    pub fn level_sizes_with(
        q: u64,
        params: &Params,
        ev: &mut Evaluator,
    ) -> Result<Sizing, SizingError> {
        validate(params)?;
        if ev.s() != params.s || ev.delta() != params.delta {
            return Err(SizingError::BadParams(
                "evaluator was built for a different (s, delta)",
            ));
        }
        let s = params.s;
        if q == 0 {
            return Ok(Sizing {
                sizes: vec![],
                levels: vec![],
                buckets: 0,
                capacity: 0,
                load: 0.0,
                budget: 0,
            });
        }

        let budget = (q as f64 / s as f64 / params.alpha).ceil() as u64;

        if q <= s {
            return Ok(Sizing {
                sizes: vec![1],
                levels: vec![Level {
                    buckets: 1,
                    keys_in: q,
                    target_load: None,
                    residue: 0,
                    rule: Rule::SingleBucket,
                }],
                buckets: 1,
                capacity: s,
                load: q as f64 / s as f64,
                budget,
            });
        }

        let deep_tolerance = params
            .pow2_tolerance
            .filter(|_| !params.pow2_first_level_only);
        let fit = |n: u64| match deep_tolerance {
            Some(tol) => pow2_at_or_below(n, tol),
            None => n,
        };
        let fit_up = |n: u64| match deep_tolerance {
            Some(_) => n.next_power_of_two(),
            None => n,
        };

        // Level 1 is overfilled: it is the cheapest to probe. The cap keeps it below a full load.
        let a1 = (params.alpha * params.ratio + params.a_increment).min(0.99);
        let asked1 = ev.buckets_for_load(q, a1);
        let n1 = match params.pow2_tolerance {
            Some(tol) => pow2_at_or_below(asked1, tol),
            None => asked1,
        };
        // The residue handed down is the target remainder, not the quantile estimate.
        let mut elements = if n1 == asked1 {
            (q as f64 - n1 as f64 * a1 * s as f64).ceil().max(0.0) as u64
        } else {
            // A level `fit` moved does not run at `a1`, so it sheds the quantile instead.
            ev.q_hat(n1, q)
        };
        let mut buckets_left = n1;
        let mut sizes = vec![n1];
        let mut levels = vec![Level {
            buckets: n1,
            keys_in: q,
            target_load: Some(a1),
            residue: elements,
            rule: Rule::First,
        }];
        if n1 >= budget {
            return Err(SizingError::FirstLevelExceedsBudget {
                buckets: n1,
                budget,
            });
        }

        while elements >= 1 {
            if sizes.len() > params.max_levels {
                return Err(SizingError::TooManyLevels { sizes });
            }

            if buckets_left <= 2 || elements <= s {
                let keys_in = elements;
                elements = elements.saturating_sub(s);
                sizes.push(1);
                buckets_left = 1;
                levels.push(Level {
                    buckets: 1,
                    keys_in,
                    target_load: None,
                    residue: elements,
                    rule: Rule::SingleBucket,
                });
                continue;
            }

            if (elements as f64) / (q as f64) < 0.02
                && (params.ratio > 1.0 || params.a_increment > 0.0)
            {
                let used: u64 = sizes.iter().sum();
                // Cap the search by the budget left to spend, not by `elements`.
                let cap = ((1.005 * budget as f64).floor() as u64).saturating_sub(used + 1);
                // Rounds up: rounding down would leave the closing bucket more than `s` keys.
                let truncation = ev.buckets_for_max_overflow(elements, s, cap).map(fit_up);
                if let Some(last) =
                    truncation.filter(|last| ((last + 1 + used) as f64) / (budget as f64) <= 1.005)
                {
                    levels.push(Level {
                        buckets: last,
                        keys_in: elements,
                        target_load: None,
                        residue: s,
                        rule: Rule::Truncation,
                    });
                    levels.push(Level {
                        buckets: 1,
                        keys_in: s,
                        target_load: None,
                        residue: 0,
                        rule: Rule::FinalBucket,
                    });
                    sizes.push(last);
                    sizes.push(1);
                    break;
                }
            }

            let used: u64 = sizes.iter().sum();
            let rem = budget as i64 - used as i64;
            let mut a = if rem == 0 {
                params.alpha
            } else {
                elements as f64 / (s as f64 * rem as f64) * params.ratio + params.a_increment
            };
            a = a.max(0.005).min(params.alpha);

            let keys_in = elements;
            let n_q = ev.buckets_for_load(keys_in, a);
            let mut n = fit(n_q);
            let mut rule = Rule::GeneralQuantile;
            let mut fired = false;
            if params.use_bound {
                let beats = if params.strict_bound_search {
                    ev.buckets_for_load_bound(keys_in, a) > n_q
                } else {
                    ev.bound_beats(keys_in, a, n_q)
                };
                if beats {
                    let n_b = ev.buckets_for_load_bound(keys_in, a);
                    // Compared on the sizes asked for; only the winner is rounded.
                    if n_b > n_q {
                        n = fit(n_b);
                        elements = ev.overflow_bound(n, keys_in).ceil().max(0.0) as u64;
                        rule = Rule::GeneralBound;
                        fired = true;
                    }
                }
            }
            if !fired {
                elements = ev.q_hat(n, keys_in);
            }

            buckets_left = n;
            sizes.push(n);
            levels.push(Level {
                buckets: n,
                keys_in,
                target_load: Some(a),
                residue: elements,
                rule,
            });
        }

        let buckets: u64 = sizes.iter().sum();
        let capacity = buckets.saturating_mul(s);
        Ok(Sizing {
            load: q as f64 / capacity as f64,
            sizes,
            levels,
            buckets,
            capacity,
            budget,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn published_configuration_at_1e5() {
            let p = Params {
                ratio: 1.1,
                ..Params::default()
            };
            let out = level_sizes(100_000, &p).unwrap();
            assert_eq!(out.sizes, vec![7373, 4953, 1379, 50, 1]);
            assert_eq!(out.capacity, 110_048);
            assert!((out.load - 0.9087).abs() < 5e-5, "load = {}", out.load);
            let residues: Vec<u64> = out.levels.iter().map(|l| l.residue).collect();
            assert_eq!(residues, vec![41606, 6817, 199, 8, 0]);
        }

        #[test]
        fn published_configuration_at_1e5_with_a_tighter_budget() {
            let p = Params {
                delta: 2f64.powi(-20),
                ratio: 1.1,
                ..Params::default()
            };
            let out = level_sizes(100_000, &p).unwrap();
            assert_eq!(out.sizes, vec![7238, 4984, 1454, 127, 1]);
            assert_eq!(out.capacity, 110_432);
            assert!((out.load - 0.9055).abs() < 5e-5, "load = {}", out.load);
        }

        #[test]
        fn published_configuration_at_1e4() {
            let p = Params {
                ratio: 1.025,
                ..Params::default()
            };
            let out = level_sizes(10_000, &p).unwrap();
            assert_eq!(out.sizes, vec![1042, 283, 54, 10, 1]);
            assert!((out.load - 0.8993).abs() < 5e-5, "load = {}", out.load);
            let feeds: Vec<u64> = out.levels.iter().map(|l| l.keys_in).collect();
            assert_eq!(feeds, vec![10_000, 2311, 378, 50, 8]);
        }

        fn a1_and_linear_remainder(q: u64, p: &Params, n1: u64) -> (f64, u64) {
            let a1 = (p.alpha * p.ratio + p.a_increment).min(0.99);
            let lin = (q as f64 - n1 as f64 * a1 * p.s as f64).ceil().max(0.0) as u64;
            (a1, lin)
        }

        #[test]
        fn without_pow2_rounding_the_first_level_hands_down_its_target_remainder() {
            let p = Params {
                ratio: 1.1,
                ..Params::default()
            };
            let out = level_sizes(100_000, &p).unwrap();
            let (_, lin) = a1_and_linear_remainder(100_000, &p, out.sizes[0]);
            assert_eq!(out.levels[0].residue, lin);
        }

        #[test]
        fn a_truncated_first_level_hands_down_its_quantile_instead() {
            let p = Params {
                pow2_tolerance: Some(0.01),
                ratio: 1.001,
                ..Params::default()
            };
            let q = 56_089;
            let out = level_sizes(q, &p).unwrap();
            let n1 = out.sizes[0];
            assert_eq!(n1, 4096);
            let mut ev = Evaluator::new(p.s, p.delta);
            assert!(ev.buckets_for_load(q, (p.alpha * p.ratio).min(0.99)) > n1);
            let (_, lin) = a1_and_linear_remainder(q, &p, n1);
            assert_eq!(out.levels[0].residue, ev.q_hat(n1, q));
            assert_eq!((out.levels[0].residue, lin), (23_641, 26_569));
            assert_eq!(out.sizes, vec![4096, 2048, 1024, 512, 1]);
        }

        #[test]
        fn a_first_level_rounded_up_hands_down_its_quantile_too() {
            let p = Params {
                pow2_tolerance: Some(0.01),
                ratio: 1.001,
                ..Params::default()
            };
            let q = 143_997;
            let out = level_sizes(q, &p).unwrap();
            let n1 = out.sizes[0];
            assert_eq!(n1, 16_384);
            let mut ev = Evaluator::new(p.s, p.delta);
            assert!(ev.buckets_for_load(q, (p.alpha * p.ratio).min(0.99)) < n1);
            let (_, lin) = a1_and_linear_remainder(q, &p, n1);
            assert_eq!(out.levels[0].residue, ev.q_hat(n1, q));
            assert!(
                out.levels[0].residue > lin,
                "{} vs {lin}",
                out.levels[0].residue
            );
        }

        #[test]
        fn a_first_level_rounding_changed_nothing_keeps_the_linear_remainder() {
            let p = Params {
                pow2_tolerance: Some(0.01),
                ratio: 1.001,
                ..Params::default()
            };
            let q = 9_224;
            let out = level_sizes(q, &p).unwrap();
            assert_eq!(out.sizes[0], 1024);
            let (_, lin) = a1_and_linear_remainder(q, &p, 1024);
            assert_eq!(out.levels[0].residue, lin);
            let plain = level_sizes(
                q,
                &Params {
                    pow2_tolerance: None,
                    ..p
                },
            )
            .unwrap();
            assert_eq!(out.levels[0].residue, plain.levels[0].residue);
        }

        #[test]
        fn a_truncated_cascade_still_covers_its_keys() {
            let p = Params {
                pow2_tolerance: Some(0.01),
                ..Params::default()
            };
            for q in [30_997u64, 44_281, 56_089, 58_894, 224_357, 942_300] {
                let out = level_sizes(q, &p).unwrap();
                assert!(out.capacity >= q, "q = {q}, capacity = {}", out.capacity);
                assert!(out.load < 1.0, "q = {q}, load = {}", out.load);
                assert!(out.sizes.iter().all(|&n| n.is_power_of_two()), "q = {q}");
            }
        }

        fn first_power_params() -> Params {
            Params {
                pow2_tolerance: Some(0.01),
                pow2_first_level_only: true,
                ..Params::default()
            }
        }

        #[test]
        fn a_first_power_level_rounding_changed_nothing_keeps_the_linear_remainder() {
            let p = Params {
                ratio: 1.001,
                ..first_power_params()
            };
            let q = 9_224;
            let out = level_sizes(q, &p).unwrap();
            assert_eq!(out.sizes[0], 1024);
            let (_, lin) = a1_and_linear_remainder(q, &p, 1024);
            assert_eq!(out.levels[0].residue, lin);
            let plain = level_sizes(
                q,
                &Params {
                    pow2_tolerance: None,
                    ..p
                },
            )
            .unwrap();
            assert_eq!(out.levels[0].residue, plain.levels[0].residue);
        }

        #[test]
        fn a_truncated_first_power_level_hands_down_its_quantile_instead() {
            let p = Params {
                ratio: 1.001,
                ..first_power_params()
            };
            let q = 56_089;
            let out = level_sizes(q, &p).unwrap();
            let n1 = out.sizes[0];
            assert_eq!(n1, 4096);
            let mut ev = Evaluator::new(p.s, p.delta);
            assert!(ev.buckets_for_load(q, (p.alpha * p.ratio).min(0.99)) > n1);
            let (_, lin) = a1_and_linear_remainder(q, &p, n1);
            assert_eq!(out.levels[0].residue, ev.q_hat(n1, q));
            assert_eq!((out.levels[0].residue, lin), (23_641, 26_569));
        }

        #[test]
        fn a_first_power_level_rounded_up_hands_down_its_quantile_too() {
            let p = Params {
                ratio: 1.001,
                ..first_power_params()
            };
            let q = 143_997;
            let out = level_sizes(q, &p).unwrap();
            let n1 = out.sizes[0];
            assert_eq!(n1, 16_384);
            let mut ev = Evaluator::new(p.s, p.delta);
            assert!(ev.buckets_for_load(q, (p.alpha * p.ratio).min(0.99)) < n1);
            let (_, lin) = a1_and_linear_remainder(q, &p, n1);
            assert_eq!(out.levels[0].residue, ev.q_hat(n1, q));
            assert!(
                out.levels[0].residue > lin,
                "{} vs {lin}",
                out.levels[0].residue
            );
        }

        #[test]
        fn first_power_rounds_the_first_level_and_leaves_the_rest_free() {
            let p = first_power_params();
            let mut any_free = false;
            for q in [30_997u64, 44_281, 56_089, 58_894, 224_357, 942_300] {
                let out = level_sizes(q, &p).unwrap();
                assert!(
                    out.sizes[0].is_power_of_two(),
                    "q = {q}, sizes = {:?}",
                    out.sizes
                );
                assert!(out.capacity >= q, "q = {q}, capacity = {}", out.capacity);
                assert!(out.load < 1.0, "q = {q}, load = {}", out.load);
                any_free |= out.sizes[1..].iter().any(|n| !n.is_power_of_two());
            }
            assert!(any_free, "no level below the first escaped the grid");
        }

        #[test]
        fn first_power_shares_the_grid_decision_but_not_the_cascade() {
            let first = first_power_params();
            let every = Params {
                pow2_first_level_only: false,
                ..first
            };
            let mut tails_differ = false;
            for q in [30_997u64, 44_281, 56_089, 58_894, 224_357, 942_300] {
                let a = level_sizes(q, &first).unwrap();
                let b = level_sizes(q, &every).unwrap();
                assert_eq!(a.sizes[0], b.sizes[0], "q = {q}");
                assert_eq!(a.levels[0].residue, b.levels[0].residue, "q = {q}");
                tails_differ |= a.sizes[1..] != b.sizes[1..];
            }
            assert!(
                tails_differ,
                "the two cascades never parted below level one"
            );
        }

        #[test]
        fn every_level_is_at_least_one_bucket_and_the_last_is_one() {
            let p = Params::default();
            for q in [10u64, 1_000, 50_000, 1_000_000, 1_000_000_000] {
                let out = level_sizes(q, &p).unwrap();
                assert!(out.sizes.iter().all(|&n| n >= 1), "q = {q}");
                assert_eq!(*out.sizes.last().unwrap(), 1, "q = {q}");
                assert!(out.capacity >= q, "q = {q}, capacity = {}", out.capacity);
            }
        }

        #[test]
        fn realized_load_stays_near_the_target() {
            let p = Params::default();
            for q in [10_000u64, 100_000, 10_000_000, 1_000_000_000] {
                let out = level_sizes(q, &p).unwrap();
                assert!(
                    out.load > p.alpha - 0.02 && out.load < 1.0,
                    "q = {q}, load = {}",
                    out.load
                );
            }
        }
    }
}

pub mod search {

    use std::collections::HashMap;

    use super::bound::worst_case_unused;
    use super::numerics::norm_ppf;
    use super::quantile::{max_overflow_tight_branch, max_overflow_tight_le};

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct Stats {
        pub qhat_evals: u64,
        pub qhat_hits: u64,
        pub qhat_predicates: u64,
        pub wcu_evals: u64,
        pub wcu_hits: u64,
    }

    /// Caches `Qhat` and the Chernoff bound for one fixed `(s, delta)`.
    pub struct Evaluator {
        s: u64,
        delta: f64,
        z: f64,
        qhat: HashMap<(u64, u64), u64>,
        wcu: HashMap<(u64, u64), f64>,
        stats: Stats,
    }

    impl Evaluator {
        pub fn new(s: u64, delta: f64) -> Self {
            assert!(s >= 1, "s must be at least 1");
            assert!(delta > 0.0 && delta < 1.0, "delta must lie in (0, 1)");
            Self {
                s,
                delta,
                z: norm_ppf(1.0 - delta),
                qhat: HashMap::new(),
                wcu: HashMap::new(),
                stats: Stats::default(),
            }
        }

        pub fn s(&self) -> u64 {
            self.s
        }

        pub fn delta(&self) -> f64 {
            self.delta
        }

        /// Work counters; the allow covers the non-test build, where nothing reads them.
        #[allow(dead_code)]
        pub fn stats(&self) -> Stats {
            self.stats
        }

        /// `Qhat(m, n)`, taking the bucket count first.
        pub fn q_hat(&mut self, n: u64, m: u64) -> u64 {
            if let Some(&v) = self.qhat.get(&(n, m)) {
                self.stats.qhat_hits += 1;
                return v;
            }
            self.stats.qhat_evals += 1;
            let v = max_overflow_tight_branch(n, m, self.s, self.delta, self.z).0;
            self.qhat.insert((n, m), v);
            v
        }

        pub fn q_hat_le(&mut self, n: u64, m: u64, bound: u64) -> bool {
            if let Some(&v) = self.qhat.get(&(n, m)) {
                self.stats.qhat_hits += 1;
                return v <= bound;
            }
            self.stats.qhat_predicates += 1;
            max_overflow_tight_le(n, m, self.s, self.delta, self.z, bound)
        }

        /// `(m - Qhat(m, n)) / (s n)`: the slots still filled once the overflow cascades away.
        pub fn stored_load(&mut self, n: u64, m: u64) -> f64 {
            let q = self.q_hat(n, m) as f64;
            (m as f64 - q) / (self.s as f64 * n as f64)
        }

        pub fn worst_case_unused(&mut self, n: u64, m: u64) -> f64 {
            if let Some(&v) = self.wcu.get(&(n, m)) {
                self.stats.wcu_hits += 1;
                return v;
            }
            self.stats.wcu_evals += 1;
            let v = worst_case_unused(m as f64, n as f64, self.s, self.delta);
            self.wcu.insert((n, m), v);
            v
        }

        /// Slack-tilt upper bound on a level's overflow, `m - s n + w*(n)`.
        pub fn overflow_bound(&mut self, n: u64, m: u64) -> f64 {
            let w = self.worst_case_unused(n, m);
            max_overflow_bound_from(m as f64, n as f64, self.s, w)
        }

        pub fn stored_load_bound(&mut self, n: u64, m: u64) -> f64 {
            let b = self.overflow_bound(n, m);
            (m as f64 - b) / (self.s as f64 * n as f64)
        }

        /// Largest `n` whose quantile-estimated stored load is at least `a`.
        pub fn buckets_for_load(&mut self, m: u64, a: f64) -> u64 {
            if m == 0 {
                return 1;
            }
            let cap = termination_cap(m, self.s, a);
            let mut lo = 1u64;
            let mut hi = 2u64;
            while hi < cap && self.stored_load(hi, m) >= a {
                lo = hi;
                hi = hi.saturating_mul(2);
            }
            hi = hi.min(cap).max(lo + 1);
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if self.stored_load(mid, m) >= a {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            lo
        }

        pub fn buckets_for_load_bound(&mut self, m: u64, a: f64) -> u64 {
            if m == 0 {
                return 1;
            }
            let cap = termination_cap(m, self.s, a);
            let mut lo = 1u64;
            let mut hi = 2u64;
            while hi < cap && self.stored_load_bound(hi, m) >= a {
                lo = hi;
                hi = hi.saturating_mul(2);
            }
            hi = hi.min(cap).max(lo + 1);
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if self.stored_load_bound(mid, m) >= a {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            lo
        }

        /// Whether the bound admits more buckets; its load is nonincreasing, so one probe decides.
        pub fn bound_beats(&mut self, m: u64, a: f64, n: u64) -> bool {
            self.stored_load_bound(n + 1, m) >= a
        }

        /// Smallest `n <= cap` with `Qhat(m, n) <= max_ov`, or `None` if none reaches it.
        pub fn buckets_for_max_overflow(&mut self, m: u64, max_ov: u64, cap: u64) -> Option<u64> {
            if self.q_hat_le(1, m, max_ov) {
                return Some(1);
            }
            // `n = 1` was just ruled out, so a cap of 1 leaves nothing to search.
            if cap <= 1 {
                return None;
            }
            let mut lo = 1u64;
            let mut hi = 1u64;
            while hi < cap && !self.q_hat_le(hi, m, max_ov) {
                lo = hi;
                hi = hi.saturating_mul(2);
            }
            hi = hi.min(cap).max(lo + 1);
            // Doubling can end at the ceiling, not on a true predicate; bisecting would lie.
            if !self.q_hat_le(hi, m, max_ov) {
                return None;
            }
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if self.q_hat_le(mid, m, max_ov) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            Some(hi)
        }
    }

    /// Termination backstop chosen never to bind: stored load is at most `m / (s n)`.
    fn termination_cap(m: u64, s: u64, a: f64) -> u64 {
        let crossing = m as f64 / (s as f64 * a.max(1e-12));
        ((2.0 * crossing).ceil() as u64).saturating_add(2)
    }

    fn max_overflow_bound_from(m: f64, n: f64, s: u64, wcu: f64) -> f64 {
        (m - (n * s as f64 - wcu)).max(0.0)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const D10: f64 = 1.0 / 1024.0;

        #[test]
        fn buckets_for_load_reproduces_the_published_levels() {
            let mut ev = Evaluator::new(8, D10);
            assert_eq!(ev.buckets_for_load(100_000, 0.99), 7334);
            let a2 = 41915.0 / (8.0 * 6555.0) * 1.1;
            assert_eq!(ev.buckets_for_load(41_915, a2), 4962);
        }

        #[test]
        fn buckets_for_max_overflow_reproduces_the_wrap_level() {
            let mut ev = Evaluator::new(8, D10);
            let n = ev.buckets_for_max_overflow(227, 8, 1_000_000);
            assert_eq!(n, Some(65));
            assert!(ev.q_hat(65, 227) <= 8);
            assert!(ev.q_hat(64, 227) > 8);
        }

        #[test]
        fn the_truncation_search_reaches_past_any_fixed_multiple_of_m() {
            let mut ev = Evaluator::new(2, 1.0 / 1_048_576.0);
            assert_eq!(ev.buckets_for_max_overflow(785, 2, 1_000_000), Some(76_663));
            assert!(ev.q_hat(76_663, 785) <= 2);
            assert!(ev.q_hat(76_662, 785) > 2);
            assert_eq!(ev.buckets_for_max_overflow(785, 2, 50_304), None);
        }

        #[test]
        fn the_truncation_search_declines_the_absurd_cheaply() {
            let mut ev = Evaluator::new(1, D10);
            assert_eq!(ev.buckets_for_max_overflow(100_000, 1, 200_000), None);
            assert!(ev.stats().qhat_evals + ev.stats().qhat_predicates < 100);
        }

        #[test]
        fn memoization_actually_serves_repeats() {
            let mut ev = Evaluator::new(8, D10);
            let _ = ev.buckets_for_load(100_000, 0.99);
            let before = ev.stats();
            let _ = ev.buckets_for_load(100_000, 0.99);
            let after = ev.stats();
            assert_eq!(after.qhat_evals, before.qhat_evals);
            assert!(after.qhat_hits > before.qhat_hits);
        }
    }
}

pub mod quantile {
    //! The approximate overflow quantile `Qhat(m, n)`: a four-way dispatch, each arm `O(s)`.

    #[cfg(test)]
    use super::numerics::binom_sf;
    use super::numerics::{binom_cdf_below_and_sf_above, pois_log_pmf, pois_sf, pois_tails4};

    /// Which arm of the dispatch produced a value.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Branch {
        /// A single bucket: the overflow is deterministic.
        Single,
        Exact,
        /// `m <= s`, so nothing can overflow.
        Trivial,
        /// Every bucket is full with probability at least `1 - delta`.
        Saturated,
        CornishFisher,
        Panjer,
    }

    /// Largest Panjer support explored; the recursion is quadratic in it, so this is a real cap.
    const PANJER_KMAX_LIMIT: usize = 1 << 16;

    /// Ceiling on the exact DP's work in inner-loop operations; a gate on `n` alone is too coarse.
    const EXACT_DP_WORK_LIMIT: f64 = 2e7;

    #[cfg(test)]
    pub fn exact_mean_overflow(m: f64, n: f64, s: u64) -> f64 {
        m * binom_sf(m - 1.0, n, s as i64) - s as f64 * n * binom_sf(m, n, s as i64 + 1)
    }

    /// Poissonized mean overflow: a dispatch gate and Panjer seed, never an answer.
    pub fn overflow_mean_pois(n: f64, m: f64, s: u64) -> f64 {
        let lam = m / n;
        n * (lam * pois_sf(lam, s as i64) - s as f64 * pois_sf(lam, s as i64 + 1))
    }

    /// De-Poissonized `(mean, variance, third cumulant)` of the aggregate overflow.
    pub fn overflow_cumulants(n: f64, m: f64, s: u64) -> (f64, f64, f64) {
        let lam = m / n;
        let (sm2, sm1, s0, sp1) = pois_tails4(lam, s);
        let sf = s as f64;
        let ex = lam;
        let ex2 = lam * lam + lam;
        let ey = lam * s0 - sf * sp1;
        let ey2 = lam * lam * sm1 + lam * (1.0 - 2.0 * sf) * s0 + sf * sf * sp1;
        let ey3 = lam * lam * lam * sm2
            + 3.0 * (1.0 - sf) * lam * lam * sm1
            + (3.0 * sf * sf - 3.0 * sf + 1.0) * lam * s0
            - sf * sf * sf * sp1;
        let eyx = lam * lam * sm1 + lam * (1.0 - sf) * s0;
        let ey2x = lam * lam * lam * sm2
            + (3.0 - 2.0 * sf) * lam * lam * sm1
            + (sf - 1.0).powi(2) * lam * s0;
        let eyx2 = lam * lam * lam * sm2 + (3.0 - sf) * lam * lam * sm1 + (1.0 - sf) * lam * s0;

        let var_y = ey2 - ey * ey;
        let cov_yx = eyx - ex * ey;
        let k3_y = ey3 - 3.0 * ey * ey2 + 2.0 * ey * ey * ey;
        let k_yyx = ey2x - 2.0 * ey * eyx - ex * ey2 + 2.0 * ey * ey * ex;
        let k_yxx = eyx2 - 2.0 * ex * eyx - ey * ex2 + 2.0 * ex * ex * ey;
        let beta = cov_yx / lam;

        let mean = n * ey;
        let var = n * (var_y - cov_yx * cov_yx / lam);
        let k3 =
            n * (k3_y - 3.0 * beta * k_yyx + 3.0 * beta * beta * k_yxx - beta * beta * beta * lam);
        (mean, var, k3)
    }

    /// Cornish-Fisher estimate of the `(1 - delta)`-quantile; `z` must be `Phi^-1(1 - delta)`.
    pub fn max_overflow_cf(n: f64, m: f64, s: u64, z: f64) -> u64 {
        let (mean, var, k3) = overflow_cumulants(n, m, s);
        // Near saturation var and k3 are pure cancellation: below the noise floor, take zero.
        let lam = m / n;
        let noise = 1e-12 * n * lam * lam;
        let (var, k3) = if var <= noise { (0.0, 0.0) } else { (var, k3) };
        let sd = var.max(0.0).sqrt();
        let g1 = if var > 0.0 { k3 / var.powf(1.5) } else { 0.0 };
        let w = z + (g1 / 6.0) * (z * z - 1.0);
        let v = mean + sd * w;
        if !v.is_finite() || v <= 0.0 {
            0
        } else {
            v.ceil() as u64
        }
    }

    /// Compound-Poisson pmf `g[k] = P[O = k]`; forward, so `g[0..=k]` does not depend on `kmax`.
    pub fn overflow_pmf_panjer(lam: f64, nn: f64, s: u64, kmax: usize) -> Vec<f64> {
        let s_us = s as usize;
        let mut g = vec![0.0f64; kmax + 1];
        let p_ev = pois_sf(lam, s as i64 + 1);
        if p_ev <= 0.0 || !p_ev.is_finite() {
            g[0] = 1.0;
            return g;
        }
        let lnpmf = pois_log_pmf(lam, s_us + kmax);
        let ln_pev = p_ev.ln();
        let cap_n = nn * p_ev;

        let mut jf = vec![0.0f64; kmax + 1];
        for (j, slot) in jf.iter_mut().enumerate().skip(1) {
            *slot = j as f64 * (lnpmf[s_us + j] - ln_pev).exp();
        }
        g[0] = (-cap_n).exp();
        for k in 1..=kmax {
            let mut conv = 0.0;
            for j in 1..=k {
                conv += jf[j] * g[k - j];
            }
            g[k] = cap_n / k as f64 * conv;
        }
        g
    }

    /// `P[O > bound]` from the Panjer head alone.
    pub fn panjer_tail_above(lam: f64, nn: f64, s: u64, bound: u64) -> f64 {
        let g = overflow_pmf_panjer(lam, nn, s, bound as usize);
        let head: f64 = g.iter().sum();
        (1.0 - head).max(0.0)
    }

    /// `(1 - p)`-quantile under the compound-Poisson model.
    pub fn max_overflow_panjer(n: f64, m: f64, s: u64, p: f64) -> u64 {
        let mu = overflow_mean_pois(n, m, s).max(0.0);
        let z = if p < 1e-6 { 5.0 } else { 4.0 };
        let mut kmax = (((mu + 3.0 * z * (mu + 1.0).sqrt()).ceil() as i64) + 16).max(8) as usize;
        let mut prev_residual = f64::INFINITY;
        loop {
            let g = overflow_pmf_panjer(m / n, n, s, kmax);
            let tot: f64 = g.iter().sum();
            let residual = 1.0 - tot;
            if residual <= p {
                let mut tail = residual;
                let mut o = kmax;
                while o >= 1 && tail + g[o] <= p {
                    tail += g[o];
                    o -= 1;
                }
                return o as u64;
            }
            // Truncation error falls superexponentially, so short of a halving it is roundoff.
            let stalled = residual >= 0.5 * prev_residual;
            if stalled || kmax >= PANJER_KMAX_LIMIT {
                let scale = if tot > 0.0 { tot } else { 1.0 };
                let mut tail = 0.0;
                let mut o = kmax;
                while o >= 1 && tail + g[o] / scale <= p {
                    tail += g[o] / scale;
                    o -= 1;
                }
                return o as u64;
            }
            prev_residual = residual;
            kmax *= 2;
        }
    }

    pub fn is_exact_excess_cheap(q: f64, n: f64, s: u64) -> bool {
        if s as f64 >= q {
            return true;
        }
        let excess = (q - s as f64).max(0.0) + 1.0;
        n.log2() * ((q + 1.0) * excess).powi(2) * q * (n * q).log2() < 3e9
    }

    /// Exact `(1 - alpha)`-quantile of the aggregate excess over `t`; `None` when too large.
    pub fn exact_max_excess(q: u64, n: u64, t: u64, alpha: f64) -> Option<u64> {
        if n <= 1 {
            return Some(q.saturating_sub(t));
        }
        if t >= q {
            return Some(0);
        }
        let qq = q as usize;
        let e_cap = (q - t) as usize + 1;
        let work = n as f64 * 0.5 * ((qq + 1) as f64).powi(2) * (e_cap as f64 + 3.0);
        if q > 400 || work > EXACT_DP_WORK_LIMIT {
            return None;
        }
        let t_us = t as usize;

        let mut d = vec![0.0f64; (qq + 1) * e_cap];
        let mut dn = vec![0.0f64; (qq + 1) * e_cap];
        d[qq * e_cap] = 1.0;

        let mut pmf = vec![0.0f64; qq + 1];
        for k in (2..=n).rev() {
            dn.iter_mut().for_each(|x| *x = 0.0);
            let kf = k as f64;
            let ln_q_k = (-1.0f64 / kf).ln_1p();
            let inv_km1 = 1.0 / (kf - 1.0);
            for r in 0..=qq {
                let row_start = r * e_cap;
                if d[row_start..row_start + e_cap].iter().all(|&x| x == 0.0) {
                    continue;
                }
                // Bin(r, 1/k) by the ratio recurrence: all terms positive, so no cancellation.
                let rf = r as f64;
                pmf[0] = (rf * ln_q_k).exp();
                for x in 0..r {
                    let xf = x as f64;
                    pmf[x + 1] = pmf[x] * (rf - xf) * inv_km1 / (xf + 1.0);
                }
                for (x, &w) in pmf.iter().enumerate().take(r + 1) {
                    if w == 0.0 {
                        continue;
                    }
                    let exc = x.saturating_sub(t_us);
                    let dst = (r - x) * e_cap;
                    if exc == 0 {
                        for e in 0..e_cap {
                            dn[dst + e] += w * d[row_start + e];
                        }
                    } else if exc < e_cap {
                        for e in 0..e_cap - exc {
                            dn[dst + e + exc] += w * d[row_start + e];
                        }
                    }
                }
            }
            std::mem::swap(&mut d, &mut dn);
        }

        // The last bucket takes whatever remains.
        let mut probs = vec![0.0f64; e_cap];
        for r in 0..=qq {
            let row_start = r * e_cap;
            if d[row_start..row_start + e_cap].iter().all(|&x| x == 0.0) {
                continue;
            }
            let exc = r.saturating_sub(t_us);
            if exc == 0 {
                for e in 0..e_cap {
                    probs[e] += d[row_start + e];
                }
            } else if exc < e_cap {
                for e in 0..e_cap - exc {
                    probs[e + exc] += d[row_start + e];
                }
            }
        }
        let total: f64 = probs.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return None;
        }
        let target = (1.0 - alpha) * total;
        let mut acc = 0.0;
        let mut count = 0u64;
        for &pr in probs.iter() {
            acc += pr;
            if acc < target {
                count += 1;
            } else {
                break;
            }
        }
        Some(count)
    }

    fn deterministic_floor(n: u64, m: u64, s: u64) -> u64 {
        (m as i128 - s as i128 * n as i128).max(0) as u64
    }

    /// `Qhat(m, n)` with its branch, floored at `max(0, m - s n)`; `z` must be `Phi^-1(1 - p)`.
    pub fn max_overflow_tight_branch(n: u64, m: u64, s: u64, p: f64, z: f64) -> (u64, Branch) {
        let (v, b) = max_overflow_tight_raw(n, m, s, p, z);
        (v.max(deterministic_floor(n, m, s)), b)
    }

    fn max_overflow_tight_raw(n: u64, m: u64, s: u64, p: f64, z: f64) -> (u64, Branch) {
        if n <= 1 {
            return (m.saturating_sub(s), Branch::Single);
        }
        let mf = m as f64;
        let nf = n as f64;
        if is_exact_excess_cheap(mf, nf, s) {
            if let Some(v) = exact_max_excess(m, n, s, p) {
                return (v, Branch::Exact);
            }
        }
        if m <= s {
            return (0, Branch::Trivial);
        }
        let (below, above) = binom_cdf_below_and_sf_above(mf, nf, s);
        if nf * below <= p {
            return (deterministic_floor(n, m, s), Branch::Saturated);
        }
        if nf * above >= 10.0 || overflow_mean_pois(nf, mf, s) >= 1000.0 {
            return (max_overflow_cf(nf, mf, s, z), Branch::CornishFisher);
        }
        (max_overflow_panjer(nf, mf, s, p).min(m - s), Branch::Panjer)
    }

    #[cfg(test)]
    pub fn max_overflow_tight(n: u64, m: u64, s: u64, p: f64) -> u64 {
        max_overflow_tight_branch(n, m, s, p, super::numerics::norm_ppf(1.0 - p)).0
    }

    /// Whether `Qhat(m, n) <= bound`, without extracting the quantile.
    pub fn max_overflow_tight_le(n: u64, m: u64, s: u64, p: f64, z: f64, bound: u64) -> bool {
        if deterministic_floor(n, m, s) > bound {
            return false;
        }
        if n <= 1 {
            return m.saturating_sub(s) <= bound;
        }
        let mf = m as f64;
        let nf = n as f64;
        if is_exact_excess_cheap(mf, nf, s) {
            if let Some(v) = exact_max_excess(m, n, s, p) {
                return v <= bound;
            }
        }
        if m <= s {
            return true;
        }
        let (below, above) = binom_cdf_below_and_sf_above(mf, nf, s);
        if nf * below <= p {
            return deterministic_floor(n, m, s) <= bound;
        }
        if nf * above >= 10.0 || overflow_mean_pois(nf, mf, s) >= 1000.0 {
            return max_overflow_cf(nf, mf, s, z) <= bound;
        }
        if m - s <= bound {
            return true;
        }
        panjer_tail_above(mf / nf, nf, s, bound) <= p
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const D10: f64 = 1.0 / 1024.0;

        #[test]
        fn dispatch_fixtures_from_the_notebook() {
            assert_eq!(max_overflow_tight(7334, 100_000, 8, D10), 41914);
            assert_eq!(max_overflow_tight(4962, 41_915, 8, D10), 7012);
            assert_eq!(max_overflow_tight(1401, 7012, 8, D10), 227);
            assert_eq!(max_overflow_tight(65, 227, 8, D10), 8);
            assert_eq!(max_overflow_tight(15, 64, 8, D10), 8);
        }

        #[test]
        fn dispatch_picks_the_documented_branches() {
            let z = super::super::numerics::norm_ppf(1.0 - D10);
            assert_eq!(
                max_overflow_tight_branch(7334, 100_000, 8, D10, z).1,
                Branch::CornishFisher
            );
            assert_eq!(
                max_overflow_tight_branch(65, 227, 8, D10, z).1,
                Branch::Panjer
            );
            let (v, b) = max_overflow_tight_branch(8, 30, 4, D10, z);
            assert_eq!(b, Branch::Exact);
            assert_eq!(v, 11);
            let (v, b) = max_overflow_tight_branch(12, 60, 6, D10, z);
            assert_eq!(b, Branch::Panjer);
            assert_eq!(v, 24);
        }

        #[test]
        fn saturated_corner_for_a_hugely_overcommitted_level() {
            let z = super::super::numerics::norm_ppf(1.0 - D10);
            let (v, b) = max_overflow_tight_branch(2, 1_000_000_000_000, 8, D10, z);
            assert_eq!(b, Branch::Saturated);
            assert_eq!(v, 1_000_000_000_000 - 16);
        }

        #[test]
        fn exact_dp_matches_brute_force_shape() {
            assert_eq!(exact_max_excess(20, 1, 8, D10), Some(12));
            assert_eq!(exact_max_excess(8, 5, 8, D10), Some(0));
        }

        #[test]
        fn mean_overflow_reference_point() {
            let eo = exact_mean_overflow(100.0, 10.0, 12);
            assert!((eo - 4.729_8).abs() < 1e-3, "E[O] = {eo}");
            let pois = overflow_mean_pois(10.0, 100.0, 12);
            assert!((pois - 5.309_2).abs() < 1e-3, "pois = {pois}");
        }

        #[test]
        fn panjer_head_predicate_agrees_with_the_quantile() {
            let z = super::super::numerics::norm_ppf(1.0 - D10);
            for n in [40u64, 55, 64, 65, 66, 80, 120] {
                let q = max_overflow_tight(n, 227, 8, D10);
                assert_eq!(
                    max_overflow_tight_le(n, 227, 8, D10, z, 8),
                    q <= 8,
                    "n = {n}, Qhat = {q}"
                );
            }
        }
    }
}

pub mod numerics {
    //! Log-space binomial and Poisson tails, stepped so the cost never depends on `m`.

    /// `ln(sum_i exp(x_i))`, stable against overflow and underflow.
    pub fn ln_sum_exp(xs: &[f64]) -> f64 {
        let mut mx = f64::NEG_INFINITY;
        for &x in xs {
            if x > mx {
                mx = x;
            }
        }
        if !mx.is_finite() {
            return mx;
        }
        let mut acc = 0.0;
        for &x in xs {
            acc += (x - mx).exp();
        }
        mx + acc.ln()
    }

    /// `ln(1 - exp(x))` for `x <= 0`, split at `ln(1/2)` to keep both sides exact.
    pub fn ln1m_exp(x: f64) -> f64 {
        if x >= 0.0 {
            f64::NEG_INFINITY
        } else if x > -std::f64::consts::LN_2 {
            (-x.exp_m1()).ln()
        } else {
            (-x.exp()).ln_1p()
        }
    }

    /// Inverse standard normal cdf, AS 241 PPND16, accurate for budgets down to `2^-40`.
    #[allow(clippy::excessive_precision)]
    pub fn norm_ppf(p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }
        if p == 0.0 {
            return f64::NEG_INFINITY;
        }
        if p == 1.0 {
            return f64::INFINITY;
        }
        const A: [f64; 8] = [
            3.387_132_872_796_366_608_0,
            1.331_416_678_917_843_774_5e2,
            1.971_590_950_306_551_442_7e3,
            1.373_169_376_550_946_112_5e4,
            4.592_195_393_154_987_145_7e4,
            6.726_577_092_700_870_085_3e4,
            3.343_057_558_358_812_810_5e4,
            2.509_080_928_730_122_672_7e3,
        ];
        const B: [f64; 8] = [
            1.0,
            4.231_333_070_160_091_125_2e1,
            6.871_870_074_920_579_083_0e2,
            5.394_196_021_424_751_107_7e3,
            2.121_379_430_158_659_586_7e4,
            3.930_789_580_009_271_061_0e4,
            2.872_908_573_572_194_267_4e4,
            5.226_495_278_852_854_561_0e3,
        ];
        const C: [f64; 8] = [
            1.423_437_110_749_683_577_34,
            4.630_337_846_156_545_295_90,
            5.769_497_221_460_691_405_50,
            3.647_848_324_763_204_605_04,
            1.270_458_252_452_368_382_58,
            2.417_807_251_774_506_117_70e-1,
            2.272_384_498_926_918_458_33e-2,
            7.745_450_142_783_414_076_40e-4,
        ];
        const D: [f64; 8] = [
            1.0,
            2.053_191_626_637_758_821_87,
            1.676_384_830_183_803_849_40,
            6.897_673_349_851_000_045_50e-1,
            1.481_039_764_274_800_745_90e-1,
            1.519_866_656_361_645_719_66e-2,
            5.475_938_084_995_344_946_00e-4,
            1.050_750_071_644_416_843_24e-9,
        ];
        const E: [f64; 8] = [
            6.657_904_643_501_103_777_20,
            5.463_784_911_164_114_369_90,
            1.784_826_539_917_291_335_80,
            2.965_605_718_285_048_912_30e-1,
            2.653_218_952_657_612_309_30e-2,
            1.242_660_947_388_078_438_60e-3,
            2.711_555_568_743_487_578_15e-5,
            2.010_334_399_292_288_132_65e-7,
        ];
        const F: [f64; 8] = [
            1.0,
            5.998_322_065_558_879_376_90e-1,
            1.369_298_809_227_358_053_10e-1,
            1.487_536_129_085_061_485_25e-2,
            7.868_691_311_456_132_591_00e-4,
            1.846_318_317_510_054_681_80e-5,
            1.421_511_758_316_445_888_70e-7,
            2.044_263_103_389_939_785_64e-15,
        ];

        let q = p - 0.5;
        if q.abs() <= 0.425 {
            let r = 0.180625 - q * q;
            return q * poly(r, &A) / poly(r, &B);
        }
        let tail = if q < 0.0 { p } else { 1.0 - p };
        let r = (-tail.ln()).sqrt();
        let val = if r <= 5.0 {
            let r = r - 1.6;
            poly(r, &C) / poly(r, &D)
        } else {
            let r = r - 5.0;
            poly(r, &E) / poly(r, &F)
        };
        if q < 0.0 {
            -val
        } else {
            val
        }
    }

    fn poly(x: f64, c: &[f64]) -> f64 {
        // c is given low-to-high in the AS 241 tables; evaluate Horner from the top.
        let mut acc = 0.0;
        for &ci in c.iter().rev() {
            acc = acc * x + ci;
        }
        acc
    }

    /// `ln P[X = k]` for `X ~ Bin(m, 1/n)`, `k = 0 ..= kmax`, in `O(kmax)`.
    pub fn binom_log_pmf(m: f64, n: f64, kmax: usize) -> Vec<f64> {
        let mut out = vec![f64::NEG_INFINITY; kmax + 1];
        if n <= 1.0 {
            // Degenerate: all mass at k = m.
            if m <= kmax as f64 && m >= 0.0 && m.fract() == 0.0 {
                out[m as usize] = 0.0;
            }
            return out;
        }
        out[0] = m * (-1.0 / n).ln_1p();
        let ln_nm1 = (n - 1.0).ln();
        for k in 0..kmax {
            let kf = k as f64;
            if kf >= m {
                break; // remaining entries stay at -inf
            }
            out[k + 1] = out[k] + (m - kf).ln() - (kf + 1.0).ln() - ln_nm1;
        }
        out
    }

    /// `P[X >= k0]` walked upward in log space; valid only above the mean, where terms decay.
    fn binom_sf_upward(m: f64, n: f64, k0: usize) -> f64 {
        if (k0 as f64) > m {
            return 0.0;
        }
        let pmf = binom_log_pmf(m, n, k0);
        let base = pmf[k0];
        if !base.is_finite() {
            return 0.0;
        }
        let ln_nm1 = (n - 1.0).ln();
        let mut acc = 1.0f64;
        let mut lp = base;
        let mut k = k0;
        loop {
            let kf = k as f64;
            if kf >= m {
                break;
            }
            lp += (m - kf).ln() - (kf + 1.0).ln() - ln_nm1;
            let term = (lp - base).exp();
            acc += term;
            if term < acc * 1e-18 {
                break;
            }
            k += 1;
        }
        (base + acc.ln()).exp()
    }

    /// `(P[X <= s-1], P[X >= s+1])` for `X ~ Bin(m, 1/n)`, summed cancellation-free.
    pub fn binom_cdf_below_and_sf_above(m: f64, n: f64, s: u64) -> (f64, f64) {
        if n <= 1.0 {
            // X = m deterministically.
            let below = if m < s as f64 { 1.0 } else { 0.0 };
            let above = if m > s as f64 { 1.0 } else { 0.0 };
            return (below, above);
        }
        let s_us = s as usize;
        let pmf = binom_log_pmf(m, n, s_us);
        let mut cdf_below = 0.0;
        for &lp in pmf.iter().take(s_us) {
            cdf_below += lp.exp();
        }
        let head = cdf_below + pmf[s_us].exp();
        let mean = m / n;
        let sf_above = if (s as f64 + 1.0) <= mean {
            (1.0 - head).max(0.0)
        } else {
            binom_sf_upward(m, n, s_us + 1)
        };
        (cdf_below.min(1.0), sf_above.min(1.0))
    }

    /// `P[X >= k]` for `X ~ Bin(m, 1/n)`.
    pub fn binom_sf(m: f64, n: f64, k: i64) -> f64 {
        if k <= 0 {
            return 1.0;
        }
        if n <= 1.0 {
            return if m >= k as f64 { 1.0 } else { 0.0 };
        }
        let mean = m / n;
        if (k as f64) <= mean {
            let pmf = binom_log_pmf(m, n, (k - 1) as usize);
            let lcdf = ln_sum_exp(&pmf);
            ln1m_exp(lcdf.min(0.0)).exp()
        } else {
            binom_sf_upward(m, n, k as usize)
        }
    }

    /// `ln P[X = k]` for `X ~ Pois(lam)`, seeded at `-lam` so nothing underflows past 745.
    pub fn pois_log_pmf(lam: f64, kmax: usize) -> Vec<f64> {
        let mut out = vec![f64::NEG_INFINITY; kmax + 1];
        if lam <= 0.0 {
            out[0] = 0.0;
            return out;
        }
        let ln_lam = lam.ln();
        out[0] = -lam;
        for k in 0..kmax {
            out[k + 1] = out[k] + ln_lam - ((k + 1) as f64).ln();
        }
        out
    }

    fn pois_sf_upward(lam: f64, k0: usize) -> f64 {
        if lam <= 0.0 {
            return 0.0;
        }
        let pmf = pois_log_pmf(lam, k0);
        let base = pmf[k0];
        if !base.is_finite() {
            return 0.0;
        }
        let ln_lam = lam.ln();
        let mut acc = 1.0f64;
        let mut lp = base;
        let mut k = k0;
        loop {
            lp += ln_lam - ((k + 1) as f64).ln();
            let term = (lp - base).exp();
            acc += term;
            if term < acc * 1e-18 || !term.is_finite() {
                break;
            }
            k += 1;
            if k > k0 + 1_000_000 {
                break;
            }
        }
        (base + acc.ln()).exp()
    }

    /// `P[X >= k]` for `X ~ Pois(lam)`; `k <= 0` gives 1.
    pub fn pois_sf(lam: f64, k: i64) -> f64 {
        if k <= 0 {
            return 1.0;
        }
        if lam <= 0.0 {
            return 0.0;
        }
        if (k as f64) <= lam {
            let pmf = pois_log_pmf(lam, (k - 1) as usize);
            let lcdf = ln_sum_exp(&pmf);
            ln1m_exp(lcdf.min(0.0)).exp()
        } else {
            pois_sf_upward(lam, k as usize)
        }
    }

    /// The four Poisson upper tails the cumulant formulas need, in one `O(s)` pass.
    pub fn pois_tails4(lam: f64, s: u64) -> (f64, f64, f64, f64) {
        let s_us = s as usize;
        let pmf = pois_log_pmf(lam, s_us + 1);
        let sp1 = if (s as f64 + 1.0) <= lam {
            let mut head = 0.0;
            for &lp in pmf.iter().take(s_us + 1) {
                head += lp.exp();
            }
            (1.0 - head).max(0.0)
        } else {
            pois_sf_upward(lam, s_us + 1)
        };
        let s0 = (sp1 + pmf[s_us].exp()).min(1.0);
        let sm1 = if s >= 1 {
            (s0 + pmf[s_us - 1].exp()).min(1.0)
        } else {
            1.0
        };
        let sm2 = if s >= 2 {
            (sm1 + pmf[s_us - 2].exp()).min(1.0)
        } else {
            1.0
        };
        (sm2, sm1, s0, sp1.min(1.0))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn close(a: f64, b: f64, rel: f64) -> bool {
            (a - b).abs() <= rel * b.abs().max(1e-300)
        }

        #[test]
        fn norm_ppf_known_points() {
            assert!(close(norm_ppf(0.975), 1.959_963_984_540_054, 1e-12));
            assert!(close(norm_ppf(0.5), 0.0, 1e-12) || norm_ppf(0.5).abs() < 1e-15);
            assert!(close(
                norm_ppf(1.0 - 2f64.powi(-10)),
                3.097_269_078_198_784_6,
                1e-12
            ));
            assert!(close(norm_ppf(1.0 - 1e-15), 7.941_444_487_415_979, 1e-10));
        }

        #[test]
        fn binomial_tails_sum_to_one() {
            let (below, above) = binom_cdf_below_and_sf_above(1000.0, 100.0, 8);
            let pmf = binom_log_pmf(1000.0, 100.0, 8);
            let at_s = pmf[8].exp();
            assert!(close(below + at_s + above, 1.0, 1e-12));
        }

        #[test]
        fn binomial_tail_survives_deep_underflow() {
            let sf = binom_sf(1e9, 1e12, 9);
            assert!(sf > 0.0 && sf < 1e-25, "sf = {sf}");
        }

        #[test]
        fn poisson_pmf_survives_large_lambda() {
            let pmf = pois_log_pmf(2000.0, 2001);
            assert!(pmf[2000].is_finite());
            // The log recurrence drifts by ulps over many steps, hence the loose tolerance.
            assert!(close(pmf[2000].exp(), 0.008_920_248_895_977_536, 1e-10));
        }

        #[test]
        fn poisson_tails_are_ordered() {
            let (sm2, sm1, s0, sp1) = pois_tails4(3.0, 8);
            assert!(sm2 >= sm1 && sm1 >= s0 && s0 >= sp1 && sp1 > 0.0);
            assert!(close(s0, pois_sf(3.0, 8), 1e-12));
            assert!(close(sm2, pois_sf(3.0, 6), 1e-12));
        }

        #[test]
        fn poisson_tails_near_one() {
            let (sm2, sm1, s0, sp1) = pois_tails4(500.0, 8);
            for v in [sm2, sm1, s0, sp1] {
                assert!(v <= 1.0 && v > 0.999_999);
            }
        }
    }
}

pub mod bound {
    //! The slack-tilt Chernoff bound, the paper's `B_4` = `m - s n + w*(n)`.

    use super::numerics::{binom_log_pmf, binom_sf, ln_sum_exp};

    /// Chernoff bound on total unused slots `sum_b (s - X_b)^+` at level `1 - delta`.
    pub fn worst_case_unused(m: f64, n: f64, s: u64, delta: f64) -> f64 {
        if s == 0 {
            return 0.0;
        }
        let s_us = s as usize;
        let ln_pmf = binom_log_pmf(m, n, s_us - 1);
        let pmf: Vec<f64> = ln_pmf.iter().map(|x| x.exp()).collect();
        let ln_sf_s = binom_sf(m, n, s as i64).ln();
        let big_l = (1.0 / delta).ln();
        let sf = s as f64;

        let weights: Vec<f64> = (0..s_us).map(|k| (s_us - k) as f64).collect();

        // Gaussian-optimal starting tilt, from the variance of the truncated slack.
        let mut m1 = 0.0;
        let mut m2 = 0.0;
        for (&p_k, &w) in pmf.iter().zip(&weights) {
            m1 += p_k * w;
            m2 += p_k * w * w;
        }
        let var = (m2 - m1 * m1).max(1e-300);
        let t0 = (2.0 * big_l / (n * var)).sqrt().min(1.5);

        let log_m = |t: f64| -> f64 {
            if t * sf < 500.0 {
                let mut acc = 0.0;
                for (&p_k, &w) in pmf.iter().zip(&weights) {
                    acc += p_k * (t * w).exp_m1();
                }
                acc.ln_1p()
            } else {
                let mut terms = Vec::with_capacity(s_us + 1);
                terms.push(ln_sf_s);
                for (&lp, &w) in ln_pmf.iter().zip(&weights) {
                    terms.push(lp + t * w);
                }
                ln_sum_exp(&terms)
            }
        };
        let h = |t: f64| -> f64 {
            let v = (n * log_m(t) + big_l) / t;
            if v.is_finite() {
                v
            } else {
                f64::INFINITY
            }
        };

        // Log-spaced scan to bracket the minimum, then golden section; T_LO sits below any tilt.
        const T_LO: f64 = 1e-12;
        const T_HI: f64 = 500.0;
        const GRID: usize = 64;
        let ln_lo = T_LO.ln();
        let ln_hi = T_HI.ln();
        let step = (ln_hi - ln_lo) / GRID as f64;

        let mut best_t = t0.clamp(T_LO, T_HI);
        let mut best_v = h(best_t);
        let mut best_i = 0usize;
        for i in 0..=GRID {
            let t = (ln_lo + step * i as f64).exp();
            let v = h(t);
            if v < best_v {
                best_v = v;
                best_t = t;
                best_i = i;
            }
        }
        let mut lo = (ln_lo + step * best_i.saturating_sub(1) as f64)
            .exp()
            .max(T_LO);
        let mut hi = (ln_lo + step * (best_i + 1).min(GRID) as f64).exp();
        if best_t < lo || best_t > hi {
            // The Gaussian start beat every grid point; bracket around it instead.
            lo = (best_t / 4.0).max(T_LO);
            hi = (best_t * 4.0).min(T_HI);
        }
        let phi = 0.5 * (5f64.sqrt() - 1.0);
        let (mut a, mut b) = (lo.ln(), hi.ln());
        let mut c = b - phi * (b - a);
        let mut dpt = a + phi * (b - a);
        let (mut fc, mut fd) = (h(c.exp()), h(dpt.exp()));
        for _ in 0..80 {
            if fc < fd {
                b = dpt;
                dpt = c;
                fd = fc;
                c = b - phi * (b - a);
                fc = h(c.exp());
            } else {
                a = c;
                c = dpt;
                fc = fd;
                dpt = a + phi * (b - a);
                fd = h(dpt.exp());
            }
        }
        let refined = fc.min(fd);
        best_v.min(refined).min(n * sf)
    }

    #[cfg(test)]
    mod tests {
        use super::super::numerics::binom_log_pmf;
        use super::*;

        const D10: f64 = 1.0 / 1024.0;

        #[test]
        fn worst_case_unused_is_a_real_bound() {
            let (m, n, s) = (1e5, 7000.0, 8u64);
            let w = worst_case_unused(m, n, s, D10);
            let lnp = binom_log_pmf(m, n, (s - 1) as usize);
            let mean_unused: f64 = (0..s as usize)
                .map(|k| lnp[k].exp() * (s as usize - k) as f64)
                .sum::<f64>()
                * n;
            assert!(w > mean_unused, "w = {w}, mean = {mean_unused}");
            assert!(w <= n * s as f64);
        }
    }
}

pub use driver::Params;

/// Adapter onto the sizing `Params`; panics, since the table constructors have no error path.
pub(crate) fn bucket_cnts(
    capacity: u64,
    bucket_size: u32,
    alpha: f64,
    max_levels: usize,
) -> Vec<u32> {
    bucket_cnts_with(capacity, bucket_size, alpha, max_levels, None, false)
}

/// The same with every level rounded to a power of two, for `powers-of-2`'s shift addressing.
pub(crate) fn bucket_cnts_powers(
    capacity: u64,
    bucket_size: u32,
    alpha: f64,
    max_levels: usize,
    tolerance: f64,
) -> Vec<u32> {
    bucket_cnts_with(
        capacity,
        bucket_size,
        alpha,
        max_levels,
        Some(tolerance),
        false,
    )
}

/// The same again with the rounding confined to the first level, for `first-power`.
pub(crate) fn bucket_cnts_first_power(
    capacity: u64,
    bucket_size: u32,
    alpha: f64,
    max_levels: usize,
    tolerance: f64,
) -> Vec<u32> {
    bucket_cnts_with(
        capacity,
        bucket_size,
        alpha,
        max_levels,
        Some(tolerance),
        true,
    )
}

fn bucket_cnts_with(
    capacity: u64,
    bucket_size: u32,
    alpha: f64,
    max_levels: usize,
    pow2_tolerance: Option<f64>,
    pow2_first_level_only: bool,
) -> Vec<u32> {
    let params = driver::Params {
        s: bucket_size as u64,
        alpha,
        max_levels,
        pow2_tolerance,
        pow2_first_level_only,
        ..driver::Params::default()
    };
    let sizing = driver::level_sizes(capacity, &params).unwrap_or_else(|e| {
        panic!(
            "paper levels-sizing failed for capacity={capacity}, s={bucket_size}, \
             alpha={alpha}, max_levels={max_levels}: {e}"
        )
    });
    assert!(
        sizing.depth() <= max_levels,
        "paper levels-sizing produced {} levels, more than this table's MAX_LEVELS = {max_levels} \
         (capacity={capacity}, s={bucket_size}, alpha={alpha})",
        sizing.depth()
    );
    sizing
        .sizes
        .into_iter()
        .map(|n| {
            u32::try_from(n).unwrap_or_else(|_| {
                panic!("paper levels-sizing level bucket count {n} overflows u32")
            })
        })
        .collect()
}

#[cfg(test)]
mod adapter_tests {
    use super::*;

    #[test]
    fn bucket_cnts_is_usable_and_covers_q_near_the_target_load() {
        let q = 50_000u64;
        let s = 8u32;
        let alpha = 0.9;
        let max_levels = 8usize;

        let cnts = bucket_cnts(q, s, alpha, max_levels);

        assert!(!cnts.is_empty());
        assert!(cnts.len() <= max_levels, "{} levels", cnts.len());
        assert!(cnts.iter().all(|&n| n >= 1), "cnts = {cnts:?}");
        assert_eq!(*cnts.last().unwrap(), 1, "cnts = {cnts:?}");

        let capacity: u64 = cnts.iter().map(|&n| n as u64).sum::<u64>() * s as u64;
        assert!(capacity >= q, "capacity {capacity} does not cover q {q}");
        let load = q as f64 / capacity as f64;
        assert!(
            load > alpha - 0.05 && load < 1.0,
            "load {load} not near target alpha {alpha}"
        );
    }

    #[test]
    fn bucket_cnts_first_power_rounds_only_the_head() {
        let (q, s, alpha, max_levels) = (50_000u64, 8u32, 0.9, 8usize);
        let cnts = bucket_cnts_first_power(q, s, alpha, max_levels, 0.01);

        assert!(!cnts.is_empty());
        assert!(cnts.len() <= max_levels, "{} levels", cnts.len());
        assert_eq!(*cnts.last().unwrap(), 1, "cnts = {cnts:?}");
        assert!(cnts[0].is_power_of_two(), "cnts = {cnts:?}");
        assert!(
            cnts[1..].iter().any(|n| !n.is_power_of_two()),
            "cnts = {cnts:?}"
        );

        let capacity: u64 = cnts.iter().map(|&n| n as u64).sum::<u64>() * s as u64;
        assert!(capacity >= q, "capacity {capacity} does not cover q {q}");
        let load = q as f64 / capacity as f64;
        assert!(load > alpha - 0.05 && load < 1.0, "load {load}");
    }

    #[test]
    fn bucket_cnts_respects_a_smaller_max_levels() {
        let cnts = bucket_cnts(50_000, 8, 0.9, 8);
        assert!(cnts.len() <= 8);
    }
}
