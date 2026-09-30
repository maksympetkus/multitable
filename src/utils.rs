#[cfg(test)]
use foldhash::fast::RandomState as FoldRandomState;
use std::convert::TryInto;
use std::hash::BuildHasher;
use std::hash::Hash;
#[cfg(any(nightly_compiler, feature = "nightly"))]
use std::simd::cmp::SimdPartialEq;
#[cfg(any(nightly_compiler, feature = "nightly"))]
use std::simd::num::SimdUint;
#[cfg(any(nightly_compiler, feature = "nightly"))]
use std::simd::Select;
#[cfg(any(nightly_compiler, feature = "nightly"))]
use std::simd::Simd;

use std::hash::Hasher;

#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn map_rand_bytes(random_instance: u32, n: u32) -> u32 {
    (((random_instance as u64) * (n as u64)) >> 32) as u32
}

#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
/// Hash of a key; callers split it into a bucket index and a filter byte.
pub fn hash_key_v3<BH: BuildHasher, const KEY_LEN: usize>(
    hash_builder: &BH,
    key: &[u8; KEY_LEN],
) -> u64 {
    let mut hasher = hash_builder.build_hasher();

    // KEY_LEN is const, so this branch is resolved at monomorphization time.
    if KEY_LEN == 8 {
        u64::from_ne_bytes(key[0..8].try_into().unwrap()).hash(&mut hasher);
    } else if KEY_LEN == 16 {
        u128::from_ne_bytes(key[0..16].try_into().unwrap()).hash(&mut hasher);
    } else {
        for i in 0..KEY_LEN / 4 {
            u32::from_ne_bytes(key[i * 4..i * 4 + 4].try_into().unwrap()).hash(&mut hasher);
        }

        if !KEY_LEN.is_multiple_of(4) {
            if KEY_LEN >= 4 {
                u32::from_ne_bytes(key[KEY_LEN - 4..KEY_LEN].try_into().unwrap()).hash(&mut hasher);
            } else {
                key.hash(&mut hasher);
            }
        }
    }

    hasher.finish()
}

// Shims so the call sites match either way: real intrinsics on nightly, safe stand-ins on stable.

#[cfg(any(nightly_compiler, feature = "nightly"))]
#[inline(always)]
pub fn likely(b: bool) -> bool {
    std::intrinsics::likely(b)
}
// `core::hint::likely` is not stable on this rustc, so the stable arms use the cold-call trick.
#[cfg(not(any(nightly_compiler, feature = "nightly")))]
#[inline(always)]
#[cold]
fn cold_path() {}

#[cfg(not(any(nightly_compiler, feature = "nightly")))]
#[inline(always)]
pub fn likely(b: bool) -> bool {
    if !b {
        cold_path();
    }
    b
}

#[cfg(any(nightly_compiler, feature = "nightly"))]
#[inline(always)]
pub fn unlikely(b: bool) -> bool {
    std::intrinsics::unlikely(b)
}
#[cfg(not(any(nightly_compiler, feature = "nightly")))]
#[inline(always)]
pub fn unlikely(b: bool) -> bool {
    if b {
        cold_path();
    }
    b
}

/// Trailing-zero count of a `u32` the caller guarantees is nonzero.
#[cfg(any(nightly_compiler, feature = "nightly"))]
#[inline(always)]
pub unsafe fn cttz_nonzero(x: u32) -> u32 {
    std::intrinsics::cttz_nonzero(x)
}
/// Stable twin of `cttz_nonzero`, with the same nonzero precondition.
#[cfg(not(any(nightly_compiler, feature = "nightly")))]
#[inline(always)]
pub unsafe fn cttz_nonzero(x: u32) -> u32 {
    debug_assert!(x != 0);
    x.trailing_zeros()
}

/// Read-prefetch hint; no stable intrinsic exists, so `prefetch_l1` covers stable per arch.
#[cfg(any(nightly_compiler, feature = "nightly"))]
#[inline(always)]
pub fn prefetch_read_data<T, const LOCALITY: i32>(data: *const T) {
    std::intrinsics::prefetch_read_data::<T, LOCALITY>(data);
}
#[cfg(not(any(nightly_compiler, feature = "nightly")))]
#[inline(always)]
pub fn prefetch_read_data<T, const LOCALITY: i32>(_data: *const T) {}

/// Read-prefetch into L1, split by target arch rather than toolchain so stable prefetches too.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn prefetch_l1<T>(data: *const T) {
    // The options keep this from acting as a barrier, so surrounding code still schedules freely.
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{0}]",
            in(reg) data,
            options(nostack, preserves_flags, readonly)
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn prefetch_l1<T>(data: *const T) {
    // Prefetch takes any address and dereferences nothing.
    unsafe { std::arch::x86_64::_mm_prefetch(data as *const i8, std::arch::x86_64::_MM_HINT_T0) };
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
pub fn prefetch_l1<T>(_data: *const T) {}

#[cfg(any(nightly_compiler, feature = "nightly"))]
#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn find_simd_old(data: &[u32], target: u32, len: usize) -> Option<usize> {
    const ARITY: usize = 8;
    type SpecSimd = Simd<u32, ARITY>;

    let target_splat = SpecSimd::splat(target);

    for chunk_idx in 0..data.len().div_ceil(ARITY) {
        let vector = SpecSimd::from_slice(&data[chunk_idx * ARITY..][..ARITY]);
        let mask = vector.simd_eq(target_splat);

        if let Some(value) = mask.first_set() {
            return if value + chunk_idx * ARITY < len {
                Some(chunk_idx * ARITY + value)
            } else {
                None
            };
        }
    }

    None
}

/// Stable twin of `find_simd_old`: plain scan over `0..len`.
#[cfg(not(any(nightly_compiler, feature = "nightly")))]
#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn find_simd_old(data: &[u32], target: u32, len: usize) -> Option<usize> {
    for (i, &x) in data.iter().enumerate() {
        if x == target {
            return if i < len { Some(i) } else { None };
        }
    }
    None
}

#[cfg(any(nightly_compiler, feature = "nightly"))]
#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn find_simd<const KEY_LEN: usize, const BUCKET_LEN: usize>(
    data: &[[u8; KEY_LEN]; BUCKET_LEN],
    target: &[u8; KEY_LEN],
    len: usize,
) -> Option<usize> {
    // Instantiated only with KEY_LEN == size_of::<$lane>(), so every `try_into` is an exact fit.
    macro_rules! scan_lanes {
        ($lane:ty, $arity:expr) => {{
            let target_splat = Simd::<$lane, $arity>::splat(<$lane>::from_ne_bytes(
                target[..].try_into().expect("Type matches KEY_LEN"),
            ));

            let interpreted_data: [$lane; BUCKET_LEN] = std::array::from_fn(|i| {
                <$lane>::from_ne_bytes(data[i][..].try_into().expect("Same ARITY KEY_LEN"))
            });

            for chunk_idx in 0..BUCKET_LEN / $arity {
                let vector = Simd::<$lane, $arity>::from_slice(
                    &interpreted_data[chunk_idx * $arity..chunk_idx * $arity + $arity],
                );
                if let Some(value) = vector.simd_eq(target_splat).first_set() {
                    let index = chunk_idx * $arity + value;
                    return if index < len { Some(index) } else { None };
                }
            }
            None
        }};
    }

    // Lane width is BUCKET_LEN itself; clamping it would need `generic_const_exprs` for no gain.
    match KEY_LEN {
        1 => scan_lanes!(u8, { BUCKET_LEN }),
        2 => scan_lanes!(u16, { BUCKET_LEN }),
        4 => scan_lanes!(u32, { BUCKET_LEN }),
        8 => scan_lanes!(u64, { BUCKET_LEN }),
        // KEY_LEN == 16 falls through as well, since `u128` is not a portable-SIMD lane type.
        _ => scan_scalar(data, target, 0, len),
    }
}

/// Stable twin of `find_simd`: reuses `scan_scalar` directly.
#[cfg(not(any(nightly_compiler, feature = "nightly")))]
#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn find_simd<const KEY_LEN: usize, const BUCKET_LEN: usize>(
    data: &[[u8; KEY_LEN]; BUCKET_LEN],
    target: &[u8; KEY_LEN],
    len: usize,
) -> Option<usize> {
    scan_scalar(&data[..], target, 0, len)
}

/// `find_simd`'s scalar arm, over `data[from..]`, with the same `len` contract.
#[inline(always)]
fn scan_scalar<const KEY_LEN: usize>(
    data: &[[u8; KEY_LEN]],
    target: &[u8; KEY_LEN],
    from: usize,
    len: usize,
) -> Option<usize> {
    #[cfg(target_arch = "aarch64")]
    if KEY_LEN == 4 {
        // Sound: KEY_LEN == 4 just confirmed, so [u8; KEY_LEN] and [u8; 4] agree in layout.
        let data4: &[[u8; 4]] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const [u8; 4], data.len()) };
        let target4: &[u8; 4] = unsafe { &*(target.as_ptr() as *const [u8; 4]) };
        return unsafe { scan_neon_u32(data4, target4, from, len) };
    }

    #[cfg(target_arch = "aarch64")]
    if KEY_LEN == 8 {
        // Sound: KEY_LEN == 8 just confirmed, so [u8; KEY_LEN] and [u8; 8] agree in layout.
        let data8: &[[u8; 8]] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const [u8; 8], data.len()) };
        let target8: &[u8; 8] = unsafe { &*(target.as_ptr() as *const [u8; 8]) };
        return unsafe { scan_neon_u64(data8, target8, from, len) };
    }

    #[cfg(target_arch = "aarch64")]
    if KEY_LEN == 16 {
        // Sound: KEY_LEN == 16 just confirmed, so [u8; KEY_LEN] and [u8; 16] agree in layout.
        let data16: &[[u8; 16]] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const [u8; 16], data.len()) };
        let target16: &[u8; 16] = unsafe { &*(target.as_ptr() as *const [u8; 16]) };
        return unsafe { scan_neon_u128(data16, target16, from, len) };
    }

    #[cfg(target_arch = "x86_64")]
    if KEY_LEN == 4 {
        // Sound: KEY_LEN == 4 just confirmed, so [u8; KEY_LEN] and [u8; 4] agree in layout.
        let data4: &[[u8; 4]] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const [u8; 4], data.len()) };
        let target4: &[u8; 4] = unsafe { &*(target.as_ptr() as *const [u8; 4]) };
        return unsafe { scan_sse2_u32(data4, target4, from, len) };
    }

    #[cfg(target_arch = "x86_64")]
    if KEY_LEN == 8 {
        // Sound: KEY_LEN == 8 just confirmed, so [u8; KEY_LEN] and [u8; 8] agree in layout.
        let data8: &[[u8; 8]] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const [u8; 8], data.len()) };
        let target8: &[u8; 8] = unsafe { &*(target.as_ptr() as *const [u8; 8]) };
        return unsafe { scan_sse2_u64(data8, target8, from, len) };
    }

    #[cfg(target_arch = "x86_64")]
    if KEY_LEN == 16 {
        // Sound: KEY_LEN == 16 just confirmed, so [u8; KEY_LEN] and [u8; 16] agree in layout.
        let data16: &[[u8; 16]] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const [u8; 16], data.len()) };
        let target16: &[u8; 16] = unsafe { &*(target.as_ptr() as *const [u8; 16]) };
        return unsafe { scan_sse2_u128(data16, target16, from, len) };
    }

    for (i, k) in data[from..].iter().enumerate() {
        if fast_eq(k, target) {
            let index = from + i;
            return if index < len { Some(index) } else { None };
        }
    }
    None
}

/// NEON twin of the KEY_LEN == 4 arm: a min-index reduction matching nightly's `first_set`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn scan_neon_u32(
    data: &[[u8; 4]],
    target: &[u8; 4],
    from: usize,
    len: usize,
) -> Option<usize> {
    use core::arch::aarch64::*;
    let slice = &data[from..];
    let tv = vdupq_n_u32(u32::from_ne_bytes(*target));
    let iota = [0u32, 1, 2, 3];
    let sentinel = vdupq_n_u32(u32::MAX);
    let mut acc = sentinel;
    let mut base = 0usize;
    while base + 4 <= slice.len() {
        let idxv = vaddq_u32(vdupq_n_u32(base as u32), vld1q_u32(iota.as_ptr()));
        let v = vld1q_u32(slice[base].as_ptr() as *const u32);
        let eq = vceqq_u32(v, tv);
        acc = vminq_u32(acc, vbslq_u32(eq, idxv, sentinel));
        base += 4;
    }
    // The lowest index always wins, so a min that fails the `len` test rules out every match.
    let min_idx = vminvq_u32(acc);
    if min_idx != u32::MAX {
        let idx = from + min_idx as usize;
        return if idx < len { Some(idx) } else { None };
    }
    while base < slice.len() {
        if fast_eq(&slice[base], target) {
            let idx = from + base;
            return if idx < len { Some(idx) } else { None };
        }
        base += 1;
    }
    None
}

/// NEON twin of the KEY_LEN == 8 arm: two keys per compare, folded into four u32 lanes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn scan_neon_u64(
    data: &[[u8; 8]],
    target: &[u8; 8],
    from: usize,
    len: usize,
) -> Option<usize> {
    use core::arch::aarch64::*;
    let slice = &data[from..];
    let tv = vdupq_n_u64(u64::from_ne_bytes(*target));
    let iota = [0u32, 1, 2, 3];
    let sentinel = vdupq_n_u32(u32::MAX);
    let mut acc = sentinel;
    let mut base = 0usize;
    while base + 4 <= slice.len() {
        let idxv = vaddq_u32(vdupq_n_u32(base as u32), vld1q_u32(iota.as_ptr()));
        let lo = vceqq_u64(vld1q_u64(slice[base].as_ptr() as *const u64), tv);
        let hi = vceqq_u64(vld1q_u64(slice[base + 2].as_ptr() as *const u64), tv);
        // A key's verdict fills both u32 halves of its lane, so the pairwise min folds it to one.
        let eq = vpminq_u32(vreinterpretq_u32_u64(lo), vreinterpretq_u32_u64(hi));
        acc = vminq_u32(acc, vbslq_u32(eq, idxv, sentinel));
        base += 4;
    }
    let min_idx = vminvq_u32(acc);
    if min_idx != u32::MAX {
        let idx = from + min_idx as usize;
        return if idx < len { Some(idx) } else { None };
    }
    while base < slice.len() {
        if fast_eq(&slice[base], target) {
            let idx = from + base;
            return if idx < len { Some(idx) } else { None };
        }
        base += 1;
    }
    None
}

/// NEON twin of the KEY_LEN == 16 arm: one key per compare, folded into four u32 lanes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn scan_neon_u128(
    data: &[[u8; 16]],
    target: &[u8; 16],
    from: usize,
    len: usize,
) -> Option<usize> {
    use core::arch::aarch64::*;
    let slice = &data[from..];
    let tv = vld1q_u32(target.as_ptr() as *const u32);
    let iota = [0u32, 1, 2, 3];
    let sentinel = vdupq_n_u32(u32::MAX);
    let mut acc = sentinel;
    let mut base = 0usize;
    while base + 4 <= slice.len() {
        let idxv = vaddq_u32(vdupq_n_u32(base as u32), vld1q_u32(iota.as_ptr()));
        let key = |i: usize| vceqq_u32(vld1q_u32(slice[base + i].as_ptr() as *const u32), tv);
        // Two pairwise rounds collapse each key's verdicts, so a lane means all 16 bytes matched.
        let pair = vpminq_u32(vpminq_u32(key(0), key(1)), vpminq_u32(key(2), key(3)));
        acc = vminq_u32(acc, vbslq_u32(pair, idxv, sentinel));
        base += 4;
    }
    let min_idx = vminvq_u32(acc);
    if min_idx != u32::MAX {
        let idx = from + min_idx as usize;
        return if idx < len { Some(idx) } else { None };
    }
    while base < slice.len() {
        if fast_eq(&slice[base], target) {
            let idx = from + base;
            return if idx < len { Some(idx) } else { None };
        }
        base += 1;
    }
    None
}

/// SSE2 twin of the KEY_LEN == 4 arm, on both toolchains: four keys per compare, scalar tail.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn scan_sse2_u32(
    data: &[[u8; 4]],
    target: &[u8; 4],
    from: usize,
    len: usize,
) -> Option<usize> {
    use core::arch::x86_64::*;
    let slice = &data[from..];
    let tv = _mm_set1_epi32(i32::from_ne_bytes(*target));
    let mut base = 0usize;
    while base + 4 <= slice.len() {
        let v = _mm_loadu_si128(slice[base].as_ptr() as *const __m128i);
        let mask = _mm_movemask_epi8(_mm_cmpeq_epi32(v, tv)) as u32;
        // Four mask bits per key, so the lowest set bit names key `tz / 4`.
        if mask != 0 {
            let idx = from + base + (mask.trailing_zeros() / 4) as usize;
            return if idx < len { Some(idx) } else { None };
        }
        base += 4;
    }
    while base < slice.len() {
        if fast_eq(&slice[base], target) {
            let idx = from + base;
            return if idx < len { Some(idx) } else { None };
        }
        base += 1;
    }
    None
}

/// SSE2 twin of the KEY_LEN == 8 arm, on both toolchains: two keys per compare, eight bits each.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn scan_sse2_u64(
    data: &[[u8; 8]],
    target: &[u8; 8],
    from: usize,
    len: usize,
) -> Option<usize> {
    use core::arch::x86_64::*;
    let slice = &data[from..];
    let tv = _mm_set1_epi64x(i64::from_ne_bytes(*target));
    let mut base = 0usize;
    while base + 2 <= slice.len() {
        let v = _mm_loadu_si128(slice[base].as_ptr() as *const __m128i);
        #[cfg(target_feature = "sse4.1")]
        let eq = _mm_cmpeq_epi64(v, tv);
        // `pcmpeqq` needs SSE4.1, so compare 32-bit halves and `and` each lane with its sibling.
        #[cfg(not(target_feature = "sse4.1"))]
        let eq = {
            let halves = _mm_cmpeq_epi32(v, tv);
            _mm_and_si128(halves, _mm_shuffle_epi32::<0b10_11_00_01>(halves))
        };
        let mask = _mm_movemask_epi8(eq) as u32;
        if mask != 0 {
            let idx = from + base + (mask.trailing_zeros() / 8) as usize;
            return if idx < len { Some(idx) } else { None };
        }
        base += 2;
    }
    while base < slice.len() {
        if fast_eq(&slice[base], target) {
            let idx = from + base;
            return if idx < len { Some(idx) } else { None };
        }
        base += 1;
    }
    None
}

/// SSE2 twin of the KEY_LEN == 16 arm, used on nightly too: one key fills a vector, so no tail.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn scan_sse2_u128(
    data: &[[u8; 16]],
    target: &[u8; 16],
    from: usize,
    len: usize,
) -> Option<usize> {
    use core::arch::x86_64::*;
    let tv = _mm_loadu_si128(target.as_ptr() as *const __m128i);
    for (i, key) in data[from..].iter().enumerate() {
        let v = _mm_loadu_si128(key.as_ptr() as *const __m128i);
        // Every one of the sixteen bytes has to agree, hence all sixteen mask bits.
        if _mm_movemask_epi8(_mm_cmpeq_epi8(v, tv)) as u32 == 0xFFFF {
            let idx = from + i;
            return if idx < len { Some(idx) } else { None };
        }
    }
    None
}

#[cfg(any(nightly_compiler, feature = "nightly"))]
#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn find_all_simd_u8<const ARITY: usize>(data: &[u8], target: u8, len: usize) -> u64 {
    debug_assert!(data.len() <= 64 && len <= data.len() && ARITY <= 32);

    let mut bitmask = 0u64;

    let target_splat = Simd::<u8, ARITY>::splat(target);
    let (chunks, remainder) = data.as_chunks::<ARITY>();

    for (chunk_idx, chunk) in chunks.iter().take((len - 1) / ARITY + 1).enumerate() {
        let vector = Simd::<u8, ARITY>::from_array(*chunk);
        let mask = vector.simd_eq(target_splat);

        bitmask |= mask.to_bitmask() << (chunk_idx * ARITY);
    }
    for (i, &x) in remainder.iter().enumerate() {
        if x == target {
            bitmask |= 1 << (data.len() - remainder.len() + i);
        }
    }

    bitmask & (u64::MAX >> (64 - len))
}

/// Stable twin of `find_all_simd_u8`: bit `i` set iff `data[i] == target`, for `i < len`.
#[cfg(not(any(nightly_compiler, feature = "nightly")))]
#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn find_all_simd_u8<const ARITY: usize>(data: &[u8], target: u8, len: usize) -> u64 {
    debug_assert!(data.len() <= 64 && len <= data.len() && ARITY <= 32);

    let mut bitmask = 0u64;
    for (i, &b) in data.iter().enumerate() {
        if b == target {
            bitmask |= 1 << i;
        }
    }
    bitmask & (u64::MAX >> (64 - len))
}

#[cfg(any(nightly_compiler, feature = "nightly"))]
#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn find_all_simd_u8_untruncated<const ARITY: usize>(data: &[u8], target: u8) -> u32 {
    let mut bitmask = 0u32;

    let target_splat = Simd::<u8, ARITY>::splat(target);
    let (chunks, remainder) = data.as_chunks::<ARITY>();

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let vector = Simd::<u8, ARITY>::from_array(*chunk);
        let mask = vector.simd_eq(target_splat);

        bitmask |= (mask.to_bitmask() as u32) << (chunk_idx * ARITY);
    }
    for (i, &x) in remainder.iter().enumerate() {
        if x == target {
            bitmask |= 1 << (data.len() - remainder.len() + i);
        }
    }

    bitmask
}

/// Stable twin of `find_all_simd_u8_untruncated`: bit `i` set iff `data[i] == target`.
#[cfg(not(any(nightly_compiler, feature = "nightly")))]
#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn find_all_simd_u8_untruncated<const ARITY: usize>(data: &[u8], target: u8) -> u32 {
    let mut bitmask = 0u32;
    for (i, &b) in data.iter().enumerate() {
        if b == target {
            bitmask |= 1 << i;
        }
    }
    bitmask
}

/// `EARLY_OUT` is inert on nightly and only steers the stable twin.
#[cfg(any(nightly_compiler, feature = "nightly"))]
#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn find_all_simd_u8_overread<const LEN: usize, const ARITY: usize, const EARLY_OUT: bool>(
    data: &[u8; ARITY],
    target: u8,
) -> u32 {
    let vector = Simd::<u8, ARITY>::from_array(*data);
    let mask = vector.simd_eq(Simd::splat(target));
    mask.to_bitmask() as u32
}

/// Lowest lane of `data` equal to `target`, or the `u8::MAX` sentinel when no lane is.
#[cfg(any(nightly_compiler, feature = "nightly"))]
#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn find_first_simd_u8_overread<const ARITY: usize>(data: &[u8; ARITY], target: u8) -> u32 {
    // x86 has no cheap horizontal min, so `pmovmskb` plus `tzcnt` answers the same question.
    #[cfg(target_arch = "x86_64")]
    if ARITY == 8 {
        // Sound: ARITY == 8 just confirmed, so [u8; ARITY] and [u8; 8] agree in layout.
        let data8: &[u8; 8] = unsafe { &*(data.as_ptr() as *const [u8; 8]) };
        return first_from_mask_x86(unsafe { eq_mask_x86_u8x8(data8, target) });
    } else if ARITY == 16 {
        // Sound: ARITY == 16 just confirmed, so [u8; ARITY] and [u8; 16] agree in layout.
        let data16: &[u8; 16] = unsafe { &*(data.as_ptr() as *const [u8; 16]) };
        return first_from_mask_x86(unsafe { eq_mask_x86_u8x16(data16, target) });
    } else if ARITY == 32 {
        // Sound: ARITY == 32 just confirmed, so [u8; ARITY] and [u8; 32] agree in layout.
        let data32: &[u8; 32] = unsafe { &*(data.as_ptr() as *const [u8; 32]) };
        return first_from_mask_x86(unsafe { eq_mask_x86_u8x32(data32, target) });
    }

    let eq = Simd::<u8, ARITY>::from_array(*data).simd_eq(Simd::splat(target));
    let lanes = Simd::<u8, ARITY>::from_array(std::array::from_fn(|i| i as u8));
    eq.select(lanes, Simd::splat(u8::MAX)).reduce_min() as u32
}

/// Stable twin of `find_first_simd_u8_overread`: `vorn` plus one `uminv` gives lane or sentinel.
#[cfg(not(any(nightly_compiler, feature = "nightly")))]
#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn find_first_simd_u8_overread<const ARITY: usize>(data: &[u8; ARITY], target: u8) -> u32 {
    #[cfg(target_arch = "aarch64")]
    if ARITY == 8 {
        // Sound: ARITY == 8 just confirmed, so [u8; ARITY] and [u8; 8] agree in layout.
        let data8: &[u8; 8] = unsafe { &*(data.as_ptr() as *const [u8; 8]) };
        return unsafe { find_first_neon_u8x8(data8, target) };
    } else if ARITY == 16 {
        // Sound: ARITY == 16 just confirmed, so [u8; ARITY] and [u8; 16] agree in layout.
        let data16: &[u8; 16] = unsafe { &*(data.as_ptr() as *const [u8; 16]) };
        return unsafe { find_first_neon_u8x16(data16, target) };
    } else if ARITY == 32 {
        // Sound: ARITY == 32 just confirmed, so [u8; ARITY] and [u8; 32] agree in layout.
        let data32: &[u8; 32] = unsafe { &*(data.as_ptr() as *const [u8; 32]) };
        return unsafe { find_first_neon_u8x32(data32, target) };
    }

    #[cfg(target_arch = "x86_64")]
    if ARITY == 8 {
        // Sound: ARITY == 8 just confirmed, so [u8; ARITY] and [u8; 8] agree in layout.
        let data8: &[u8; 8] = unsafe { &*(data.as_ptr() as *const [u8; 8]) };
        return first_from_mask_x86(unsafe { eq_mask_x86_u8x8(data8, target) });
    } else if ARITY == 16 {
        // Sound: ARITY == 16 just confirmed, so [u8; ARITY] and [u8; 16] agree in layout.
        let data16: &[u8; 16] = unsafe { &*(data.as_ptr() as *const [u8; 16]) };
        return first_from_mask_x86(unsafe { eq_mask_x86_u8x16(data16, target) });
    } else if ARITY == 32 {
        // Sound: ARITY == 32 just confirmed, so [u8; ARITY] and [u8; 32] agree in layout.
        let data32: &[u8; 32] = unsafe { &*(data.as_ptr() as *const [u8; 32]) };
        return first_from_mask_x86(unsafe { eq_mask_x86_u8x32(data32, target) });
    }

    data.iter()
        .position(|&byte| byte == target)
        .map_or(u8::MAX as u32, |i| i as u32)
}

/// ARITY == 8 min-index scan.
#[cfg(all(
    target_arch = "aarch64",
    not(any(nightly_compiler, feature = "nightly"))
))]
#[inline(always)]
unsafe fn find_first_neon_u8x8(data: &[u8; 8], target: u8) -> u32 {
    use core::arch::aarch64::*;
    let eq = vceq_u8(vld1_u8(data.as_ptr()), vdup_n_u8(target));
    let iota = [0u8, 1, 2, 3, 4, 5, 6, 7];
    vminv_u8(vorn_u8(vld1_u8(iota.as_ptr()), eq)) as u32
}

/// ARITY == 16 min-index scan.
#[cfg(all(
    target_arch = "aarch64",
    not(any(nightly_compiler, feature = "nightly"))
))]
#[inline(always)]
unsafe fn find_first_neon_u8x16(data: &[u8; 16], target: u8) -> u32 {
    use core::arch::aarch64::*;
    let eq = vceqq_u8(vld1q_u8(data.as_ptr()), vdupq_n_u8(target));
    let iota = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    vminvq_u8(vornq_u8(vld1q_u8(iota.as_ptr()), eq)) as u32
}

/// ARITY == 32 min-index scan: both halves fold into one 16-lane min before the `uminv`.
#[cfg(all(
    target_arch = "aarch64",
    not(any(nightly_compiler, feature = "nightly"))
))]
#[inline(always)]
unsafe fn find_first_neon_u8x32(data: &[u8; 32], target: u8) -> u32 {
    use core::arch::aarch64::*;
    let splat = vdupq_n_u8(target);
    let iota_lo = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let iota_hi = [
        16u8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    ];
    let lo = vceqq_u8(vld1q_u8(data.as_ptr()), splat);
    let hi = vceqq_u8(vld1q_u8(data.as_ptr().add(16)), splat);
    let lo = vornq_u8(vld1q_u8(iota_lo.as_ptr()), lo);
    let hi = vornq_u8(vld1q_u8(iota_hi.as_ptr()), hi);
    vminvq_u8(vminq_u8(lo, hi)) as u32
}

/// Stable twin of `find_all_simd_u8_overread`: bit `i` set iff `data[i] == target`.
/// `EARLY_OUT` callers only need zero versus nonzero, so an arm may skip building the exact mask.
#[cfg(not(any(nightly_compiler, feature = "nightly")))]
#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn find_all_simd_u8_overread<const LEN: usize, const ARITY: usize, const EARLY_OUT: bool>(
    data: &[u8; ARITY],
    target: u8,
) -> u32 {
    #[cfg(target_arch = "aarch64")]
    if ARITY == 8 {
        // Sound: ARITY == 8 just confirmed, so [u8; ARITY] and [u8; 8] agree in layout.
        let data8: &[u8; 8] = unsafe { &*(data.as_ptr() as *const [u8; 8]) };
        return unsafe { find_all_neon_u8x8::<EARLY_OUT>(data8, target) };
    } else if ARITY == 16 {
        // Sound: ARITY == 16 just confirmed, so [u8; ARITY] and [u8; 16] agree in layout.
        let data16: &[u8; 16] = unsafe { &*(data.as_ptr() as *const [u8; 16]) };
        return unsafe { find_all_neon_u8x16(data16, target) };
    } else if ARITY == 32 {
        // Sound: ARITY == 32 just confirmed, so [u8; ARITY] and [u8; 32] agree in layout.
        let data32: &[u8; 32] = unsafe { &*(data.as_ptr() as *const [u8; 32]) };
        return unsafe { find_all_neon_u8x32(data32, target) };
    }

    // `EARLY_OUT` is inert on x86: the mask is the compare's result, with no reduction to skip.
    #[cfg(target_arch = "x86_64")]
    if ARITY == 8 {
        // Sound: ARITY == 8 just confirmed, so [u8; ARITY] and [u8; 8] agree in layout.
        let data8: &[u8; 8] = unsafe { &*(data.as_ptr() as *const [u8; 8]) };
        return unsafe { eq_mask_x86_u8x8(data8, target) };
    } else if ARITY == 16 {
        // Sound: ARITY == 16 just confirmed, so [u8; ARITY] and [u8; 16] agree in layout.
        let data16: &[u8; 16] = unsafe { &*(data.as_ptr() as *const [u8; 16]) };
        return unsafe { eq_mask_x86_u8x16(data16, target) };
    } else if ARITY == 32 {
        // Sound: ARITY == 32 just confirmed, so [u8; ARITY] and [u8; 32] agree in layout.
        let data32: &[u8; 32] = unsafe { &*(data.as_ptr() as *const [u8; 32]) };
        return unsafe { eq_mask_x86_u8x32(data32, target) };
    }

    let mut mask = 0u32;
    for (i, &b) in data.iter().enumerate() {
        if b == target {
            mask |= 1 << i;
        }
    }
    mask
}

/// ARITY == 8 filter scan: a weight-and-add bitmask, skipped only for `EARLY_OUT` callers.
#[cfg(all(
    target_arch = "aarch64",
    not(any(nightly_compiler, feature = "nightly"))
))]
#[inline(always)]
unsafe fn find_all_neon_u8x8<const EARLY_OUT: bool>(data: &[u8; 8], target: u8) -> u32 {
    use core::arch::aarch64::*;
    let eq = vceq_u8(vld1_u8(data.as_ptr()), vdup_n_u8(target));
    if EARLY_OUT && vmaxv_u8(eq) == 0 {
        return 0;
    }
    let bit_idx = [1u8, 2, 4, 8, 16, 32, 64, 128];
    let mask = vaddv_u8(vand_u8(eq, vld1_u8(bit_idx.as_ptr()))) as u32;
    if EARLY_OUT {
        assert_matched(mask)
    } else {
        mask
    }
}

/// ARITY == 16 filter scan, any-lane test first because most scans match nothing at all.
#[cfg(all(
    target_arch = "aarch64",
    not(any(nightly_compiler, feature = "nightly"))
))]
#[inline(always)]
unsafe fn find_all_neon_u8x16(data: &[u8; 16], target: u8) -> u32 {
    use core::arch::aarch64::*;
    let eq = vceqq_u8(vld1q_u8(data.as_ptr()), vdupq_n_u8(target));
    if vmaxvq_u8(eq) == 0 {
        return 0;
    }
    assert_matched(mask_from_eq_u8x16(eq))
}

/// 16-lane compare to a 16-bit mask: zipping the weighted halves lets one `addv` do both bytes.
#[cfg(all(
    target_arch = "aarch64",
    not(any(nightly_compiler, feature = "nightly"))
))]
#[inline(always)]
unsafe fn mask_from_eq_u8x16(eq: core::arch::aarch64::uint8x16_t) -> u32 {
    use core::arch::aarch64::*;
    let bit_idx = [1u8, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let bits = vandq_u8(eq, vld1q_u8(bit_idx.as_ptr()));
    let paired = vzip1q_u8(bits, vextq_u8::<8>(bits, bits));
    vaddvq_u16(vreinterpretq_u16_u8(paired)) as u32
}

/// Sound only where an any-lane test has already ruled a zero mask out; it tells LLVM so.
#[cfg(all(
    target_arch = "aarch64",
    not(any(nightly_compiler, feature = "nightly"))
))]
#[inline(always)]
unsafe fn assert_matched(mask: u32) -> u32 {
    if mask == 0 {
        core::hint::unreachable_unchecked();
    }
    mask
}

/// ARITY == 32 filter scan: two 16-lane compares under one any-lane test over their `or`.
#[cfg(all(
    target_arch = "aarch64",
    not(any(nightly_compiler, feature = "nightly"))
))]
#[inline(always)]
unsafe fn find_all_neon_u8x32(data: &[u8; 32], target: u8) -> u32 {
    use core::arch::aarch64::*;
    let splat = vdupq_n_u8(target);
    let lo = vceqq_u8(vld1q_u8(data.as_ptr()), splat);
    let hi = vceqq_u8(vld1q_u8(data.as_ptr().add(16)), splat);
    if vmaxvq_u8(vorrq_u8(lo, hi)) == 0 {
        return 0;
    }
    assert_matched(mask_from_eq_u8x16(lo) | (mask_from_eq_u8x16(hi) << 16))
}

/// ARITY == 8 filter scan, x86: the load zero-fills lanes 8..16, so the mask has to drop them.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn eq_mask_x86_u8x8(data: &[u8; 8], target: u8) -> u32 {
    use core::arch::x86_64::*;
    let v = _mm_loadl_epi64(data.as_ptr() as *const __m128i);
    let splat = _mm_set1_epi8(target as i8);
    #[cfg(all(target_feature = "avx512vl", target_feature = "avx512bw"))]
    {
        return (_mm_cmpeq_epi8_mask(v, splat) as u32) & 0xFF;
    }
    #[cfg(not(all(target_feature = "avx512vl", target_feature = "avx512bw")))]
    {
        return (_mm_movemask_epi8(_mm_cmpeq_epi8(v, splat)) as u32) & 0xFF;
    }
}

/// ARITY == 16 filter scan, x86: one xmm compare, sixteen mask bits.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn eq_mask_x86_u8x16(data: &[u8; 16], target: u8) -> u32 {
    use core::arch::x86_64::*;
    let v = _mm_loadu_si128(data.as_ptr() as *const __m128i);
    let splat = _mm_set1_epi8(target as i8);
    #[cfg(all(target_feature = "avx512vl", target_feature = "avx512bw"))]
    {
        return _mm_cmpeq_epi8_mask(v, splat) as u32;
    }
    #[cfg(not(all(target_feature = "avx512vl", target_feature = "avx512bw")))]
    {
        return _mm_movemask_epi8(_mm_cmpeq_epi8(v, splat)) as u32;
    }
}

/// ARITY == 32 filter scan, x86: one ymm compare under AVX-512VL, two xmm halves otherwise.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn eq_mask_x86_u8x32(data: &[u8; 32], target: u8) -> u32 {
    use core::arch::x86_64::*;
    #[cfg(all(target_feature = "avx512vl", target_feature = "avx512bw"))]
    {
        let v = _mm256_loadu_si256(data.as_ptr() as *const __m256i);
        return _mm256_cmpeq_epi8_mask(v, _mm256_set1_epi8(target as i8)) as u32;
    }
    #[cfg(not(all(target_feature = "avx512vl", target_feature = "avx512bw")))]
    {
        let splat = _mm_set1_epi8(target as i8);
        let lo = _mm_cmpeq_epi8(_mm_loadu_si128(data.as_ptr() as *const __m128i), splat);
        let hi = _mm_cmpeq_epi8(
            _mm_loadu_si128(data.as_ptr().add(16) as *const __m128i),
            splat,
        );
        return (_mm_movemask_epi8(lo) as u32) | ((_mm_movemask_epi8(hi) as u32) << 16);
    }
}

/// Lowest set bit of an equality mask, or the `u8::MAX` no-match sentinel.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn first_from_mask_x86(mask: u32) -> u32 {
    if mask == 0 {
        u8::MAX as u32
    } else {
        mask.trailing_zeros()
    }
}

/// The bucket's available slots: those still `EMPTY_FILTER`, plus the last one when a delete left
/// `TOMBSTONE_FILTER` there. Lanes past the last slot are the next bucket's and are dropped.
#[cfg_attr(not(feature = "profiling"), inline(always))]
#[cfg_attr(feature = "profiling", inline(never))]
pub(crate) fn available_lanes<const BUCKET_SIZE: usize, const SIMD_ARITY: usize>(
    window: &[u8; SIMD_ARITY],
) -> u32 {
    let mut available =
        find_all_simd_u8_overread::<BUCKET_SIZE, SIMD_ARITY, false>(window, crate::EMPTY_FILTER);
    if SIMD_ARITY > BUCKET_SIZE {
        available &= u32::MAX >> (32 - BUCKET_SIZE);
    }
    if window[BUCKET_SIZE - 1] == crate::TOMBSTONE_FILTER {
        available |= 1u32 << (BUCKET_SIZE - 1);
    }
    available
}

#[cfg(test)]
fn scalar_find_all(data: &[u8], target: u8) -> u32 {
    let mut mask = 0u32;
    for (i, &b) in data.iter().enumerate() {
        if b == target {
            mask |= 1 << i;
        }
    }
    mask
}

#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn find_all_u64(data: &u64, target: u8, len: usize) -> u64 {
    let one_r = 0x0101010101010101u64;
    let cmp = data ^ (target as u64 * one_r);
    let bitmask = cmp.wrapping_sub(one_r) & !cmp & 0x8080808080808080u64;
    let masked = bitmask & (u64::MAX >> (64 - len * 8));
    (masked.wrapping_mul(0x0002040810204081)) >> 56
}

#[cfg(any(nightly_compiler, feature = "nightly"))]
#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn find_all_simd_u8_exact<const ARITY: usize>(
    data: [u8; ARITY],
    target: u8,
    len: usize,
) -> u64 {
    let target_splat = Simd::<u8, ARITY>::splat(target);

    let vector = Simd::<u8, ARITY>::from_array(data);
    let mask = vector.simd_eq(target_splat);

    let bitmask = mask.to_bitmask();

    bitmask & (u64::MAX >> (64 - len))
}

/// Stable twin of `find_all_simd_u8_exact`: bit `i` set iff `data[i] == target`, below `len`.
#[cfg(not(any(nightly_compiler, feature = "nightly")))]
#[cfg_attr(not(feature = "profiling"), inline)]
#[cfg_attr(feature = "profiling", inline(never))]
pub fn find_all_simd_u8_exact<const ARITY: usize>(
    data: [u8; ARITY],
    target: u8,
    len: usize,
) -> u64 {
    let mut bitmask = 0u64;
    for (i, &b) in data.iter().enumerate() {
        if b == target {
            bitmask |= 1 << i;
        }
    }
    bitmask & (u64::MAX >> (64 - len))
}

#[inline(always)]
pub fn fast_eq<const N: usize>(a: &[u8; N], b: &[u8; N]) -> bool {
    if N > 32 {
        for i in 0..N / 16 {
            let a1 = u128::from_ne_bytes(a[i * 16..i * 16 + 16].try_into().unwrap());
            let b1 = u128::from_ne_bytes(b[i * 16..i * 16 + 16].try_into().unwrap());
            if a1 != b1 {
                return false;
            }
        }
        if !N.is_multiple_of(16) {
            let a1 = u128::from_ne_bytes(a[N - 16..N].try_into().unwrap());
            let b1 = u128::from_ne_bytes(b[N - 16..N].try_into().unwrap());
            if a1 != b1 {
                return false;
            }
        }
        return true;
    }

    if N >= 16 {
        let a1 = u128::from_ne_bytes(a[0..16].try_into().unwrap());
        let b1 = u128::from_ne_bytes(b[0..16].try_into().unwrap());

        // The tail load may overlap the head, which is harmless and saves a length branch.
        let a2 = u128::from_ne_bytes(a[N - 16..N].try_into().unwrap());
        let b2 = u128::from_ne_bytes(b[N - 16..N].try_into().unwrap());

        return a1 == b1 && a2 == b2;
    }

    if N >= 8 {
        let a1 = u64::from_ne_bytes(a[0..8].try_into().unwrap());
        let b1 = u64::from_ne_bytes(b[0..8].try_into().unwrap());

        let a2 = u64::from_ne_bytes(a[N - 8..N].try_into().unwrap());
        let b2 = u64::from_ne_bytes(b[N - 8..N].try_into().unwrap());

        return a1 == b1 && a2 == b2;
    }

    if N >= 4 {
        // A key is a byte array with no alignment, hence `read_unaligned`.
        let a1 = unsafe { (a.as_ptr() as *const u32).read_unaligned() };
        let b1 = unsafe { (b.as_ptr() as *const u32).read_unaligned() };

        let a2 = u32::from_ne_bytes(a[N - 4..N].try_into().unwrap());
        let b2 = u32::from_ne_bytes(b[N - 4..N].try_into().unwrap());

        return a1 == b1 && a2 == b2;
    }

    if N >= 2 {
        let a1 = u16::from_ne_bytes(a[0..2].try_into().unwrap());
        let b1 = u16::from_ne_bytes(b[0..2].try_into().unwrap());

        let a2 = u16::from_ne_bytes(a[N - 2..N].try_into().unwrap());
        let b2 = u16::from_ne_bytes(b[N - 2..N].try_into().unwrap());

        return a1 == b1 && a2 == b2;
    }

    if N == 1 {
        return a[0] == b[0];
    }

    true
}

/// The paper's level sizing, as-is: what the `agile` table builds with.
pub fn bucket_cnts_for_utilization(q: u64, s: u32, a: f64, max_levels: usize) -> Vec<u32> {
    crate::sizing::bucket_cnts(q, s, a, max_levels)
}

/// How far past its asked-for size a level may round up to reach a power of two.
pub const POWERS_TOLERANCE: f64 = 0.01;

/// The paper's level sizing with every level rounded to a power of two for shift addressing.
pub fn bucket_cnts_for_utilization_powers(q: u64, s: u32, a: f64, max_levels: usize) -> Vec<u32> {
    crate::sizing::bucket_cnts_powers(q, s, a, max_levels, POWERS_TOLERANCE)
}

/// The paper's level sizing with only the first level rounded to a power of two.
pub fn bucket_cnts_for_utilization_first_power(
    q: u64,
    s: u32,
    a: f64,
    max_levels: usize,
) -> Vec<u32> {
    crate::sizing::bucket_cnts_first_power(q, s, a, max_levels, POWERS_TOLERANCE)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Cross-checks both `find_all_simd_u8_overread` forms against the scalar reference.
    fn assert_overread_matches<const LEN: usize, const ARITY: usize>(
        data: &[u8; ARITY],
        target: u8,
    ) {
        let expected = scalar_find_all(&data[..], target);
        let early = find_all_simd_u8_overread::<LEN, ARITY, true>(data, target);
        let eager = find_all_simd_u8_overread::<LEN, ARITY, false>(data, target);
        assert_eq!(early, expected, "early-out: data={data:?} target={target}");
        assert_eq!(eager, expected, "eager: data={data:?} target={target}");
    }

    #[test]
    fn overread_arity_8() {
        assert_overread_matches::<8, 8>(&[0xAAu8; 8], 0x42);
        assert_overread_matches::<8, 8>(&[0x42u8; 8], 0x42);
        for i in 0..8 {
            let mut data = [0xAAu8; 8];
            data[i] = 0x42;
            assert_overread_matches::<8, 8>(&data, 0x42);
            assert_eq!(
                find_all_simd_u8_overread::<8, 8, true>(&data, 0x42),
                1u32 << i
            );
        }
    }

    #[test]
    fn overread_no_match_arity_16() {
        assert_overread_matches::<8, 16>(&[0xAAu8; 16], 0x42);
    }

    #[test]
    fn overread_no_match_arity_32() {
        assert_overread_matches::<16, 32>(&[0xAAu8; 32], 0x42);
    }

    #[test]
    fn overread_each_single_position_sets_its_own_bit_arity_16() {
        for i in 0..16 {
            let mut data = [0xAAu8; 16];
            data[i] = 0x42;
            assert_overread_matches::<8, 16>(&data, 0x42);
            assert_eq!(
                find_all_simd_u8_overread::<8, 16, true>(&data, 0x42),
                1u32 << i
            );
        }
    }

    #[test]
    fn overread_each_single_position_sets_its_own_bit_arity_32() {
        for i in 0..32 {
            let mut data = [0xAAu8; 32];
            data[i] = 0x42;
            assert_overread_matches::<16, 32>(&data, 0x42);
            assert_eq!(
                find_all_simd_u8_overread::<16, 32, true>(&data, 0x42),
                1u32 << i
            );
        }
    }

    #[test]
    fn overread_multiple_matches_arity_16() {
        let mut data = [0xAAu8; 16];
        for &i in &[0usize, 5, 8, 15] {
            data[i] = 0x42;
        }
        assert_overread_matches::<8, 16>(&data, 0x42);
        assert_overread_matches::<8, 16>(&[0x42u8; 16], 0x42);
    }

    #[test]
    fn overread_multiple_matches_arity_32() {
        let mut data = [0xAAu8; 32];
        for &i in &[0usize, 5, 8, 17, 31] {
            data[i] = 0x42;
        }
        assert_overread_matches::<16, 32>(&data, 0x42);
        assert_overread_matches::<16, 32>(&[0x42u8; 32], 0x42);
    }

    #[test]
    fn overread_tail_region_only_arity_16() {
        // LEN=8 < ARITY=16, so these matches sit only in the over-read tail past the bucket.
        let mut data = [0xAAu8; 16];
        data[9] = 0x42;
        data[15] = 0x42;
        assert_overread_matches::<8, 16>(&data, 0x42);
    }

    #[test]
    fn overread_tail_region_only_arity_32() {
        let mut data = [0xAAu8; 32];
        data[17] = 0x42;
        data[31] = 0x42;
        assert_overread_matches::<16, 32>(&data, 0x42);
    }

    #[test]
    fn overread_target_empty_filter_byte_arity_16() {
        // Mirrors lib.rs's EMPTY_FILTER (0u8), used to find open slots.
        let mut data = [5u8; 16];
        data[2] = 0;
        data[11] = 0;
        assert_overread_matches::<8, 16>(&data, 0);
    }

    #[test]
    fn overread_target_empty_filter_byte_arity_32() {
        let mut data = [5u8; 32];
        data[2] = 0;
        data[20] = 0;
        assert_overread_matches::<16, 32>(&data, 0);
    }

    #[test]
    fn overread_target_tombstone_filter_byte_arity_16() {
        // Mirrors lib.rs's TOMBSTONE_FILTER (1u8), what a delete leaves in a bucket's last slot.
        let mut data = [5u8; 16];
        data[0] = 1;
        data[15] = 1;
        assert_overread_matches::<8, 16>(&data, 1);
    }

    #[test]
    fn overread_target_tombstone_filter_byte_arity_32() {
        let mut data = [5u8; 32];
        data[0] = 1;
        data[31] = 1;
        assert_overread_matches::<16, 32>(&data, 1);
    }

    #[test]
    fn available_lanes_offers_a_tombstoned_last_slot_but_not_an_over_read_one() {
        // An 8-slot bucket under a 16-lane window: lane 7 is its last slot, lane 15 a neighbour's.
        let mut window = [crate::MIN_KEY_FILTER; 16];
        window[7] = crate::TOMBSTONE_FILTER;
        window[15] = crate::TOMBSTONE_FILTER;
        assert_eq!(available_lanes::<8, 16>(&window), 1 << 7);
        window[3] = crate::EMPTY_FILTER;
        window[12] = crate::EMPTY_FILTER;
        assert_eq!(available_lanes::<8, 16>(&window), (1 << 3) | (1 << 7));
        window[7] = crate::MIN_KEY_FILTER;
        assert_eq!(available_lanes::<8, 16>(&window), 1 << 3);

        // A full-width window: the tombstone in the last lane is the bucket's own.
        let mut full = [crate::MIN_KEY_FILTER; 16];
        full[15] = crate::TOMBSTONE_FILTER;
        assert_eq!(available_lanes::<16, 16>(&full), 1 << 15);
        full[15] = crate::EMPTY_FILTER;
        assert_eq!(available_lanes::<16, 16>(&full), 1 << 15);
        full[15] = crate::MIN_KEY_FILTER;
        assert_eq!(available_lanes::<16, 16>(&full), 0);
    }

    #[test]
    fn overread_brute_force_cross_check_arity_16() {
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next_byte = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as u8
        };
        for _trial in 0..50 {
            let mut data = [0u8; 16];
            for b in data.iter_mut() {
                *b = next_byte();
            }
            let target = if next_byte() % 2 == 0 {
                data[next_byte() as usize % 16]
            } else {
                next_byte()
            };
            assert_overread_matches::<8, 16>(&data, target);
        }
    }

    #[test]
    fn overread_brute_force_cross_check_arity_32() {
        let mut state = 0x2545F4914F6CDD1Du64 ^ 0xA5A5_A5A5_A5A5_A5A5;
        let mut next_byte = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as u8
        };
        for _trial in 0..50 {
            let mut data = [0u8; 32];
            for b in data.iter_mut() {
                *b = next_byte();
            }
            let target = if next_byte() % 2 == 0 {
                data[next_byte() as usize % 32]
            } else {
                next_byte()
            };
            assert_overread_matches::<16, 32>(&data, target);
        }
    }

    /// Cross-checks `find_first_simd_u8_overread` against its contract and the mask form.
    fn assert_first_matches<const ARITY: usize>(data: &[u8; ARITY], target: u8) {
        let expected = data
            .iter()
            .position(|&b| b == target)
            .map_or(u8::MAX as u32, |i| i as u32);
        let actual = find_first_simd_u8_overread::<ARITY>(data, target);
        assert_eq!(actual, expected, "data={data:?} target={target}");

        let mask = find_all_simd_u8_overread::<ARITY, ARITY, false>(data, target);
        let from_mask = if mask == 0 {
            u8::MAX as u32
        } else {
            mask.trailing_zeros()
        };
        assert_eq!(
            actual, from_mask,
            "mask disagrees: data={data:?} target={target}"
        );
    }

    fn first_overread_suite<const ARITY: usize>() {
        assert_first_matches::<ARITY>(&[0xAAu8; ARITY], 0x42);
        assert_first_matches::<ARITY>(&[0x42u8; ARITY], 0x42);
        for i in 0..ARITY {
            let mut data = [0xAAu8; ARITY];
            data[i] = 0x42;
            assert_first_matches::<ARITY>(&data, 0x42);
            assert_eq!(find_first_simd_u8_overread::<ARITY>(&data, 0x42), i as u32);
        }
        // Two matches: the earlier one wins, whatever the reduction order.
        for i in 0..ARITY {
            for j in (i + 1)..ARITY {
                let mut data = [0xAAu8; ARITY];
                data[i] = 0x42;
                data[j] = 0x42;
                assert_eq!(find_first_simd_u8_overread::<ARITY>(&data, 0x42), i as u32);
            }
        }
        // 0 and 1 are lib.rs's EMPTY_FILTER and TOMBSTONE_FILTER; zero catches an unmasked load.
        for target in [0u8, 1u8] {
            assert_first_matches::<ARITY>(&[5u8; ARITY], target);
            let mut data = [5u8; ARITY];
            data[ARITY - 1] = target;
            assert_first_matches::<ARITY>(&data, target);
        }
    }

    #[test]
    fn first_overread_arity_8() {
        first_overread_suite::<8>();
    }

    #[test]
    fn first_overread_arity_16() {
        first_overread_suite::<16>();
    }

    #[test]
    fn first_overread_arity_32() {
        first_overread_suite::<32>();
    }

    #[test]
    fn first_overread_brute_force_cross_check() {
        let mut state = 0x2545F4914F6CDD1Du64 ^ 0x5A5A_5A5A_5A5A_5A5A;
        let mut next_byte = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as u8
        };
        for _trial in 0..50 {
            let mut data = [0u8; 32];
            for b in data.iter_mut() {
                *b = next_byte();
            }
            let target = if next_byte() % 2 == 0 {
                data[next_byte() as usize % 32]
            } else {
                next_byte()
            };
            let head8: &[u8; 8] = data[..8].try_into().unwrap();
            let head16: &[u8; 16] = data[..16].try_into().unwrap();
            assert_first_matches::<8>(head8, target);
            assert_first_matches::<16>(head16, target);
            assert_first_matches::<32>(&data, target);
        }
    }

    #[test]
    fn hash_key_v3_key_len_8_arm_distinguishes_distinct_keys() {
        let hasher = FoldRandomState::default();
        let hashes: std::collections::HashSet<u64> = (0..256u64)
            .map(|i| hash_key_v3(&hasher, &i.to_ne_bytes()))
            .collect();
        assert!(hashes.len() >= 250, "too many collisions: {}", hashes.len());
    }

    #[test]
    fn hash_key_v3_key_len_16_arm_distinguishes_distinct_keys() {
        let hasher = FoldRandomState::default();
        let hashes: std::collections::HashSet<u64> = (0..256u128)
            .map(|i| hash_key_v3(&hasher, &i.to_ne_bytes()))
            .collect();
        assert!(hashes.len() >= 250, "too many collisions: {}", hashes.len());
    }
}
