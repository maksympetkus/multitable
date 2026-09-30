#![cfg_attr(
    any(nightly_compiler, feature = "nightly"),
    feature(portable_simd, core_intrinsics)
)]
// The crate opts into compiler-internal `core_intrinsics` for the SIMD arms build.rs turns on.
#![cfg_attr(any(nightly_compiler, feature = "nightly"), allow(internal_features))]
//! Exactly one variant feature must be on: `agile` (default), `powers-of-2` or `first-power`.
//! `build.rs` turns the SIMD paths on for a nightly compiler by itself; `nightly` forces them.

#[cfg(not(any(feature = "agile", feature = "powers-of-2", feature = "first-power")))]
compile_error!(
    "Exactly one variant feature must be enabled: `agile`, `powers-of-2`, or `first-power`."
);

#[cfg(any(
    all(feature = "agile", feature = "powers-of-2"),
    all(feature = "agile", feature = "first-power"),
    all(feature = "powers-of-2", feature = "first-power")
))]
compile_error!("Features `agile`, `powers-of-2`, or `first-power` are mutually exclusive.");

mod sizing;
pub mod utils;

use std::hash::BuildHasher;

use crate::utils::*;
use deepsize::DeepSizeOf;
use std::iter::once;

/// The progressive overfill ratio the paper's sizing builds its cascade with.
pub fn sizing_ratio() -> f64 {
    sizing::Params::default().ratio
}

#[cfg(feature = "jema")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(feature = "mima")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// A bucket keeps its keys ahead of its values so the key scan runs while the values load.
#[derive(Clone, Copy)]
pub struct Bucket<
    const KEY_LEN: usize,
    const VAL_LEN: usize,
    const BUCKET_SIZE: usize = DEFAULT_BUCKET_SIZE,
> {
    keys: [[u8; KEY_LEN]; BUCKET_SIZE],
    values: [[u8; VAL_LEN]; BUCKET_SIZE],
}

impl<const KEY_LEN: usize, const VAL_LEN: usize, const BUCKET_SIZE: usize> Default
    for Bucket<KEY_LEN, VAL_LEN, BUCKET_SIZE>
{
    fn default() -> Self {
        Self {
            keys: [[0u8; KEY_LEN]; BUCKET_SIZE],
            values: [[0u8; VAL_LEN]; BUCKET_SIZE],
        }
    }
}

impl<const KEY_LEN: usize, const VAL_LEN: usize, const BUCKET_SIZE: usize> DeepSizeOf
    for Bucket<KEY_LEN, VAL_LEN, BUCKET_SIZE>
{
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        0
    }
}

const CACHE_LINE: usize = 64;

/// A cache line when a bucket is a whole number of them, since over-aligning anything else pads.
const fn bucket_align<T>() -> usize {
    if size_of::<T>() != 0 && size_of::<T>().is_multiple_of(CACHE_LINE) {
        CACHE_LINE
    } else {
        align_of::<T>()
    }
}

/// The buckets in one zeroed allocation: a `Vec<T>` in all but the alignment it asks for.
pub struct BucketStore<T> {
    ptr: std::ptr::NonNull<T>,
    len: usize,
}

// SAFETY: the allocation is owned exclusively, as `Vec`'s is.
unsafe impl<T: Send> Send for BucketStore<T> {}
unsafe impl<T: Sync> Sync for BucketStore<T> {}

impl<T> BucketStore<T> {
    /// Nothing here drops its elements, so nothing stored here may need it.
    const NO_DROP: () = assert!(
        !std::mem::needs_drop::<T>(),
        "BucketStore never runs element destructors"
    );

    fn layout(len: usize) -> std::alloc::Layout {
        std::alloc::Layout::array::<T>(len)
            .and_then(|layout| layout.align_to(bucket_align::<T>()))
            .expect("bucket storage exceeds the address space")
    }

    /// One zeroed allocation of `len` elements, over-aligned when a bucket is whole cache lines.
    ///
    /// # Safety
    /// All-zero must be a valid `T`, as it is for the byte-array buckets.
    pub unsafe fn new_zeroed(len: usize) -> Self {
        let () = Self::NO_DROP;

        let layout = Self::layout(len);
        if layout.size() == 0 {
            return Self {
                ptr: std::ptr::NonNull::dangling(),
                len,
            };
        }
        match std::ptr::NonNull::new(std::alloc::alloc_zeroed(layout).cast::<T>()) {
            Some(ptr) => Self { ptr, len },
            None => std::alloc::handle_alloc_error(layout),
        }
    }
}

impl<T> Drop for BucketStore<T> {
    fn drop(&mut self) {
        let layout = Self::layout(self.len);
        if layout.size() != 0 {
            // The same layout the allocation used, and a zero-size layout never allocated.
            unsafe { std::alloc::dealloc(self.ptr.as_ptr().cast::<u8>(), layout) };
        }
    }
}

impl<T> std::ops::Deref for BucketStore<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // `ptr` and `len` are what `new_zeroed` allocated, and neither is ever handed out alone.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> std::ops::DerefMut for BucketStore<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // The pair `new_zeroed` allocated, and `&mut self` makes this the only slice over it.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for BucketStore<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

impl<T: DeepSizeOf> DeepSizeOf for BucketStore<T> {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.iter()
            .map(|bucket| bucket.deep_size_of_children(context))
            .sum::<usize>()
            + self.len * size_of::<T>()
    }
}

/// Best-effort `MADV_HUGEPAGE`, Linux and `MT_HUGEPAGE=1` only, so a benchmark stays comparable.
#[cfg(target_os = "linux")]
fn advise_hugepages<T>(slice: &[T]) {
    const MADV_HUGEPAGE: i32 = 14;
    // A host with a larger base page simply declines the advice.
    const PAGE: usize = 4096;

    extern "C" {
        fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
    }

    if std::env::var_os("MT_HUGEPAGE").is_none_or(|on| on != "1") {
        return;
    }

    let base = slice.as_ptr() as usize;
    let (start, end) = (
        base.next_multiple_of(PAGE),
        (base + size_of_val(slice)) / PAGE * PAGE,
    );
    if end > start {
        // `start` and `end` are page-aligned bounds inside the slice, and the range is nonempty.
        unsafe { madvise(start as *mut std::ffi::c_void, end - start, MADV_HUGEPAGE) };
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_hugepages<T>(_slice: &[T]) {}

include!("plain.rs");

type LevelOffset = u32;
type BucketCnts = u32;

/// A level's hash selector: the full hash rotated by `8 * level + 4`, so no level reuses a half.
#[cfg_attr(not(feature = "profiling"), inline(always))]
#[cfg_attr(feature = "profiling", inline(never))]
fn level_hash_selector(raw: u64, level: usize) -> u32 {
    raw.rotate_right((8 * level as u32 + 4) % 64) as u32
}

/// The filtered table's level 0 bucket: a multiply-high on the full hash, strong bits leading.
#[cfg(feature = "agile")]
#[cfg_attr(not(feature = "profiling"), inline(always))]
#[cfg_attr(feature = "profiling", inline(never))]
fn first_level_bucket(raw: u64, cnt: BucketCnts) -> u32 {
    (((raw as u128) * (cnt as u128)) >> 64) as u32
}

/// The same for the cfgs that shift-address the filtered table's level 0: the hash's top half.
#[cfg(any(feature = "powers-of-2", feature = "first-power"))]
#[cfg_attr(not(feature = "profiling"), inline(always))]
#[cfg_attr(feature = "profiling", inline(never))]
fn first_hash_selector(raw: u64) -> u32 {
    (raw >> 32) as u32
}

/// Default `BUCKET_SIZE`, set at build time with `BUCKET_SIZE=<n>`.
pub const DEFAULT_BUCKET_SIZE: usize = match option_env!("BUCKET_SIZE") {
    Some(s) => match usize::from_str_radix(s, 10) {
        Ok(size) => size,
        _ => unreachable!(),
    },
    None => 8,
};

/// Separate from `BUCKET_SIZE` so a filter scan keeps a fixed-width SIMD load at any bucket width.
pub const DEFAULT_SIMD_ARITY: usize = if DEFAULT_BUCKET_SIZE <= 8 {
    8
} else if DEFAULT_BUCKET_SIZE <= 16 {
    16
} else if DEFAULT_BUCKET_SIZE <= 32 {
    32
} else {
    64
};

pub const DEFAULT_FILTERED_BUCKET_SIZE: usize = match option_env!("BUCKET_SIZE") {
    Some(s) => match usize::from_str_radix(s, 10) {
        Ok(size) => size,
        _ => unreachable!(),
    },
    None => 16,
};

/// Separate from `BUCKET_SIZE` so a filter scan keeps a fixed-width SIMD load at any bucket width.
pub const DEFAULT_FILTERED_SIMD_ARITY: usize = if DEFAULT_FILTERED_BUCKET_SIZE <= 8 {
    8
} else if DEFAULT_FILTERED_BUCKET_SIZE <= 16 {
    16
} else if DEFAULT_FILTERED_BUCKET_SIZE <= 32 {
    32
} else {
    64
};

include!("filtered.rs");

use rustc_hash::{FxBuildHasher, FxHasher};
type HasherBuilder = FxBuildHasher;

/// Builds `FxHasher`s from the seed it carries; `Default` draws that seed at random.
#[derive(Copy, Clone)]
pub struct SeededHasherBuilder(pub usize);

impl BuildHasher for SeededHasherBuilder {
    type Hasher = FxHasher;
    fn build_hasher(&self) -> FxHasher {
        FxHasher::with_seed(self.0)
    }
}

impl Default for SeededHasherBuilder {
    fn default() -> Self {
        Self(rand::random::<u64>() as usize)
    }
}

use seq_macro::seq;

type BucketNumber = u32;
type KeyIndex = u32;
type Level = usize;

/// What a [`MultiTableFiltered`] walk found; `GET_VALUE` picks the `Found` or `FoundValue` arm.
#[derive(PartialEq)]
pub enum SearchResult<'a, const VAL_LEN: usize> {
    NotFound,
    NotFoundFirstEmpty(BucketNumber, KeyIndex, Level, u8),
    Found(BucketNumber, KeyIndex),
    FoundValue(&'a [u8; VAL_LEN]),
}

/// A slot never written, or one a delete cleared. In a last slot it says the bucket never
/// filled, which is the probe's stop. Whichever of the two bytes it writes, a delete also has to
/// zero the table's `plain_insert_frontier`, or the inserts never go looking for the room it left.
const EMPTY_FILTER: u8 = 0u8;
/// A deleted last slot, the one place a delete must not write zero, since that byte has to keep
/// saying the bucket once filled. The recording walk offers it for reuse.
const TOMBSTONE_FILTER: u8 = 1u8;
/// Floor on a key's filter byte, past the tombstone, so no live key ever reads as deleted.
const MIN_KEY_FILTER: u8 = TOMBSTONE_FILTER + 1;

#[cfg(test)]
mod tests;
