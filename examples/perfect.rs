use deepsize::DeepSizeOf;
use multitable::MultiTable;
use rand::prelude::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rustc_hash::FxBuildHasher;

const KEY_LEN: usize = 32;
const VAL_LEN: usize = 128;
const BUCKET_SIZE_FOR_IDEAL: usize = 64;
fn gen_unique_numbers(n: usize) -> Vec<u32> {
    let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(2);
    let mut nums: Vec<_> = rand::seq::index::sample(&mut rng, u32::MAX as usize, n)
        .iter()
        .map(|v| v as u32)
        .collect();
    nums.shuffle(&mut rng);
    nums
}

const MAX_LEVELS: usize = 16;
fn init_ideal(
    cnts: Vec<u32>,
    size: u64,
) -> MultiTable<
    KEY_LEN,
    VAL_LEN,
    MAX_LEVELS,
    FxBuildHasher,
    BUCKET_SIZE_FOR_IDEAL,
    BUCKET_SIZE_FOR_IDEAL,
> {
    let mut mtable = MultiTable::<
        KEY_LEN,
        VAL_LEN,
        MAX_LEVELS,
        FxBuildHasher,
        BUCKET_SIZE_FOR_IDEAL,
        BUCKET_SIZE_FOR_IDEAL,
    >::new_from_cnts(cnts);

    let keys = gen_unique_numbers(size as usize);
    for i in 0..size {
        let mut key = [0u8; KEY_LEN];
        key[0..16].copy_from_slice(&(keys[i as usize] as u128).to_ne_bytes()[..]);
        let mut value = [0u8; VAL_LEN];
        value[0..8].copy_from_slice((i * 2).to_ne_bytes().as_slice());

        let res = mtable.insert(key, value);
        if res.is_err() {
            println!("Err: {:?}, key: {:?}", res, key);
        }
        assert!(res.is_ok(), "Shall succeed {i}");
    }

    mtable
}
fn main() {
    // Computed separately for load factor 1
    let cnts = vec![4872, 1947, 752, 280, 100, 34, 11, 1, 1, 1, 1];
    let size = (cnts.iter().sum::<u32>() as u64) * BUCKET_SIZE_FOR_IDEAL as u64;
    let mt = init_ideal(cnts, size);
    let byte_size = mt.deep_size_of();
    let a = (size * (KEY_LEN + VAL_LEN) as u64) as f32 / byte_size as f32;
    println!("100% of {size} elements inserted, theoretical load factor: 1");
    println!("Raw load factor: {a}");
}
