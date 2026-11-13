//! Whether a value is in a short list, compared with every value at once with
//! each target's vector instructions. Left to itself, the compiler vectorizes
//! the comparison or not depending on what it's inlined into.

/// Whether `cell` is one of `list`, whose length is a multiple of 4.
#[inline]
pub(crate) fn listed<const N: usize>(list: &[i64; N], cell: i64) -> bool {
    const { assert!(N.is_multiple_of(4)) };
    imp::listed(list, cell)
}

/// AVX2: four values a register.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod imp {
    use core::arch::x86_64::{
        __m256i, _mm256_cmpeq_epi64, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi64x,
        _mm256_setzero_si256, _mm256_testz_si256,
    };

    #[inline]
    pub(super) fn listed<const N: usize>(list: &[i64; N], cell: i64) -> bool {
        // SAFETY: the build has AVX2, and each load reads 4 values of `list`.
        unsafe {
            let cell = _mm256_set1_epi64x(cell);
            let mut found = _mm256_setzero_si256();
            for four in list.as_chunks::<4>().0 {
                #[expect(clippy::cast_ptr_alignment, reason = "an unaligned load")]
                let four = _mm256_loadu_si256(four.as_ptr().cast::<__m256i>());
                found = _mm256_or_si256(found, _mm256_cmpeq_epi64(four, cell));
            }
            _mm256_testz_si256(found, found) == 0
        }
    }
}

/// NEON: two values a register.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod imp {
    use core::arch::aarch64::{
        vceqq_s64, vdupq_n_s64, vdupq_n_u64, vld1q_s64, vmaxvq_u32, vorrq_u64,
        vreinterpretq_u32_u64,
    };

    #[inline]
    pub(super) fn listed<const N: usize>(list: &[i64; N], cell: i64) -> bool {
        // SAFETY: NEON is part of aarch64, and each load reads 2 values of
        // `list`.
        unsafe {
            let cell = vdupq_n_s64(cell);
            let mut found = vdupq_n_u64(0);
            for two in list.as_chunks::<2>().0 {
                found = vorrq_u64(found, vceqq_s64(vld1q_s64(two.as_ptr()), cell));
            }
            vmaxvq_u32(vreinterpretq_u32_u64(found)) != 0
        }
    }
}

/// Anywhere else: compared one by one, without a branch. On wasm, explicit
/// vectors made the build bigger, with no benchmark there to show a gain.
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
)))]
mod imp {
    #[inline]
    #[expect(clippy::needless_bitwise_bool, reason = "without a branch")]
    pub(super) fn listed<const N: usize>(list: &[i64; N], cell: i64) -> bool {
        list.iter().fold(false, |found, &value| found | (value == cell))
    }
}
