//! Comparisons of eight values at once, with each target's vector
//! instructions, answering as the bits of a byte: bit `j` for the value at
//! `at + j`. Where a target has none, `EIGHT_AT_A_TIME` is false and filters
//! don't call them.

/// Whether this target compares eight values at once.
pub(crate) const EIGHT_AT_A_TIME: bool = cfg!(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128"),
));

/// Which of the eight `cells` from `at` are in `lo..=lo + span`, taken as
/// `(cell - lo) as u64 <= span`.
///
/// # Safety
///
/// `cells` must have a cell at `at + 7`.
#[inline]
pub(crate) unsafe fn in_range(cells: &[i64], at: usize, lo: i64, span: u64) -> u8 {
    // SAFETY: as the caller promises.
    unsafe { imp::in_range(cells.as_ptr().add(at), lo, span) }
}

/// Which of the eight `indices` from `at` are `index`.
///
/// # Safety
///
/// `indices` must have an index at `at + 7`.
#[inline]
pub(crate) unsafe fn equal(indices: &[u32], at: usize, index: u32) -> u8 {
    // SAFETY: as the caller promises.
    unsafe { imp::equal(indices.as_ptr().add(at), index) }
}

/// AVX2: four lanes a register. Its comparisons are signed, so the unsigned
/// one flips each top bit first.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod imp {
    use core::arch::x86_64::{
        __m256i, _mm256_castsi256_pd, _mm256_castsi256_ps, _mm256_cmpeq_epi32, _mm256_cmpgt_epi64,
        _mm256_loadu_si256, _mm256_movemask_pd, _mm256_movemask_ps, _mm256_set1_epi32,
        _mm256_set1_epi64x, _mm256_sub_epi64, _mm256_xor_si256,
    };

    #[inline]
    pub(super) unsafe fn in_range(cells: *const i64, lo: i64, span: u64) -> u8 {
        // SAFETY: the build has AVX2, and the caller that the cells are there.
        unsafe {
            let top = _mm256_set1_epi64x(i64::MIN);
            let lo = _mm256_set1_epi64x(lo);
            let span = _mm256_xor_si256(_mm256_set1_epi64x(span.cast_signed()), top);
            let outside = |cells: *const i64| {
                #[expect(clippy::cast_ptr_alignment, reason = "an unaligned load")]
                let keys = _mm256_loadu_si256(cells.cast::<__m256i>());
                let offsets = _mm256_xor_si256(_mm256_sub_epi64(keys, lo), top);
                _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(offsets, span)))
            };
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "8 bits")]
            let outside = (outside(cells) | outside(cells.add(4)) << 4) as u8;
            !outside
        }
    }

    #[inline]
    pub(super) unsafe fn equal(indices: *const u32, index: u32) -> u8 {
        // SAFETY: the build has AVX2, and the caller that the indices are there.
        unsafe {
            #[expect(clippy::cast_ptr_alignment, reason = "an unaligned load")]
            let eight = _mm256_loadu_si256(indices.cast::<__m256i>());
            let equal = _mm256_cmpeq_epi32(eight, _mm256_set1_epi32(index.cast_signed()));
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "8 bits")]
            let byte = _mm256_movemask_ps(_mm256_castsi256_ps(equal)) as u8;
            byte
        }
    }
}

/// NEON: two 64-bit lanes a register, with unsigned comparisons. It has no
/// instruction gathering a bit a lane, so each lane's mask is narrowed to a
/// byte, kept to its own bit, and the bytes added across.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod imp {
    use core::arch::aarch64::{
        uint8x8_t, vaddv_u8, vand_u8, vceqq_u32, vcleq_u64, vcombine_u16, vcombine_u32,
        vdupq_n_s64, vdupq_n_u32, vdupq_n_u64, vld1_u8, vld1q_s64, vld1q_u32, vmovn_u16, vmovn_u32,
        vmovn_u64, vreinterpretq_u64_s64, vsubq_s64,
    };

    /// Bit `j` in byte `j`.
    static BITS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

    /// Each of eight bytes, all ones or all zeros, as one bit of a byte.
    #[inline]
    unsafe fn bits(eight: uint8x8_t) -> u8 {
        // SAFETY: NEON is part of aarch64.
        unsafe { vaddv_u8(vand_u8(eight, vld1_u8(BITS.as_ptr()))) }
    }

    #[inline]
    pub(super) unsafe fn in_range(cells: *const i64, lo: i64, span: u64) -> u8 {
        // SAFETY: NEON is part of aarch64, and the caller says the cells are
        // there.
        unsafe {
            let lo = vdupq_n_s64(lo);
            let span = vdupq_n_u64(span);
            let two = |at: usize| {
                let offsets = vreinterpretq_u64_s64(vsubq_s64(vld1q_s64(cells.add(at)), lo));
                vmovn_u64(vcleq_u64(offsets, span))
            };
            let four = |at: usize| vmovn_u32(vcombine_u32(two(at), two(at + 2)));
            bits(vmovn_u16(vcombine_u16(four(0), four(4))))
        }
    }

    #[inline]
    pub(super) unsafe fn equal(indices: *const u32, index: u32) -> u8 {
        // SAFETY: NEON is part of aarch64, and the caller says the indices are
        // there.
        unsafe {
            let index = vdupq_n_u32(index);
            let four = |at: usize| vmovn_u32(vceqq_u32(vld1q_u32(indices.add(at)), index));
            bits(vmovn_u16(vcombine_u16(four(0), four(4))))
        }
    }
}

/// wasm's 128-bit SIMD: two 64-bit lanes a register, compared signed, so the
/// unsigned comparison flips each top bit first.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod imp {
    use core::arch::wasm32::{
        i32x4_bitmask, i32x4_eq, i32x4_splat, i64x2_bitmask, i64x2_gt, i64x2_splat, i64x2_sub,
        v128, v128_load, v128_not, v128_xor,
    };

    #[inline]
    pub(super) unsafe fn in_range(cells: *const i64, lo: i64, span: u64) -> u8 {
        let top = i64x2_splat(i64::MIN);
        let lo = i64x2_splat(lo);
        let span = v128_xor(i64x2_splat(span.cast_signed()), top);
        let two = |at: usize| {
            // SAFETY: the caller says the cells are there.
            #[expect(clippy::cast_ptr_alignment, reason = "wasm loads needn't be aligned")]
            let keys = unsafe { v128_load(cells.add(at).cast::<v128>()) };
            let offsets = v128_xor(i64x2_sub(keys, lo), top);
            i64x2_bitmask(v128_not(i64x2_gt(offsets, span)))
        };
        two(0) | two(2) << 2 | two(4) << 4 | two(6) << 6
    }

    #[inline]
    pub(super) unsafe fn equal(indices: *const u32, index: u32) -> u8 {
        let index = i32x4_splat(index.cast_signed());
        let four = |at: usize| {
            // SAFETY: the caller says the indices are there.
            #[expect(clippy::cast_ptr_alignment, reason = "wasm loads needn't be aligned")]
            let eight = unsafe { v128_load(indices.add(at).cast::<v128>()) };
            i32x4_bitmask(i32x4_eq(eight, index))
        };
        four(0) | four(4) << 4
    }
}

/// Anywhere else: not called, as `EIGHT_AT_A_TIME` is false, but defined so
/// callers compile everywhere.
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128"),
)))]
mod imp {
    pub(super) unsafe fn in_range(_: *const i64, _: i64, _: u64) -> u8 {
        failed!()
    }

    pub(super) unsafe fn equal(_: *const u32, _: u32) -> u8 {
        failed!()
    }
}
