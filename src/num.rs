//! A thin abstraction over the different numerical types we support.

use std::hash::Hash;

use malachite::Rational;
use ordered_float::NotNan;
use ordered_float::OrderedFloat;

/// A wrapper for `f64` that implements `Ord`.
///
/// Unlike the more principled wrappers in the `ordered_float` crate, this
/// one just panics when comparing NaNs -- it doesn't order them, nor does
/// it guard against them on construction. This makes things substantially
/// faster: I measured a 20% improvement to some benchmarks by switching
/// from `OrderedFloat` to `CheapOrderedFloat`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct CheapOrderedFloat(f64);

impl std::ops::Add<CheapOrderedFloat> for CheapOrderedFloat {
    type Output = Self;

    fn add(self, rhs: CheapOrderedFloat) -> Self::Output {
        CheapOrderedFloat(self.0 + rhs.0)
    }
}

impl<'a> std::ops::Add<&'a CheapOrderedFloat> for CheapOrderedFloat {
    type Output = Self;

    fn add(self, rhs: &'a CheapOrderedFloat) -> Self::Output {
        CheapOrderedFloat(self.0 + rhs.0)
    }
}

impl std::ops::Sub<CheapOrderedFloat> for CheapOrderedFloat {
    type Output = Self;

    fn sub(self, rhs: CheapOrderedFloat) -> Self::Output {
        CheapOrderedFloat(self.0 - rhs.0)
    }
}

impl<'a> std::ops::Sub<&'a CheapOrderedFloat> for CheapOrderedFloat {
    type Output = Self;

    fn sub(self, rhs: &'a CheapOrderedFloat) -> Self::Output {
        CheapOrderedFloat(self.0 - rhs.0)
    }
}

impl std::ops::Mul<CheapOrderedFloat> for CheapOrderedFloat {
    type Output = Self;

    fn mul(self, rhs: CheapOrderedFloat) -> Self::Output {
        CheapOrderedFloat(self.0 * rhs.0)
    }
}

impl<'a> std::ops::Mul<&'a CheapOrderedFloat> for CheapOrderedFloat {
    type Output = Self;

    fn mul(self, rhs: &'a CheapOrderedFloat) -> Self::Output {
        CheapOrderedFloat(self.0 * rhs.0)
    }
}

impl std::ops::Div<CheapOrderedFloat> for CheapOrderedFloat {
    type Output = Self;

    fn div(self, rhs: CheapOrderedFloat) -> Self::Output {
        CheapOrderedFloat(self.0 / rhs.0)
    }
}

impl<'a> std::ops::Div<&'a CheapOrderedFloat> for CheapOrderedFloat {
    type Output = Self;

    fn div(self, rhs: &'a CheapOrderedFloat) -> Self::Output {
        CheapOrderedFloat(self.0 / rhs.0)
    }
}

impl std::ops::Neg for CheapOrderedFloat {
    type Output = Self;

    fn neg(self) -> Self::Output {
        CheapOrderedFloat(-self.0)
    }
}

impl Hash for CheapOrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state)
    }
}

impl CheapOrderedFloat {
    /// Retrieve the inner `f64`.
    pub fn into_inner(self) -> f64 {
        self.0
    }
}

// Now comes the fishy stuff.
impl Eq for CheapOrderedFloat {}

impl PartialOrd for CheapOrderedFloat {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CheapOrderedFloat {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.0 < other.0 {
            std::cmp::Ordering::Less
        } else if self.0 > other.0 {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

impl Float for CheapOrderedFloat {
    fn from_f32(x: f32) -> Self {
        Self(x.into())
    }

    fn to_exact(&self) -> Rational {
        self.0.try_into().unwrap()
    }

    #[inline(always)]
    fn abs(self) -> Self {
        CheapOrderedFloat(self.0.abs())
    }

    fn is_subnormal(&self) -> bool {
        self.0.is_subnormal()
    }
}

impl From<f64> for CheapOrderedFloat {
    fn from(value: f64) -> Self {
        CheapOrderedFloat(value)
    }
}

/// A trait for abstracting over the properties we need from numerical types.
///
/// This is implemented for `NotNan<f64>`, `NotNan<f32>`, and `malachite::Rational`.
pub trait Float:
    Sized
    + std::ops::Add<Self, Output = Self>
    + std::ops::Sub<Self, Output = Self>
    + std::ops::Mul<Self, Output = Self>
    + std::ops::Div<Self, Output = Self>
    + std::ops::Neg<Output = Self>
    + for<'a> std::ops::Add<&'a Self, Output = Self>
    + for<'a> std::ops::Sub<&'a Self, Output = Self>
    + for<'a> std::ops::Mul<&'a Self, Output = Self>
    + for<'a> std::ops::Div<&'a Self, Output = Self>
    + Clone
    + std::fmt::Debug
    + Ord
    + Eq
    + Hash
    + 'static
{
    /// Convert from a `f32`. This is allowed to panic if `x` is infinite or NaN.
    fn from_f32(x: f32) -> Self;

    /// Convert this number to a rational, for exact computation.
    fn to_exact(&self) -> Rational;

    /// The absolute value.
    fn abs(self) -> Self;

    /// Is this a subnormal number?
    fn is_subnormal(&self) -> bool;
}

impl Float for Rational {
    fn from_f32(x: f32) -> Self {
        Rational::try_from(x).unwrap()
    }

    fn to_exact(&self) -> Rational {
        self.clone()
    }

    fn abs(self) -> Self {
        <Rational as malachite::num::arithmetic::traits::Abs>::abs(self)
    }

    fn is_subnormal(&self) -> bool {
        false
    }
}

impl Float for NotNan<f32> {
    fn from_f32(x: f32) -> Self {
        NotNan::try_from(x).unwrap()
    }

    fn to_exact(&self) -> Rational {
        self.into_inner().try_into().unwrap()
    }

    fn abs(self) -> Self {
        self.into_inner().abs().try_into().unwrap()
    }

    fn is_subnormal(&self) -> bool {
        self.into_inner().is_subnormal()
    }
}

impl Float for NotNan<f64> {
    fn from_f32(x: f32) -> Self {
        NotNan::try_from(f64::from(x)).unwrap()
    }

    fn to_exact(&self) -> Rational {
        self.into_inner().try_into().unwrap()
    }

    fn abs(self) -> Self {
        self.into_inner().abs().try_into().unwrap()
    }

    fn is_subnormal(&self) -> bool {
        self.into_inner().is_subnormal()
    }
}

impl Float for OrderedFloat<f64> {
    fn from_f32(x: f32) -> Self {
        OrderedFloat::from(f64::from(x))
    }

    fn to_exact(&self) -> Rational {
        self.into_inner().try_into().unwrap()
    }

    fn abs(self) -> Self {
        self.into_inner().abs().into()
    }

    fn is_subnormal(&self) -> bool {
        self.into_inner().is_subnormal()
    }
}

impl Float for OrderedFloat<f32> {
    fn from_f32(x: f32) -> Self {
        OrderedFloat::from(x)
    }

    fn to_exact(&self) -> Rational {
        self.into_inner().try_into().unwrap()
    }

    fn abs(self) -> Self {
        self.into_inner().abs().into()
    }

    fn is_subnormal(&self) -> bool {
        self.into_inner().is_subnormal()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use proptest::prelude::*;

    // Kind of like Arbitrary, but
    // - it's a local trait, so we can impl it for whatever we want, and
    // - it only returns "reasonable" values.
    pub trait Reasonable {
        type Strategy: Strategy<Value = Self>;
        fn reasonable() -> Self::Strategy;
    }

    impl<S: Reasonable, T: Reasonable> Reasonable for (S, T) {
        type Strategy = (S::Strategy, T::Strategy);

        fn reasonable() -> Self::Strategy {
            (S::reasonable(), T::reasonable())
        }
    }

    impl Reasonable for NotNan<f32> {
        type Strategy = BoxedStrategy<NotNan<f32>>;

        fn reasonable() -> Self::Strategy {
            (-1e6f32..1e6).prop_map(|x| NotNan::new(x).unwrap()).boxed()
        }
    }

    impl Reasonable for NotNan<f64> {
        type Strategy = BoxedStrategy<NotNan<f64>>;

        fn reasonable() -> Self::Strategy {
            (-1e6..1e6).prop_map(|x| NotNan::new(x).unwrap()).boxed()
        }
    }

    impl Reasonable for Rational {
        type Strategy = BoxedStrategy<Rational>;

        fn reasonable() -> Self::Strategy {
            (-1e6..1e6)
                .prop_map(|x| Rational::try_from(x).unwrap())
                .boxed()
        }
    }
}
