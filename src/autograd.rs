// To give credit, this code is an adaptation of easy_ml code:
// https://docs.rs/easy-ml/latest/src/easy_ml/differentiation.rs.html
//
// I got mad that I wasted hours on lifetime shenanigans and wanted a version where the variables
// for which we are computing gradient values have no &-references inside them. No lifetimes. No
// lifetimes is no pain. I hope. Instead I use Rc<>s.

use num::traits::{One, Zero};
use std::cell::RefCell;
use std::rc::Rc;

/// Tape, aka "Wengert list". It stores operations involving automatic differentiation.
#[derive(Clone, Debug)]
pub struct Tape<T> {
    ops: Rc<RefCell<Vec<Op<T>>>>,
}

impl<T: Clone> Tape<T> {
    pub fn new() -> Self {
        Self {
            ops: Rc::new(RefCell::new(vec![])),
        }
    }

    pub fn reset(&mut self) {
        self.ops.borrow_mut().clear();
    }

    fn add_reset(&self, v1: T) -> usize {
        let mut ops = self.ops.borrow_mut();
        let new_idx = ops.len();
        ops.push(Op {
            left: new_idx,
            right: new_idx,
            dleft: v1.clone(),
            dright: v1,
        });
        new_idx
    }

    fn add_op(&self, op: Op<T>) -> usize {
        let mut ops = self.ops.borrow_mut();
        let new_idx = ops.len();
        ops.push(op);
        new_idx
    }
}

#[derive(Clone, Debug)]
struct Op<T> {
    left: usize,
    right: usize,
    dleft: T,
    dright: T,
}

impl<T: Clone + Zero> Op<T> {
    #[inline]
    fn unary(left: usize, dleft: T) -> Self {
        Op {
            left,
            right: left,
            dleft,
            dright: T::zero(),
        }
    }

    #[inline]
    fn bin(left: usize, dleft: T, right: usize, dright: T) -> Self {
        Op {
            left,
            right,
            dleft,
            dright,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Reverse<T> {
    pub(crate) value: T,
    pub(crate) tape: Option<Tape<T>>,
    pub(crate) index: usize,
}

impl<T: PartialEq> PartialEq for Reverse<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: Eq> Eq for Reverse<T> {}

impl<T: PartialOrd> PartialOrd for Reverse<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: Ord> Ord for Reverse<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<T: Clone + One + Zero> Zero for Reverse<T> {
    #[inline]
    fn zero() -> Self {
        Self::auto(T::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
}

impl<T: Clone + PartialEq + One + Zero> One for Reverse<T> {
    #[inline]
    fn one() -> Self {
        Self::auto(T::one())
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.value.is_one()
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Derivatives<T> {
    pub(crate) derivatives: Vec<T>,
}

impl<T> Derivatives<T> {
    pub(crate) fn empty() -> Self {
        Derivatives {
            derivatives: Vec::new(),
        }
    }
}

impl<T> std::ops::Index<Reverse<T>> for Derivatives<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: Reverse<T>) -> &Self::Output {
        &self.derivatives[index.index]
    }
}

impl<T> std::ops::Index<&Reverse<T>> for Derivatives<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: &Reverse<T>) -> &Self::Output {
        &self.derivatives[index.index]
    }
}

impl<T> std::ops::Index<&mut Reverse<T>> for Derivatives<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: &mut Reverse<T>) -> &Self::Output {
        &self.derivatives[index.index]
    }
}

impl<T: Clone + Zero + One> Reverse<T> {
    #[inline]
    pub fn value(&self) -> &T {
        &self.value
    }

    #[inline]
    pub fn reversible(v: T, tape: Tape<T>) -> Self {
        let index = tape.add_reset(T::zero());
        Reverse {
            value: v,
            tape: Some(tape),
            index,
        }
    }

    #[inline]
    pub fn auto(v: T) -> Self {
        Reverse {
            value: v,
            tape: None,
            index: 0,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        match self.tape {
            None => (),
            Some(ref mut tape) => {
                self.index = tape.add_reset(T::zero());
            }
        }
    }

    #[inline]
    pub fn unary_op<F, F2>(&self, eval: F, deriv: F2) -> Self
    where
        F: Fn(T) -> T,
        F2: Fn(T) -> T,
    {
        let value = eval(self.value.clone());
        match &self.tape {
            None => Reverse {
                value,
                tape: None,
                index: 0,
            },
            Some(ref tape) => Reverse {
                value,
                tape: Some(tape.clone()),
                index: tape.add_op(Op::unary(self.index, deriv(self.value.clone()))),
            },
        }
    }

    #[inline]
    pub fn bin_op<F, F2, F3>(
        &self,
        other: &Reverse<T>,
        eval: F,
        deriv_left: F2,
        deriv_right: F3,
    ) -> Self
    where
        F: Fn(T, T) -> T,
        F2: Fn(T, T) -> T,
        F3: Fn(T, T) -> T,
    {
        let value = eval(self.value.clone(), other.value.clone());
        match (&self.tape, &other.tape) {
            (None, None) => Reverse {
                value,
                tape: None,
                index: 0,
            },
            (Some(ref tape_left), None) => Reverse {
                value,
                tape: Some(tape_left.clone()),
                index: tape_left.add_op(Op::unary(
                    self.index,
                    deriv_left(self.value.clone(), other.value.clone()),
                )),
            },
            (None, Some(ref tape_right)) => Reverse {
                value,
                tape: Some(tape_right.clone()),
                index: tape_right.add_op(Op::unary(
                    other.index,
                    deriv_right(self.value.clone(), other.value.clone()),
                )),
            },
            (Some(ref tape_left), Some(ref _tape_right)) => Reverse {
                value,
                tape: Some(tape_left.clone()),
                index: tape_left.add_op(Op::bin(
                    self.index,
                    deriv_left(self.value.clone(), other.value.clone()),
                    other.index,
                    deriv_right(self.value.clone(), other.value.clone()),
                )),
            },
        }
    }

    pub fn derivatives(&self) -> Derivatives<T> {
        if self.tape.is_none() {
            return Derivatives::empty();
        }
        let tape = self.tape.as_ref().unwrap();
        let ops = tape.ops.borrow();

        let mut derivatives: Vec<T> = vec![T::zero(); ops.len()];
        derivatives[self.index] = T::one();

        for idx in (0..ops.len()).rev() {
            derivatives[ops[idx].left] = derivatives[ops[idx].left].clone()
                + derivatives[idx].clone() * ops[idx].dleft.clone();
            derivatives[ops[idx].right] = derivatives[ops[idx].right].clone()
                + derivatives[idx].clone() * ops[idx].dright.clone();
        }

        Derivatives { derivatives }
    }
}

impl Reverse<f32> {
    // TODO: no rigorous testing has been done on any of these
    #[inline]
    pub fn ln(&self) -> Self {
        self.unary_op(|v| v.ln(), |v| v.recip())
    }

    #[inline]
    pub fn abs(&self) -> Self {
        self.unary_op(|v| v.abs(), |v| v.signum())
    }

    #[inline]
    pub fn signum(&self) -> Self {
        self.unary_op(|v| v.signum(), |_| 0.0)
    }

    #[inline]
    pub fn exp(&self) -> Self {
        self.unary_op(f32::exp, f32::exp)
    }

    #[inline]
    pub fn sqrt(&self) -> Self {
        self.unary_op(f32::sqrt, |v| 0.5 * v.sqrt().recip())
    }

    #[inline]
    pub fn powi(&self, n: i32) -> Self {
        self.unary_op(|v| v.powi(n), |v| (n as f32) * v.clone().powi(n - 1))
    }
}

impl Reverse<f64> {
    #[inline]
    pub fn ln(&self) -> Self {
        self.unary_op(|v| v.ln(), |v| v.recip())
    }

    #[inline]
    pub fn abs(&self) -> Self {
        self.unary_op(|v| v.abs(), |v| v.signum())
    }

    #[inline]
    pub fn signum(&self) -> Self {
        self.unary_op(|v| v.signum(), |_| 0.0)
    }

    #[inline]
    pub fn exp(&self) -> Self {
        self.unary_op(f64::exp, f64::exp)
    }

    #[inline]
    pub fn sqrt(&self) -> Self {
        self.unary_op(f64::sqrt, |v| 0.5 * v.sqrt().recip())
    }

    #[inline]
    pub fn powi(&self, n: i32) -> Self {
        self.unary_op(|v| v.powi(n), |v| (n as f64) * v.clone().powi(n - 1))
    }
}

impl<T: Clone + One + Zero + std::ops::Add> std::ops::Add for Reverse<T> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        self.bin_op(&other, |a, b| a + b, |_, _| T::one(), |_, _| T::one())
    }
}

impl<T: Clone + One + Zero + std::ops::Add> std::ops::Add for &Reverse<T> {
    type Output = Reverse<T>;

    #[inline]
    fn add(self, other: Self) -> Self::Output {
        self.bin_op(&other, |a, b| a + b, |_, _| T::one(), |_, _| T::one())
    }
}

impl<T: Clone + One + Zero + std::ops::AddAssign> std::ops::AddAssign for Reverse<T> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = self.bin_op(&other, |a, b| a + b, |_, _| T::one(), |_, _| T::one())
    }
}

impl<
        T: Clone + One + Zero + std::ops::Neg + std::ops::Neg<Output = T> + std::ops::Sub<Output = T>,
    > std::ops::Sub for Reverse<T>
{
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        self.bin_op(&other, |a, b| a - b, |_, _| T::one(), |_, _| -T::one())
    }
}

impl<
        T: Clone + One + Zero + std::ops::Neg + std::ops::Neg<Output = T> + std::ops::Sub<Output = T>,
    > std::ops::Sub for &Reverse<T>
{
    type Output = Reverse<T>;

    #[inline]
    fn sub(self, other: Self) -> Self::Output {
        self.bin_op(&other, |a, b| a - b, |_, _| T::one(), |_, _| -T::one())
    }
}

impl<
        T: Clone + One + Zero + std::ops::Neg + std::ops::Neg<Output = T> + std::ops::Sub<Output = T>,
    > std::ops::SubAssign for Reverse<T>
{
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = self.bin_op(&other, |a, b| a - b, |_, _| T::one(), |_, _| -T::one())
    }
}

impl<T: Clone + One + Zero + std::ops::Mul> std::ops::Mul for Reverse<T> {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        self.bin_op(&other, |a, b| a * b, |_a, b| b, |a, _b| a)
    }
}

impl<T: Clone + One + Zero + std::ops::Mul> std::ops::Mul for &Reverse<T> {
    type Output = Reverse<T>;

    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        self.bin_op(&other, |a, b| a * b, |_a, b| b, |a, _b| a)
    }
}

impl<T: Clone + One + Zero + std::ops::Mul> std::ops::MulAssign for Reverse<T> {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = self.bin_op(&other, |a, b| a * b, |_a, b| b, |a, _b| a)
    }
}

impl<T: Clone + One + Zero + std::ops::Neg + std::ops::Neg<Output = T>> std::ops::Neg
    for Reverse<T>
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        self.unary_op(|a| -a, |_| -T::one())
    }
}

impl<T: Clone + One + Zero + std::ops::Neg + std::ops::Neg<Output = T>> std::ops::Neg
    for &Reverse<T>
{
    type Output = Reverse<T>;

    #[inline]
    fn neg(self) -> Self::Output {
        self.unary_op(|a| -a, |_| -T::one())
    }
}

impl<
        T: Clone
            + One
            + Zero
            + std::ops::Neg
            + std::ops::Neg<Output = T>
            + std::ops::Div
            + std::ops::Div<Output = T>,
    > std::ops::Div for Reverse<T>
{
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self {
        self.bin_op(
            &other,
            |a, b| a / b,
            |_a, b| T::one() / b,
            |a, b| -a / (b.clone() * b),
        )
    }
}

impl<
        T: Clone
            + One
            + Zero
            + std::ops::Neg
            + std::ops::Neg<Output = T>
            + std::ops::Div
            + std::ops::Div<Output = T>,
    > std::ops::Div for &Reverse<T>
{
    type Output = Reverse<T>;

    #[inline]
    fn div(self, other: Self) -> Self::Output {
        self.bin_op(
            &other,
            |a, b| a / b,
            |_a, b| T::one() / b,
            |a, b| -a / (b.clone() * b),
        )
    }
}

impl<
        T: Clone
            + One
            + Zero
            + std::ops::Neg
            + std::ops::Neg<Output = T>
            + std::ops::Div
            + std::ops::Div<Output = T>,
    > std::ops::DivAssign for Reverse<T>
{
    #[inline]
    fn div_assign(&mut self, other: Self) {
        *self = self.bin_op(
            &other,
            |a, b| a / b,
            |_a, b| T::one() / b,
            |a, b| -a / (b.clone() * b),
        )
    }
}
