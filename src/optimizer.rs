/// Optimizer algorithms.
use crate::autograd::{Derivatives, Reverse};
use num::traits::cast::NumCast;
use num::traits::real::Real;
use num::traits::{One, Zero};

pub trait Optimizer<T> {
    fn step(&mut self, derivatives: &Derivatives<T>, params: &mut [&mut Reverse<T>]);
}

/// Simple gradient descent. Simply moves weights in the direction of the negative gradient at a
/// fixed learning rate.
pub struct SimpleGradientDescent<T> {
    learning_rate: T,
}

impl<T: Clone + std::ops::Neg<Output = T> + std::ops::Mul<Output = T>> SimpleGradientDescent<T> {
    pub fn new(learning_rate: T) -> Self {
        Self { learning_rate }
    }
}

impl<
        T: Zero
            + One
            + Clone
            + std::ops::Sub<Output = T>
            + std::ops::Neg<Output = T>
            + std::ops::Mul<Output = T>,
    > Optimizer<T> for SimpleGradientDescent<T>
{
    fn step(&mut self, derivatives: &Derivatives<T>, params: &mut [&mut Reverse<T>]) {
        for p in params.iter_mut() {
            let p: &mut Reverse<T> = *p;
            *p = p.clone()
                - Reverse::auto(derivatives[p.clone()].clone() * self.learning_rate.clone());
        }
    }
}

/// Adam(W) optimizer.
pub struct AdamW<T> {
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    t: T,
    weight_decay: T,

    first_momentum: Vec<T>,
    second_momentum: Vec<T>,
}

impl<T: NumCast + One + Zero + Clone + std::ops::Neg<Output = T> + std::ops::Mul<Output = T>>
    AdamW<T>
{
    pub fn default(learning_rate: f64) -> Self {
        Self {
            learning_rate: T::from(learning_rate).unwrap(),
            beta1: T::from(0.9).unwrap(),
            beta2: T::from(0.999).unwrap(),
            epsilon: T::from(1e-8).unwrap(),
            t: T::one(),
            weight_decay: T::from(0.01).unwrap(),

            first_momentum: Vec::new(),
            second_momentum: Vec::new(),
        }
    }
}

impl<T: Real + Clone + One + std::ops::Neg<Output = T> + std::ops::Mul<Output = T>> Optimizer<T>
    for AdamW<T>
{
    fn step(&mut self, derivatives: &Derivatives<T>, params: &mut [&mut Reverse<T>]) {
        let nparams = params.len();
        let derivatives: &[T] = &derivatives.derivatives;
        let nderivatives = derivatives.len();

        let mut param_indices: Vec<usize> = Vec::with_capacity(nparams);

        // Apply weight decay
        if !self.weight_decay.is_zero() {
            for idx in 0..nparams {
                param_indices.push(params[idx].index);
                *params[idx] = params[idx].clone()
                    - Reverse::auto(self.learning_rate * self.weight_decay) * params[idx].clone();
            }
        }

        if self.first_momentum.is_empty() {
            self.first_momentum = vec![T::zero(); nderivatives];
        }

        if self.second_momentum.is_empty() {
            self.second_momentum = vec![T::zero(); nderivatives];
        }

        for idx in 0..nderivatives {
            self.first_momentum[idx] = self.beta1.clone() * self.first_momentum[idx].clone()
                + (T::one() - self.beta1.clone()) * derivatives[idx].clone();
        }

        for idx in 0..nderivatives {
            self.second_momentum[idx] = self.beta2.clone() * self.second_momentum[idx].clone()
                + (T::one() - self.beta2.clone())
                    * derivatives[idx].clone()
                    * derivatives[idx].clone();
        }

        let mut first_momentum_hat: Vec<T> = Vec::with_capacity(nderivatives);
        let mut second_momentum_hat: Vec<T> = Vec::with_capacity(nderivatives);

        for idx in 0..nderivatives {
            first_momentum_hat.push(
                self.first_momentum[idx].clone()
                    / (T::one() - self.beta1.clone().powi(self.t.clone().to_i32().unwrap())),
            );
            second_momentum_hat.push(
                self.second_momentum[idx].clone()
                    / (T::one() - self.beta2.clone().powi(self.t.clone().to_i32().unwrap())),
            );
        }

        for idx in 0..nparams {
            *params[idx] = params[idx].clone()
                - Reverse::auto(
                    self.learning_rate.clone() * first_momentum_hat[param_indices[idx]].clone()
                        / (second_momentum_hat[param_indices[idx]].clone().sqrt()
                            + self.epsilon.clone()),
                );
        }

        self.t = self.t.clone() + T::one();
    }
}
