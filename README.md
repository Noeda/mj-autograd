mj-autograd
-----------

Small library for doing reverse-mode automatic differentation in Rust.

The implementation strategy uses a tape of operations and a special
`Reverse<T>` type (where `T` is usually something like f32 or f64) to keep
track of operations.

Compared to `easy_ml` and `reverse` crates, this crate is based on using
reference counters instead of juggling lifetimes to make code cleaner (and is
the reason I wrote this instead of using those crates because I was going
crazy with a lifetime bug I couldn't figure out).

Also includes a simple `AdamW` optimizer.

Note: this is something I'm using for personal private projects and I have no
intention of making this production ready. However this library is quite small
so if you are interested in how to implement reverse autodiff, you could look
at the source code.

# Crappy example

```rust
use mj_autograd::*;

fn main() {
    // Gradient descent for the Rosenbrock function.

    let mut tape = Tape::new();
    let mut x: Reverse<f64> = Reverse::reversible(3.41, tape.clone());
    let mut y: Reverse<f64> = Reverse::reversible(2.0, tape.clone());

    let mut sgd = AdamW::default(0.001);

    loop {
        tape.reset();
        x.reset();
        y.reset();

        let z = (Reverse::auto(1.0) - x.clone()).powi(2)
            + Reverse::auto(100.0) * (y.clone() - x.powi(2)).powi(2);
        let derivs = z.derivatives();

        println!("x: {}, y: {}, z: {}", derivs[&x], derivs[&y], z.value());

        sgd.step(&derivs, &mut [&mut x, &mut y]);
        println!("{} {}", x.value(), y.value());
    }
}
```
