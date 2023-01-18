use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mj_autograd::*;

pub fn rosenbrock(c: &mut Criterion) {
    c.bench_function("rosenbrock f64", |b| {
        b.iter(|| {
            let mut tape = Tape::new();
            let mut x: Reverse<f64> = Reverse::reversible(3.41, tape.clone());
            let mut y: Reverse<f64> = Reverse::reversible(2.0, tape.clone());

            let mut sgd: AdamW<f64> = AdamW::default(0.001);
            for _ in 0..1000 {
                tape.reset();
                x.reset();
                y.reset();

                let z = (Reverse::auto(1.0) - x.clone()).powi(2)
                    + Reverse::auto(100.0) * (y.clone() - x.powi(2)).powi(2);
                let derivs = black_box(z.derivatives());

                sgd.step(&derivs, &mut [&mut x, &mut y]);
            }
        });
    });
    c.bench_function("rosenbrock f32", |b| {
        b.iter(|| {
            let mut tape = Tape::new();
            let mut x: Reverse<f32> = Reverse::reversible(3.41, tape.clone());
            let mut y: Reverse<f32> = Reverse::reversible(2.0, tape.clone());

            let mut sgd: SimpleGradientDescent<f32> = SimpleGradientDescent::new(0.001);
            for _ in 0..1000 {
                tape.reset();
                x.reset();
                y.reset();

                let z = (Reverse::auto(1.0) - x.clone()).powi(2)
                    + Reverse::auto(100.0) * (y.clone() - x.powi(2)).powi(2);
                let derivs = black_box(z.derivatives());

                sgd.step(&derivs, &mut [&mut x, &mut y]);
            }
        });
    });
}

criterion_group!(benches, rosenbrock);
criterion_main!(benches);
