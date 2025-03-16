#![no_main]

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;
use linesweeper::{arbitrary::quadratic, curve::Quadratic};

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let q: Quadratic = quadratic(1e8, &mut u).unwrap();
    let signs = q.signs();

    assert!(!signs.smaller_root.is_nan());
    assert!(!signs.bigger_root.is_nan());

    // How much relative accuracy do we expect?
    let accuracy = 1e-11;

    if signs.smaller_root.powi(2).is_finite() {
        // The smaller root should be approximately a root. How approximate is
        // "approximately"? It depends on the relative error, and the biggest
        // number we're computing with. (Note that we square the root when
        // evaluating, so it gets squared for computing relative error also.)
        let threshold = accuracy * q.max_coeff().max(signs.smaller_root.powi(2));
        assert!(q.eval(signs.smaller_root).abs() <= threshold);

        // The quadratic should agree with the initial sign up to the smaller root.
        if signs.initial_sign.signum() == 0.0 {
            assert!(q.eval(signs.smaller_root - 1.0).abs() <= threshold);
        } else {
            assert_eq!(
                q.eval(signs.smaller_root - 1.0).signum(),
                signs.initial_sign.signum()
            );
        }
    }

    if signs.bigger_root.is_finite() {
        // The bigger root should also be a approximately a root.
        let threshold = accuracy * q.max_coeff().max(signs.bigger_root.powi(2));
        assert!(q.eval(signs.bigger_root).abs() <= threshold);

        // The quadratic should agree with the initial sign after the bigger root.
        if signs.initial_sign.signum() == 0.0 {
            assert!(q.eval(signs.bigger_root + 1.0).abs() <= threshold);
        } else {
            assert_eq!(
                q.eval(signs.bigger_root + 1.0).signum(),
                signs.initial_sign.signum()
            );

            // The quadratic should have the opposite sign between the roots, at least if it's
            // measurably non-zero. We don't check the finiteness of `smaller_root` here because
            // if it's infinite then the threshold will be inifinity.
            let threshold = threshold.max(accuracy * q.max_coeff().max(signs.smaller_root.powi(2)));
            let mid_val = q.eval((signs.bigger_root + signs.smaller_root) / 2.0);
            if mid_val.abs() > threshold {
                assert_ne!(mid_val.signum(), signs.initial_sign.signum());
            }
        }
    }
});
