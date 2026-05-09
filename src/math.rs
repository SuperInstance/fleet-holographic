//! Internal math utilities for holographic storage operations.
//!
//! Provides `no_std`-compatible cosine similarity, MSE, and normalization
//! with zero external dependencies.

/// Compute cosine similarity between two vectors.
///
/// Returns a value in [-1, 1] where 1 = identical direction, 0 = orthogonal,
/// -1 = opposite direction. Handles zero-vector edge cases gracefully.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let dot: f64 = a[..len].iter().zip(b[..len].iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a[..len].iter().map(|x| x * x).sum();
    let norm_b: f64 = b[..len].iter().map(|x| x * x).sum();

    let denom = (norm_a * norm_b).sqrt();
    if denom < f64::EPSILON {
        0.0 // One or both vectors are zero
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// Compute mean squared error between two vectors.
pub fn mean_squared_error(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return f64::MAX;
    }

    let sum_sq: f64 = a[..len]
        .iter()
        .zip(b[..len].iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();

    sum_sq / len as f64
}

/// Normalize a vector to unit length.
pub fn normalize(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < f64::EPSILON {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_same() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10, "Same vectors should have cosine=1");
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10, "Orthogonal vectors should have cosine=0");
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 2.0];
        let b = vec![-1.0, -2.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-10, "Opposite vectors should have cosine=-1");
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Zero vector should return 0");
    }

    #[test]
    fn test_mse_perfect() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let mse = mean_squared_error(&a, &b);
        assert!(mse.abs() < 1e-10, "Identical vectors should have MSE=0");
    }

    #[test]
    fn test_mse_nonzero() {
        let a = vec![1.0, 1.0];
        let b = vec![0.0, 0.0];
        let mse = mean_squared_error(&a, &b);
        assert!((mse - 1.0).abs() < 1e-10, "MSE should be 1.0 for [1,1] vs [0,0]");
    }

    #[test]
    fn test_mse_different_lengths() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let mse = mean_squared_error(&a, &b);
        assert!(mse.is_finite(), "MSE should handle different lengths");
    }

    #[test]
    fn test_mse_empty() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        let mse = mean_squared_error(&a, &b);
        assert_eq!(mse, f64::MAX);
    }

    #[test]
    fn test_normalize_unit() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        let norm: f64 = n.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10, "Normalized vector should have length 1");
    }

    #[test]
    fn test_normalize_zero() {
        let v = vec![0.0, 0.0];
        let n = normalize(&v);
        assert_eq!(n, v, "Zero vector should stay unchanged");
    }

    #[test]
    fn test_cosine_different_lengths() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.0, "Partial overlap should have positive similarity");
    }
}
