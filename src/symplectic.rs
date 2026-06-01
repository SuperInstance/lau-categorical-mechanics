//! Symplectic Category: objects = symplectic manifolds, morphisms = symplectomorphisms
//!
//! A symplectic manifold (M, ω) consists of a smooth manifold M equipped with a closed,
//! non-degenerate 2-form ω. In our discrete/computational setting, we work with finite-
//! dimensional vector spaces equipped with symplectic forms represented by matrices.

use nalgebra::{DMatrix, DVector, ComplexField};
use serde::{Serialize, Deserialize};

/// A symplectic manifold represented as a vector space with a symplectic form.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymplecticManifold {
    /// Dimension of the underlying space (must be even)
    pub dimension: usize,
    /// The symplectic form as an antisymmetric, non-degenerate 2-form matrix
    pub omega: DMatrix<f64>,
    /// Human-readable label
    pub label: String,
}

impl SymplecticManifold {
    /// Create a new symplectic manifold with the canonical symplectic form.
    /// For dimension 2n, the canonical form is [[0, I_n], [-I_n, 0]].
    pub fn canonical(dimension: usize, label: impl Into<String>) -> Self {
        assert!(dimension % 2 == 0, "Symplectic manifold dimension must be even");
        let n = dimension / 2;
        let mut omega = DMatrix::zeros(dimension, dimension);
        for i in 0..n {
            omega[(i, n + i)] = 1.0;
            omega[(n + i, i)] = -1.0;
        }
        Self {
            dimension,
            omega,
            label: label.into(),
        }
    }

    /// Create from a given symplectic form matrix.
    pub fn from_omega(omega: DMatrix<f64>, label: impl Into<String>) -> Option<Self> {
        let dim = omega.nrows();
        if dim != omega.ncols() || dim % 2 != 0 {
            return None;
        }
        // Check antisymmetry: ω^T = -ω
        let antisym = omega.transpose().iter().zip(omega.iter())
            .all(|(a, b)| (a + b).abs() < 1e-10);
        if !antisym {
            return None;
        }
        Some(Self {
            dimension: dim,
            omega,
            label: label.into(),
        })
    }

    /// Check if a point (vector) is in this manifold.
    pub fn contains_point(&self, p: &DVector<f64>) -> bool {
        p.len() == self.dimension
    }

    /// The symplectic pairing of two vectors: ⟨u, v⟩_ω = u^T ω v
    pub fn symplectic_pairing(&self, u: &DVector<f64>, v: &DVector<f64>) -> f64 {
        let result = u.transpose() * &self.omega * v;
        result[(0, 0)]
    }

    /// Dimension of the configuration space (half the full dimension).
    pub fn config_dimension(&self) -> usize {
        self.dimension / 2
    }
}

/// A symplectomorphism: a diffeomorphism that preserves the symplectic form.
/// φ: (M₁, ω₁) → (M₂, ω₂) such that φ*ω₂ = ω₁.
/// Represented as a matrix A where A^T ω₂ A = ω₁.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Symplectomorphism {
    pub source_dim: usize,
    pub target_dim: usize,
    pub matrix: DMatrix<f64>,
    pub label: String,
}

impl Symplectomorphism {
    /// Create a symplectomorphism from a matrix, verifying it preserves the symplectic form.
    pub fn new(
        matrix: DMatrix<f64>,
        source: &SymplecticManifold,
        target: &SymplecticManifold,
        label: impl Into<String>,
    ) -> Option<Self> {
        if matrix.nrows() != target.dimension || matrix.ncols() != source.dimension {
            return None;
        }
        // Check: A^T ω_target A = ω_source
        let result = &matrix.transpose() * &target.omega * &matrix;
        let preserves = result.iter().zip(source.omega.iter())
            .all(|(a, b)| (a - b).abs() < 1e-8);
        if !preserves {
            return None;
        }
        Some(Self {
            source_dim: source.dimension,
            target_dim: target.dimension,
            matrix,
            label: label.into(),
        })
    }

    /// Create without verification (for trusted computations).
    pub fn new_unchecked(
        matrix: DMatrix<f64>,
        source_dim: usize,
        target_dim: usize,
        label: impl Into<String>,
    ) -> Self {
        Self {
            source_dim,
            target_dim,
            matrix,
            label: label.into(),
        }
    }

    /// Identity symplectomorphism on a manifold.
    pub fn identity(manifold: &SymplecticManifold) -> Self {
        Self {
            source_dim: manifold.dimension,
            target_dim: manifold.dimension,
            matrix: DMatrix::identity(manifold.dimension, manifold.dimension),
            label: "id".into(),
        }
    }

    /// Apply this symplectomorphism to a point.
    pub fn apply(&self, point: &DVector<f64>) -> DVector<f64> {
        &self.matrix * point
    }

    /// Compose two symplectomorphisms: (g ∘ f)(x) = g(f(x)).
    pub fn compose(&self, other: &Symplectomorphism) -> Option<Symplectomorphism> {
        if self.source_dim != other.target_dim {
            return None;
        }
        Some(Symplectomorphism::new_unchecked(
            &self.matrix * &other.matrix,
            other.source_dim,
            self.target_dim,
            format!("{}∘{}", self.label, other.label),
        ))
    }

    /// Inverse symplectomorphism.
    pub fn inverse(&self) -> Option<Symplectomorphism> {
        let inv = self.matrix.clone().try_inverse()?;
        Some(Symplectomorphism::new_unchecked(
            inv,
            self.target_dim,
            self.source_dim,
            format!("{}⁻¹", self.label),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_symplectic_form() {
        let m = SymplecticManifold::canonical(4, "T*R²");
        assert_eq!(m.dimension, 4);
        assert_eq!(m.config_dimension(), 2);
        // ω should be antisymmetric
        let antisym = m.omega.transpose().iter().zip(m.omega.iter())
            .all(|(a, b)| (a + b).abs() < 1e-10);
        assert!(antisym);
    }

    #[test]
    fn test_symplectic_pairing() {
        let m = SymplecticManifold::canonical(2, "T*R");
        let q = DVector::from_vec(vec![1.0, 0.0]); // position
        let p = DVector::from_vec(vec![0.0, 1.0]); // momentum
        // ⟨q, p⟩ = 1 for canonical coordinates
        let pairing = m.symplectic_pairing(&q, &p);
        assert!((pairing - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_odd_dimension_rejected() {
        let result = std::panic::catch_unwind(|| {
            SymplecticManifold::canonical(3, "bad");
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_from_omega_rejects_symmetric() {
        let omega = DMatrix::identity(2, 2); // symmetric, not antisymmetric
        assert!(SymplecticManifold::from_omega(omega, "bad").is_none());
    }

    #[test]
    fn test_identity_symplectomorphism() {
        let m = SymplecticManifold::canonical(2, "T*R");
        let id = Symplectomorphism::identity(&m);
        let p = DVector::from_vec(vec![3.0, 4.0]);
        let result = id.apply(&p);
        assert!((result - p).norm() < 1e-10);
    }

    #[test]
    fn test_symplectomorphism_composition() {
        let m = SymplecticManifold::canonical(2, "T*R");
        let id = Symplectomorphism::identity(&m);
        let comp = id.compose(&id).unwrap();
        let p = DVector::from_vec(vec![1.0, 2.0]);
        let result = comp.apply(&p);
        assert!((result - p).norm() < 1e-10);
    }

    #[test]
    fn test_symplectomorphism_inverse() {
        let m = SymplecticManifold::canonical(2, "T*R");
        let id = Symplectomorphism::identity(&m);
        let inv = id.inverse().unwrap();
        let comp = id.compose(&inv).unwrap();
        let p = DVector::from_vec(vec![5.0, -3.0]);
        assert!((comp.apply(&p) - p).norm() < 1e-10);
    }

    #[test]
    fn test_contains_point() {
        let m = SymplecticManifold::canonical(4, "T*R²");
        let p = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(m.contains_point(&p));
        let bad = DVector::from_vec(vec![1.0, 2.0]);
        assert!(!m.contains_point(&bad));
    }

    #[test]
    fn test_composition_dimension_mismatch() {
        let m2 = SymplecticManifold::canonical(2, "2d");
        let m4 = SymplecticManifold::canonical(4, "4d");
        let f = Symplectomorphism::identity(&m2);
        let g = Symplectomorphism::identity(&m4);
        assert!(g.compose(&f).is_none());
    }

    #[test]
    fn test_canonical_form_block_structure() {
        let m = SymplecticManifold::canonical(6, "T*R³");
        let n = 3;
        // Upper-right should be I₃
        for i in 0..n {
            assert!((m.omega[(i, n + i)] - 1.0).abs() < 1e-10);
        }
        // Lower-left should be -I₃
        for i in 0..n {
            assert!((m.omega[(n + i, i)] + 1.0).abs() < 1e-10);
        }
    }
}
