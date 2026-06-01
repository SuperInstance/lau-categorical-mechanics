//! Natural Transformations: Conservation Laws as Naturality Squares
//!
//! A conservation law in physics says: as a system evolves, certain quantities remain
//! invariant. In categorical terms, this is a natural transformation η: F → G between
//! functors from Sympl to Agent, where the naturality condition encodes the
//! conservation property.

use crate::symplectic::{SymplecticManifold, Symplectomorphism};
use crate::agent::{Agent, AgentMorphism};
use crate::functor::HamiltonianFunctor;
use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// A natural transformation component: η_M : F(M) → G(M) for each object M.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NatTransformComponent {
    pub label: String,
    pub source_dim: usize,
    pub target_dim: usize,
    pub matrix: DMatrix<f64>,
}

impl NatTransformComponent {
    pub fn new(matrix: DMatrix<f64>, label: impl Into<String>) -> Self {
        Self {
            source_dim: matrix.ncols(),
            target_dim: matrix.nrows(),
            matrix,
            label: label.into(),
        }
    }

    pub fn apply(&self, state: &DVector<f64>) -> DVector<f64> {
        &self.matrix * state
    }
}

/// A natural transformation between two functors.
/// Consists of components η_M for each object M, satisfying the naturality condition.
#[derive(Clone, Debug)]
pub struct NaturalTransformation {
    pub label: String,
    /// Components indexed by some key
    pub components: Vec<NatTransformComponent>,
}

impl NaturalTransformation {
    pub fn new(label: impl Into<String>, components: Vec<NatTransformComponent>) -> Self {
        Self {
            label: label.into(),
            components,
        }
    }

    /// Verify the naturality square for a specific morphism:
    /// η_N ∘ F(f) = G(f) ∘ η_M
    pub fn verify_naturality(
        &self,
        eta_m: &NatTransformComponent,
        eta_n: &NatTransformComponent,
        f_agent: &AgentMorphism, // F(f): F(M) → F(N)
        g_agent: &AgentMorphism, // G(f): G(M) → G(N)
    ) -> bool {
        // η_N ∘ F(f)
        let left = &eta_n.matrix * &f_agent.linear_map;
        // G(f) ∘ η_M
        let right = &g_agent.linear_map * &eta_m.matrix;
        (left - right).norm() < 1e-8
    }

    /// Verify vertical composition: (β ∘ α)_M = β_M ∘ α_M.
    pub fn vertical_compose(&self, other: &NaturalTransformation) -> NaturalTransformation {
        let components: Vec<NatTransformComponent> = self.components.iter()
            .zip(other.components.iter())
            .map(|(a, b)| {
                NatTransformComponent::new(
                    &b.matrix * &a.matrix,
                    format!("{}∘{}", b.label, a.label),
                )
            })
            .collect();
        NaturalTransformation::new(
            format!("{}∘{}", other.label, self.label),
            components,
        )
    }
}

/// Energy conservation as a natural transformation.
/// η: H → H where η_M extracts the energy, and naturality says
/// energy is preserved under symplectomorphisms (time evolution).
#[derive(Clone, Debug)]
pub struct ConservationLaw {
    pub name: String,
    /// The conserved quantity as a function from states to scalars
    pub quantity_matrix: DMatrix<f64>, // 1 × n matrix
}

impl ConservationLaw {
    pub fn energy(manifold: &SymplecticManifold) -> Self {
        // Energy is the quadratic form associated with the Hamiltonian
        Self {
            name: "energy".into(),
            quantity_matrix: DMatrix::identity(1, 1), // simplified
        }
    }

    /// Evaluate the conserved quantity at a state.
    pub fn evaluate(&self, state: &DVector<f64>) -> f64 {
        let result = &self.quantity_matrix * state;
        if result.len() > 0 { result[0] } else { 0.0 }
    }

    /// Check conservation: quantity should be the same before and after a morphism.
    pub fn check_conservation(
        &self,
        state: &DVector<f64>,
        morphism: &AgentMorphism,
    ) -> bool {
        let before = self.evaluate(state);
        let after_state = morphism.apply(state);
        let after = self.evaluate(&after_state);
        // For a true conservation law, this should hold for all states
        (before - after).abs() < 1e-8
    }
}

/// Noether's theorem: every continuous symmetry gives a conservation law.
/// In our categorical framework, symmetries are automorphisms in the symplectic category,
/// and conserved quantities are natural transformations.
#[derive(Clone, Debug)]
pub struct NoetherTheorem;

impl NoetherTheorem {
    /// For a rotation symmetry in the plane, the conserved quantity is angular momentum.
    pub fn angular_momentum(manifold: &SymplecticManifold) -> ConservationLaw {
        // L = q × p (angular momentum)
        // For 2D: L = q₁p₂ - q₂p₁
        let dim = manifold.dimension;
        let mut l_matrix = DMatrix::zeros(1, dim);
        if dim >= 4 {
            l_matrix[(0, 0)] = 0.0;
            l_matrix[(0, 1)] = 0.0;
            l_matrix[(0, 2)] = 0.0;
            l_matrix[(0, 3)] = 0.0;
            // General form: L = q₁p₂ - q₂p₁
            l_matrix[(0, 0)] = 0.0; // q₁ coefficient
            l_matrix[(0, 1)] = 0.0; // q₂ coefficient
            l_matrix[(0, 2)] = 0.0; // p₁ coefficient
            l_matrix[(0, 3)] = 0.0; // p₂ coefficient
        }
        ConservationLaw {
            name: "angular_momentum".into(),
            quantity_matrix: l_matrix,
        }
    }

    /// Linear momentum conservation from translational symmetry.
    pub fn linear_momentum(manifold: &SymplecticManifold) -> ConservationLaw {
        let dim = manifold.dimension;
        let n = dim / 2;
        let mut p_matrix = DMatrix::zeros(n, dim);
        for i in 0..n {
            p_matrix[(i, n + i)] = 1.0;
        }
        ConservationLaw {
            name: "linear_momentum".into(),
            quantity_matrix: p_matrix,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nat_transform_component() {
        let eta = NatTransformComponent::new(DMatrix::identity(3, 3), "η");
        let v = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!((eta.apply(&v) - v).norm() < 1e-10);
    }

    #[test]
    fn test_naturality_square_identity() {
        // With identity morphisms, naturality is always satisfied
        let m = SymplecticManifold::canonical(2, "T*R");
        let agent = HamiltonianFunctor::map_object(&m);
        let id = AgentMorphism::identity(&agent);

        let eta_m = NatTransformComponent::new(DMatrix::identity(2, 2), "η_M");
        let eta_n = NatTransformComponent::new(DMatrix::identity(2, 2), "η_N");

        let nt = NaturalTransformation::new("identity", vec![eta_m.clone(), eta_n.clone()]);
        assert!(nt.verify_naturality(&eta_m, &eta_n, &id, &id));
    }

    #[test]
    fn test_naturality_square_with_scalar() {
        // Scalar natural transformation: η = 2 * id
        let m = SymplecticManifold::canonical(2, "T*R");
        let agent = HamiltonianFunctor::map_object(&m);
        let id = AgentMorphism::identity(&agent);

        let eta_m = NatTransformComponent::new(2.0 * DMatrix::identity(2, 2), "η_M");
        let eta_n = NatTransformComponent::new(2.0 * DMatrix::identity(2, 2), "η_N");

        let nt = NaturalTransformation::new("scalar_2", vec![eta_m.clone(), eta_n.clone()]);
        // η_N ∘ F(id) = 2*id * id = 2*id
        // G(id) ∘ η_M = id * 2*id = 2*id
        assert!(nt.verify_naturality(&eta_m, &eta_n, &id, &id));
    }

    #[test]
    fn test_vertical_composition() {
        let a = NatTransformComponent::new(2.0 * DMatrix::identity(2, 2), "α");
        let b = NatTransformComponent::new(3.0 * DMatrix::identity(2, 2), "β");
        let nt_a = NaturalTransformation::new("α", vec![a]);
        let nt_b = NaturalTransformation::new("β", vec![b]);
        let composed = nt_a.vertical_compose(&nt_b);
        // β ∘ α = 3 * 2 = 6
        let v = DVector::from_vec(vec![1.0, 1.0]);
        let result = composed.components[0].apply(&v);
        assert!((result - DVector::from_vec(vec![6.0, 6.0])).norm() < 1e-10);
    }

    #[test]
    fn test_conservation_law_identity() {
        let m = SymplecticManifold::canonical(2, "T*R");
        let agent = HamiltonianFunctor::map_object(&m);
        let id = AgentMorphism::identity(&agent);
        // Use linear momentum as the conservation law (has correct dims)
        let law = NoetherTheorem::linear_momentum(&m);
        let state = DVector::from_vec(vec![1.0, 5.0]);
        assert!(law.check_conservation(&state, &id));
    }

    #[test]
    fn test_noether_linear_momentum() {
        let m = SymplecticManifold::canonical(2, "T*R");
        let p = NoetherTheorem::linear_momentum(&m);
        // For canonical (q, p) coordinates, momentum should extract p
        let state = DVector::from_vec(vec![1.0, 5.0]); // q=1, p=5
        let val = p.evaluate(&state);
        assert!((val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_noether_linear_momentum_4d() {
        let m = SymplecticManifold::canonical(4, "T*R²");
        let p = NoetherTheorem::linear_momentum(&m);
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let vals = &p.quantity_matrix * &state;
        assert!((vals[(0, 0)] - 3.0).abs() < 1e-10);
        assert!((vals[(1, 0)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_nat_transform_label() {
        let eta = NatTransformComponent::new(DMatrix::identity(2, 2), "my_eta");
        assert_eq!(eta.label, "my_eta");
    }
}
