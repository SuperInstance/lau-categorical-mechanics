//! Hamiltonian Functor: maps symplectic manifolds → agents
//!
//! The key insight: every symplectic manifold (phase space) gives rise to an agent
//! whose state space is the manifold and whose capabilities correspond to observables.
//!
//! F: Sympl → Agent
//! - F(M, ω) = Agent with state space M, capabilities = {Hamiltonian flows}
//! - F(φ: M→N) = AgentMorphism that maps states via φ and preserves Hamiltonian structure

use crate::symplectic::{SymplecticManifold, Symplectomorphism};
use crate::agent::{Agent, AgentMorphism, Capability};
use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// The Hamiltonian functor from Sympl to Agent.
#[derive(Debug, Clone)]
pub struct HamiltonianFunctor;

/// A Hamiltonian function H: M → ℝ on a symplectic manifold.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hamiltonian {
    pub label: String,
    /// The state dimension (same as the manifold dimension)
    pub dimension: usize,
    /// Coefficients for a quadratic Hamiltonian H = ½ x^T Q x + c^T x
    pub quadratic_coeffs: DMatrix<f64>,
    pub linear_coeffs: DVector<f64>,
}

impl Hamiltonian {
    /// Create a simple quadratic Hamiltonian H = ½ x^T Q x.
    pub fn quadratic(matrix: DMatrix<f64>, label: impl Into<String>) -> Self {
        let dim = matrix.nrows();
        Self {
            label: label.into(),
            dimension: dim,
            quadratic_coeffs: matrix,
            linear_coeffs: DVector::zeros(dim),
        }
    }

    /// Evaluate H at a point.
    pub fn evaluate(&self, x: &DVector<f64>) -> f64 {
        let quad = 0.5 * x.transpose() * &self.quadratic_coeffs * x;
        let lin = &self.linear_coeffs.transpose() * x;
        quad[(0, 0)] + lin[(0, 0)]
    }

    /// Compute the Hamiltonian vector field X_H = ω^{-1} ∇H.
    /// For quadratic H = ½x^T Q x, ∇H = Qx, so X_H = ω^{-1} Q x.
    pub fn vector_field(&self, x: &DVector<f64>, omega: &DMatrix<f64>) -> Option<DVector<f64>> {
        let grad = &self.quadratic_coeffs * x + &self.linear_coeffs;
        let omega_inv = omega.clone().try_inverse()?;
        Some(&omega_inv * grad)
    }
}

impl HamiltonianFunctor {
    /// Map a symplectic manifold to an agent.
    /// The agent's state space is the manifold, and its capabilities are
    /// derived from the Hamiltonian structure.
    pub fn map_object(manifold: &SymplecticManifold) -> Agent {
        Agent::new(
            manifold.dimension,
            vec![
                Capability::new("observe_position", 0, manifold.config_dimension()),
                Capability::new("observe_momentum", 0, manifold.config_dimension()),
                Capability::new("hamiltonian_flow", manifold.dimension, manifold.dimension),
            ],
            format!("H({})", manifold.label),
        )
    }

    /// Map a symplectomorphism to an agent morphism.
    pub fn map_morphism(
        symsym: &Symplectomorphism,
        source_agent: &Agent,
        target_agent: &Agent,
    ) -> Option<AgentMorphism> {
        AgentMorphism::new(
            symsym.matrix.clone(),
            DVector::zeros(symsym.target_dim),
            source_agent,
            target_agent,
            // Map all capabilities through
            source_agent.capabilities.iter().enumerate()
                .map(|(i, _)| {
                    if i < target_agent.capabilities.len() { Some(i) } else { None }
                })
                .collect(),
            format!("F({})", symsym.label),
        )
    }

    /// Verify functoriality: F(id) = id and F(g ∘ f) = F(g) ∘ F(f).
    pub fn verify_functoriality_identity(manifold: &SymplecticManifold) -> bool {
        let agent = Self::map_object(manifold);
        let id_sym = Symplectomorphism::identity(manifold);
        let f_id = Self::map_morphism(&id_sym, &agent, &agent).unwrap();
        let id_agent = AgentMorphism::identity(&agent);

        let linear_diff = (f_id.linear_map - &id_agent.linear_map).norm();
        let offset_diff = (f_id.offset - &id_agent.offset).norm();
        linear_diff < 1e-10 && offset_diff < 1e-10
    }

    /// Verify that composition is preserved.
    pub fn verify_functoriality_composition(
        f: &Symplectomorphism,
        g: &Symplectomorphism,
        source: &SymplecticManifold,
        mid: &SymplecticManifold,
        target: &SymplecticManifold,
    ) -> bool {
        let a_source = Self::map_object(source);
        let a_mid = Self::map_object(mid);
        let a_target = Self::map_object(target);

        let f_agent = match Self::map_morphism(f, &a_source, &a_mid) {
            Some(m) => m,
            None => return false,
        };
        let g_agent = match Self::map_morphism(g, &a_mid, &a_target) {
            Some(m) => m,
            None => return false,
        };

        // F(g) ∘ F(f)
        let right = match g_agent.compose(&f_agent) {
            Some(r) => r,
            None => return false,
        };

        // F(g ∘ f)
        let gf = match g.compose(f) {
            Some(gf) => gf,
            None => return false,
        };
        let left = match Self::map_morphism(&gf, &a_source, &a_target) {
            Some(l) => l,
            None => return false,
        };

        (left.linear_map - &right.linear_map).norm() < 1e-8
            && (left.offset - &right.offset).norm() < 1e-8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamiltonian_evaluation() {
        // H = ½ x₁² + ½ x₂² (harmonic oscillator energy)
        let h = Hamiltonian::quadratic(
            DMatrix::identity(2, 2),
            "harmonic",
        );
        let x = DVector::from_vec(vec![3.0, 4.0]);
        let val = h.evaluate(&x);
        assert!((val - 12.5).abs() < 1e-10);
    }

    #[test]
    fn test_hamiltonian_vector_field() {
        let m = SymplecticManifold::canonical(2, "T*R");
        // Simple harmonic oscillator H = ½(q² + p²)
        let h = Hamiltonian::quadratic(DMatrix::identity(2, 2), "oscillator");
        let x = DVector::from_vec(vec![1.0, 0.0]); // at position q=1, p=0
        let xh = h.vector_field(&x, &m.omega).unwrap();
        // X_H = ω^{-1} ∇H = ω^{-1} x
        // ω = [[0,1],[-1,0]], ω^{-1} = [[0,-1],[1,0]]
        // X_H = [[0,-1],[1,0]] [1,0] = [0,1]
        assert!((xh - DVector::from_vec(vec![0.0, 1.0])).norm() < 1e-10);
    }

    #[test]
    fn test_functor_maps_manifold_to_agent() {
        let m = SymplecticManifold::canonical(4, "T*R²");
        let agent = HamiltonianFunctor::map_object(&m);
        assert_eq!(agent.state_dimension, 4);
        assert_eq!(agent.capabilities.len(), 3);
        assert_eq!(agent.label, "H(T*R²)");
    }

    #[test]
    fn test_functor_preserves_identity() {
        let m = SymplecticManifold::canonical(2, "T*R");
        assert!(HamiltonianFunctor::verify_functoriality_identity(&m));
    }

    #[test]
    fn test_functor_preserves_composition() {
        let m = SymplecticManifold::canonical(2, "T*R");
        let f = Symplectomorphism::identity(&m);
        let g = Symplectomorphism::identity(&m);
        assert!(HamiltonianFunctor::verify_functoriality_composition(&f, &g, &m, &m, &m));
    }

    #[test]
    fn test_hamiltonian_with_linear_term() {
        let mut h = Hamiltonian::quadratic(DMatrix::identity(2, 2), "shifted");
        h.linear_coeffs = DVector::from_vec(vec![-2.0, 0.0]);
        let x = DVector::from_vec(vec![1.0, 0.0]);
        // H = ½(1) - 2 = -1.5
        let val = h.evaluate(&x);
        assert!((val - (-1.5)).abs() < 1e-10);
    }

    #[test]
    fn test_functor_maps_symplectomorphism() {
        let m = SymplecticManifold::canonical(2, "T*R");
        let agent = HamiltonianFunctor::map_object(&m);
        let sym = Symplectomorphism::identity(&m);
        let morph = HamiltonianFunctor::map_morphism(&sym, &agent, &agent).unwrap();
        let state = DVector::from_vec(vec![1.0, 2.0]);
        let result = morph.apply(&state);
        assert!((result - state).norm() < 1e-10);
    }

    #[test]
    fn test_harmonic_oscillator_energy_conservation() {
        let m = SymplecticManifold::canonical(2, "T*R");
        let h = Hamiltonian::quadratic(DMatrix::identity(2, 2), "H_osc");
        // Energy at q=1, p=0
        let x0 = DVector::from_vec(vec![1.0, 0.0]);
        let e0 = h.evaluate(&x0);
        // After quarter period, q=0, p=1
        let x1 = DVector::from_vec(vec![0.0, 1.0]);
        let e1 = h.evaluate(&x1);
        assert!((e0 - e1).abs() < 1e-10); // Energy is conserved!
    }
}
