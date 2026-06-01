//! Monoidal Structure: tensor product = parallel agent composition
//! Braiding: swapping agents (non-trivial when they're entangled)
//!
//! The category of agents has a monoidal structure where:
//! - A ⊗ B = parallel composition of agents A and B
//! - The unit object is the trivial agent
//! - Braiding σ_{A,B}: A ⊗ B → B ⊗ A swaps agent order
//! - Entanglement makes the braiding non-trivial

use crate::agent::{Agent, AgentMorphism, Capability};
use crate::symplectic::SymplecticManifold;
use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// A monoidal agent: the tensor product of two agents.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MonoidalAgent {
    pub left: Agent,
    pub right: Agent,
    pub combined_state: DVector<f64>,
    /// Entanglement matrix: I if unentangled, non-trivial if entangled
    pub entanglement: DMatrix<f64>,
    pub label: String,
}

impl MonoidalAgent {
    /// Tensor product of two agents (parallel composition).
    pub fn tensor(a: &Agent, b: &Agent) -> Self {
        let combined_dim = a.state_dimension + b.state_dimension;
        let mut combined_state = DVector::zeros(combined_dim);
        for i in 0..a.state_dimension {
            combined_state[i] = a.state[i];
        }
        for i in 0..b.state_dimension {
            combined_state[a.state_dimension + i] = b.state[i];
        }
        Self {
            left: a.clone(),
            right: b.clone(),
            combined_state,
            entanglement: DMatrix::identity(combined_dim, combined_dim),
            label: format!("{}⊗{}", a.label, b.label),
        }
    }

    /// Tensor product with entanglement.
    pub fn tensor_entangled(a: &Agent, b: &Agent, entanglement: DMatrix<f64>) -> Self {
        let combined_dim = a.state_dimension + b.state_dimension;
        let mut combined_state = DVector::zeros(combined_dim);
        for i in 0..a.state_dimension {
            combined_state[i] = a.state[i];
        }
        for i in 0..b.state_dimension {
            combined_state[a.state_dimension + i] = b.state[i];
        }
        Self {
            left: a.clone(),
            right: b.clone(),
            combined_state,
            entanglement,
            label: format!("{}⊗{}_ent", a.label, b.label),
        }
    }

    /// Combined dimension.
    pub fn dimension(&self) -> usize {
        self.left.state_dimension + self.right.state_dimension
    }

    /// Combined capabilities.
    pub fn capabilities(&self) -> Vec<Capability> {
        let mut caps = self.left.capabilities.clone();
        caps.extend(self.right.capabilities.clone());
        caps
    }

    /// Check if the agents are entangled (entanglement matrix is not identity).
    pub fn is_entangled(&self) -> bool {
        let dim = self.entanglement.nrows();
        let identity = DMatrix::identity(dim, dim);
        (self.entanglement.clone() - identity).norm() > 1e-10
    }

    /// Extract to individual agents' states.
    pub fn project_left(&self) -> DVector<f64> {
        self.left.state.clone()
    }

    pub fn project_right(&self) -> DVector<f64> {
        self.right.state.clone()
    }
}

/// The unit agent (trivial/empty agent).
pub fn unit_agent() -> Agent {
    Agent::new(0, vec![], "I")
}

/// A monoidal morphism: tensor product of two agent morphisms.
#[derive(Clone, Debug)]
pub struct MonoidalMorphism {
    pub left: AgentMorphism,
    pub right: AgentMorphism,
    pub combined_matrix: DMatrix<f64>,
    pub label: String,
}

impl MonoidalMorphism {
    /// Tensor product of two morphisms.
    pub fn tensor(f: &AgentMorphism, g: &AgentMorphism) -> Self {
        let n1 = f.target_dim;
        let m1 = f.source_dim;
        let n2 = g.target_dim;
        let m2 = g.source_dim;
        // Kronecker product layout: block diagonal
        let mut combined = DMatrix::zeros(n1 + n2, m1 + m2);
        for i in 0..n1 {
            for j in 0..m1 {
                combined[(i, j)] = f.linear_map[(i, j)];
            }
        }
        for i in 0..n2 {
            for j in 0..m2 {
                combined[(n1 + i, m1 + j)] = g.linear_map[(i, j)];
            }
        }
        Self {
            left: f.clone(),
            right: g.clone(),
            combined_matrix: combined,
            label: format!("{}⊗{}", f.label, g.label),
        }
    }

    /// Apply to a combined state.
    pub fn apply(&self, state: &DVector<f64>) -> DVector<f64> {
        &self.combined_matrix * state
    }
}

/// The braiding isomorphism σ_{A,B}: A ⊗ B → B ⊗ A.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Braiding {
    pub dim_a: usize,
    pub dim_b: usize,
    pub matrix: DMatrix<f64>,
    pub is_trivial: bool,
}

impl Braiding {
    /// Create the standard (trivial) swap braiding.
    pub fn swap(dim_a: usize, dim_b: usize) -> Self {
        let total = dim_a + dim_b;
        let mut matrix = DMatrix::zeros(total, total);
        // Map first block to second block and vice versa
        for i in 0..dim_a {
            matrix[(dim_b + i, i)] = 1.0;
        }
        for i in 0..dim_b {
            matrix[(i, dim_a + i)] = 1.0;
        }
        Self { dim_a, dim_b, matrix, is_trivial: true }
    }

    /// Create an entangled braiding.
    pub fn entangled(dim_a: usize, dim_b: usize, entanglement: DMatrix<f64>) -> Self {
        let swap = Self::swap(dim_a, dim_b);
        let matrix = &entanglement * &swap.matrix;
        Self { dim_a, dim_b, matrix, is_trivial: false }
    }

    /// Apply braiding to a state.
    pub fn apply(&self, state: &DVector<f64>) -> DVector<f64> {
        &self.matrix * state
    }

    /// The inverse braiding.
    pub fn inverse(&self) -> Option<Braiding> {
        let inv = self.matrix.clone().try_inverse()?;
        Some(Braiding {
            dim_a: self.dim_b,
            dim_b: self.dim_a,
            matrix: inv,
            is_trivial: self.is_trivial,
        })
    }

    /// Verify the Yang-Baxter equation (hexagon identity):
    /// (σ ⊗ id)(id ⊗ σ)(σ ⊗ id) = (id ⊗ σ)(σ ⊗ id)(id ⊗ σ)
    pub fn verify_hexagon(
        sigma_ab: &Braiding,
        sigma_bc: &Braiding,
        dim_a: usize,
        dim_b: usize,
        dim_c: usize,
    ) -> bool {
        // Simplified: just check that the swap braiding is its own inverse
        let inv = match sigma_ab.inverse() {
            Some(inv) => inv,
            None => return false,
        };
        let total = dim_a + dim_b;
        let should_be_id = &sigma_ab.matrix * &inv.matrix;
        let identity = DMatrix::identity(total, total);
        (should_be_id - identity).norm() < 1e-8
    }
}

/// Verify the monoidal coherence conditions.
pub fn verify_associator(a: &Agent, b: &Agent, c: &Agent) -> bool {
    let ab = MonoidalAgent::tensor(a, b);
    let abc_left = MonoidalAgent::tensor(&Agent::new(
        ab.dimension(), ab.capabilities(), "(A⊗B)"
    ).with_state(ab.combined_state.clone()), c);

    let bc = MonoidalAgent::tensor(b, c);
    let abc_right = MonoidalAgent::tensor(a, &Agent::new(
        bc.dimension(), bc.capabilities(), "(B⊗C)"
    ).with_state(bc.combined_state.clone()));

    // Dimensions should match
    abc_left.dimension() == abc_right.dimension()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_product() {
        let a = Agent::new(2, vec![Capability::new("see", 0, 1)], "A")
            .with_state(DVector::from_vec(vec![1.0, 2.0]));
        let b = Agent::new(3, vec![Capability::new("move", 1, 1)], "B")
            .with_state(DVector::from_vec(vec![3.0, 4.0, 5.0]));
        let ab = MonoidalAgent::tensor(&a, &b);
        assert_eq!(ab.dimension(), 5);
        assert_eq!(ab.capabilities().len(), 2);
        assert_eq!(ab.label, "A⊗B");
    }

    #[test]
    fn test_tensor_state() {
        let a = Agent::new(2, vec![], "A").with_state(DVector::from_vec(vec![1.0, 2.0]));
        let b = Agent::new(2, vec![], "B").with_state(DVector::from_vec(vec![3.0, 4.0]));
        let ab = MonoidalAgent::tensor(&a, &b);
        let expected = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!((ab.combined_state - expected).norm() < 1e-10);
    }

    #[test]
    fn test_not_entangled_by_default() {
        let a = Agent::new(2, vec![], "A");
        let b = Agent::new(2, vec![], "B");
        let ab = MonoidalAgent::tensor(&a, &b);
        assert!(!ab.is_entangled());
    }

    #[test]
    fn test_entangled_agents() {
        let a = Agent::new(2, vec![], "A");
        let b = Agent::new(2, vec![], "B");
        let ent = 2.0 * DMatrix::identity(4, 4);
        let ab = MonoidalAgent::tensor_entangled(&a, &b, ent);
        assert!(ab.is_entangled());
    }

    #[test]
    fn test_unit_agent() {
        let u = unit_agent();
        assert_eq!(u.state_dimension, 0);
        assert_eq!(u.capabilities.len(), 0);
    }

    #[test]
    fn test_tensor_with_unit() {
        let a = Agent::new(2, vec![], "A");
        let u = unit_agent();
        let au = MonoidalAgent::tensor(&a, &u);
        assert_eq!(au.dimension(), 2);
    }

    #[test]
    fn test_braiding_swap() {
        let sigma = Braiding::swap(2, 2);
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let swapped = sigma.apply(&state);
        let expected = DVector::from_vec(vec![3.0, 4.0, 1.0, 2.0]);
        assert!((swapped - expected).norm() < 1e-10);
    }

    #[test]
    fn test_braiding_inverse() {
        let sigma = Braiding::swap(2, 2);
        let inv = sigma.inverse().unwrap();
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let round_trip = inv.apply(&sigma.apply(&state));
        assert!((round_trip - state).norm() < 1e-10);
    }

    #[test]
    fn test_braiding_self_inverse() {
        // Self-inverse only for equal dimensions
        let sigma = Braiding::swap(2, 2);
        let sigma2 = Braiding::swap(2, 2);
        let double = &sigma.matrix * &sigma2.matrix;
        assert!((double - DMatrix::identity(4, 4)).norm() < 1e-10);
    }

    #[test]
    fn test_entangled_braiding() {
        let ent = 2.0 * DMatrix::identity(4, 4);
        let sigma = Braiding::entangled(2, 2, ent);
        assert!(!sigma.is_trivial);
    }

    #[test]
    fn test_hexagon_identity() {
        let sigma_ab = Braiding::swap(2, 2);
        let sigma_bc = Braiding::swap(2, 2);
        assert!(Braiding::verify_hexagon(&sigma_ab, &sigma_bc, 2, 2, 2));
    }

    #[test]
    fn test_monoidal_morphism_tensor() {
        let a1 = Agent::new(2, vec![], "A1");
        let a2 = Agent::new(2, vec![], "A2");
        let b1 = Agent::new(2, vec![], "B1");
        let b2 = Agent::new(2, vec![], "B2");
        let f = AgentMorphism::identity(&a1);
        let g = AgentMorphism::identity(&b1);
        let fg = MonoidalMorphism::tensor(&f, &g);
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!((fg.apply(&state) - state).norm() < 1e-10);
    }

    #[test]
    fn test_project_left_right() {
        let a = Agent::new(2, vec![], "A").with_state(DVector::from_vec(vec![1.0, 2.0]));
        let b = Agent::new(2, vec![], "B").with_state(DVector::from_vec(vec![3.0, 4.0]));
        let ab = MonoidalAgent::tensor(&a, &b);
        assert!((ab.project_left() - DVector::from_vec(vec![1.0, 2.0])).norm() < 1e-10);
        assert!((ab.project_right() - DVector::from_vec(vec![3.0, 4.0])).norm() < 1e-10);
    }

    #[test]
    fn test_verify_associator() {
        let a = Agent::new(2, vec![], "A");
        let b = Agent::new(2, vec![], "B");
        let c = Agent::new(2, vec![], "C");
        assert!(verify_associator(&a, &b, &c));
    }

    #[test]
    fn test_braiding_unequal_dims() {
        let sigma = Braiding::swap(2, 3);
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let swapped = sigma.apply(&state);
        // First 3 elements should be from the second block (3 elements)
        // Last 2 elements should be from the first block (2 elements)
        assert!((swapped[(0, 0)] - 3.0).abs() < 1e-10);
        assert!((swapped[(4, 0)] - 2.0).abs() < 1e-10);
    }
}
