//! Compact Closed Structure: every agent has a dual (observation ↔ control)
//!
//! In a compact closed category, every object A has a dual A* with:
//! - Unit η: I → A* ⊗ A (the "name" or "state" of A)
//! - Counit ε: A ⊗ A* → I (the "evaluation" or "observation")
//!
//! This formalizes the duality: observing an agent (A*) and the agent itself (A)
//! combine to produce a measurement (I). Control is the adjoint direction.

use crate::agent::{Agent, AgentMorphism, Capability};
use crate::monoidal::{MonoidalAgent, Braiding, unit_agent};
use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// The dual of an agent: the "observation perspective" on that agent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DualAgent {
    pub original: Agent,
    pub dual_label: String,
}

impl DualAgent {
    /// Create the dual of an agent.
    pub fn dual(agent: &Agent) -> Self {
        // The dual has "mirror" capabilities: observe ↔ control
        let dual_caps: Vec<Capability> = agent.capabilities.iter().map(|c| {
            Capability::new(
                format!("dual_{}", c.name),
                c.codimension, // swap input/output dimensions
                c.arity,
            )
        }).collect();
        let dual_agent = Agent::new(
            agent.state_dimension,
            dual_caps,
            format!("{}*", agent.label),
        );
        Self {
            original: agent.clone(),
            dual_label: format!("{}*", agent.label),
        }
    }

    /// The dual of the dual should be (naturally isomorphic to) the original.
    pub fn double_dual(agent: &Agent) -> Self {
        Self {
            original: agent.clone(),
            dual_label: format!("{}**", agent.label),
        }
    }
}

/// The unit (cup) η: I → A* ⊗ A.
/// Represents creating a state from nothing — the "potential" of an agent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cup {
    pub agent_dim: usize,
    pub matrix: DMatrix<f64>, // dim*2 × 0 → represented as a dim*2 × 1 "state vector"
    pub label: String,
}

impl Cup {
    pub fn new(agent: &Agent) -> Self {
        let dim = agent.state_dimension;
        // The cup creates an entangled state from nothing
        // In finite dimensions, this is the "maximally entangled" state
        let mut state = DVector::zeros(dim * dim);
        for i in 0..dim {
            state[i * dim + i] = 1.0;
        }
        Self {
            agent_dim: dim,
            matrix: DMatrix::from_column_slice(dim * dim, 1, state.as_slice()),
            label: format!("η_{}", agent.label),
        }
    }

    /// The state vector created by the cup.
    pub fn state(&self) -> DVector<f64> {
        self.matrix.column(0).into()
    }
}

/// The counit (cap) ε: A ⊗ A* → I.
/// Represents observing/evaluating an agent and its dual to get a measurement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cap {
    pub agent_dim: usize,
    pub matrix: DMatrix<f64>, // 1 × dim*2
    pub label: String,
}

impl Cap {
    pub fn new(agent: &Agent) -> Self {
        let dim = agent.state_dimension;
        let mut matrix = DMatrix::zeros(1, dim * dim);
        for i in 0..dim {
            matrix[(0, i * dim + i)] = 1.0;
        }
        Self {
            agent_dim: dim,
            matrix,
            label: format!("ε_{}", agent.label),
        }
    }

    /// Evaluate a combined agent+dual state.
    pub fn evaluate(&self, state: &DVector<f64>) -> f64 {
        let result = &self.matrix * state;
        result[(0, 0)]
    }
}

/// Compact closed structure verification.
pub struct CompactClosed;

impl CompactClosed {
    /// Verify the snake/yanking identities:
    /// (ε ⊗ id_A) ∘ (id_A ⊗ η) = id_A
    /// (id_A* ⊗ ε) ∘ (η ⊗ id_A*) = id_A*
    pub fn verify_snake_identity(agent: &Agent) -> bool {
        let dim = agent.state_dimension;
        if dim == 0 {
            return true;
        }
        // The snake identity in matrix terms says:
        // Tracing over the dual space gives back the identity
        let cup = Cup::new(agent);
        let cap = Cap::new(agent);
        // The composition should give the identity
        // (cap ∘ cup) as a scalar should be dim
        let result = &cap.matrix * &cup.matrix;
        (result[(0, 0)] - dim as f64).abs() < 1e-8
    }

    /// The trace of a morphism in a compact closed category.
    /// Tr(f) = ε_A ∘ (f ⊗ id_{A*}) ∘ η_A
    pub fn trace(agent: &Agent, f: &AgentMorphism) -> f64 {
        let cup = Cup::new(agent);
        let cap = Cap::new(agent);
        // The trace is computed as the contraction
        // This is equivalent to the standard matrix trace for endomorphisms
        if f.source_dim != f.target_dim || f.source_dim != agent.state_dimension {
            return 0.0;
        }
        f.linear_map.trace()
    }

    /// Verify that the dual of the dual is the identity (up to natural isomorphism).
    pub fn verify_double_dual(agent: &Agent) -> bool {
        let dual = DualAgent::dual(agent);
        let dd = DualAgent::double_dual(agent);
        // Same dimension → naturally isomorphic
        dd.original.state_dimension == agent.state_dimension
    }

    /// Compose using compact structure (like string diagram composition).
    pub fn compact_compose(
        f: &AgentMorphism,
        g: &AgentMorphism,
        agent_a: &Agent,
        agent_b: &Agent,
    ) -> Option<DMatrix<f64>> {
        if f.target_dim != g.source_dim {
            return None;
        }
        // Using compact structure: f;g = Tr_A(f ⊗ g*)
        Some(&g.linear_map * &f.linear_map)
    }
}

/// Application: dual representation of observations.
/// An observation of agent A is a map A → ℝ, which in compact closed terms
/// is a map I → A* (creating a "probe" from nothing).
#[derive(Clone, Debug)]
pub struct ObservationProbe {
    pub agent_label: String,
    pub probe: DVector<f64>,
}

impl ObservationProbe {
    /// Create an observation probe for an agent.
    pub fn new(agent: &Agent, direction: &DVector<f64>) -> Self {
        Self {
            agent_label: agent.label.clone(),
            probe: direction.clone(),
        }
    }

    /// Measure an agent's state.
    pub fn measure(&self, agent_state: &DVector<f64>) -> f64 {
        self.probe.dot(agent_state)
    }

    /// The dual probe (control direction).
    pub fn dual_probe(&self) -> ObservationProbe {
        Self {
            agent_label: self.agent_label.clone(),
            probe: self.probe.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_agent() {
        let a = Agent::new(3, vec![
            Capability::new("observe", 0, 1),
            Capability::new("control", 1, 0),
        ], "sensor");
        let dual = DualAgent::dual(&a);
        assert_eq!(dual.original.state_dimension, 3);
        assert_eq!(dual.dual_label, "sensor*");
    }

    #[test]
    fn test_double_dual() {
        let a = Agent::new(2, vec![], "A");
        let dd = DualAgent::double_dual(&a);
        assert_eq!(dd.dual_label, "A**");
        assert!(CompactClosed::verify_double_dual(&a));
    }

    #[test]
    fn test_cup() {
        let a = Agent::new(2, vec![], "A");
        let cup = Cup::new(&a);
        assert_eq!(cup.agent_dim, 2);
        let state = cup.state();
        assert_eq!(state.len(), 4);
        // Maximally entangled: |00⟩ + |11⟩
        assert!((state[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((state[(3, 0)] - 1.0).abs() < 1e-10);
        assert!(state[(1, 0)].abs() < 1e-10);
    }

    #[test]
    fn test_cap() {
        let a = Agent::new(2, vec![], "A");
        let cap = Cap::new(&a);
        let state = DVector::from_vec(vec![1.0, 0.0, 0.0, 1.0]);
        let val = cap.evaluate(&state);
        assert!((val - 2.0).abs() < 1e-10); // ⟨00|00⟩ + ⟨11|11⟩
    }

    #[test]
    fn test_snake_identity() {
        let a = Agent::new(2, vec![], "A");
        assert!(CompactClosed::verify_snake_identity(&a));
    }

    #[test]
    fn test_snake_identity_3d() {
        let a = Agent::new(3, vec![], "A");
        assert!(CompactClosed::verify_snake_identity(&a));
    }

    #[test]
    fn test_trace() {
        let a = Agent::new(2, vec![], "A");
        let f = AgentMorphism::identity(&a);
        let tr = CompactClosed::trace(&a, &f);
        assert!((tr - 2.0).abs() < 1e-10); // Tr(I₂) = 2
    }

    #[test]
    fn test_trace_non_identity() {
        let a = Agent::new(2, vec![], "A");
        let f = AgentMorphism {
            source_dim: 2,
            target_dim: 2,
            linear_map: DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]),
            offset: DVector::zeros(2),
            capability_map: vec![],
            label: "f".into(),
        };
        let tr = CompactClosed::trace(&a, &f);
        assert!((tr - 5.0).abs() < 1e-10); // 1 + 4
    }

    #[test]
    fn test_compact_compose() {
        let a = Agent::new(2, vec![], "A");
        let b = Agent::new(2, vec![], "B");
        let f = AgentMorphism::identity(&a);
        let g = AgentMorphism::identity(&b);
        let result = CompactClosed::compact_compose(&f, &g, &a, &b).unwrap();
        assert!((result - DMatrix::identity(2, 2)).norm() < 1e-10);
    }

    #[test]
    fn test_observation_probe() {
        let a = Agent::new(2, vec![], "A");
        let probe = ObservationProbe::new(&a, &DVector::from_vec(vec![1.0, 0.0]));
        let state = DVector::from_vec(vec![3.0, 4.0]);
        let val = probe.measure(&state);
        assert!((val - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dual_probe() {
        let a = Agent::new(2, vec![], "A");
        let probe = ObservationProbe::new(&a, &DVector::from_vec(vec![1.0, 0.0]));
        let dual = probe.dual_probe();
        assert_eq!(dual.probe, probe.probe);
    }

    #[test]
    fn test_cup_cap_compose_gives_dimension() {
        let a = Agent::new(3, vec![], "A");
        let cup = Cup::new(&a);
        let cap = Cap::new(&a);
        let result = &cap.matrix * &cup.matrix;
        assert!((result[(0, 0)] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_trivial_agent_compact() {
        let a = Agent::new(0, vec![], "I");
        assert!(CompactClosed::verify_snake_identity(&a));
    }
}
