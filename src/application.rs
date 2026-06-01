//! Application: Categorical Foundation for the Lau Ecosystem
//!
//! This module ties everything together, showing how categorical mechanics
//! provides the mathematical foundation for multi-agent systems.

use crate::symplectic::{SymplecticManifold, Symplectomorphism};
use crate::agent::{Agent, AgentMorphism, Capability};
use crate::functor::{HamiltonianFunctor, Hamiltonian};
use crate::natural::{NaturalTransformation, NatTransformComponent, ConservationLaw, NoetherTheorem};
use crate::presheaf::{Observable, ObservablePresheaf};
use crate::yoneda::{YonedaEmbedding, AgentProfile, RepresentableFunctor};
use crate::adjunction::{ObsCtrlAdjunction, StateSpace};
use crate::monoidal::{MonoidalAgent, Braiding, unit_agent};
use crate::compact::{DualAgent, Cup, Cap, CompactClosed, ObservationProbe};
use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// A complete categorical mechanics system.
#[derive(Clone, Debug)]
pub struct LauSystem {
    pub label: String,
    pub agents: Vec<Agent>,
    pub symplectic_manifolds: Vec<SymplecticManifold>,
}

impl LauSystem {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            agents: Vec::new(),
            symplectic_manifolds: Vec::new(),
        }
    }

    /// Add a physical agent derived from a symplectic manifold.
    pub fn add_physical_agent(&mut self, manifold: &SymplecticManifold) -> &Agent {
        let agent = HamiltonianFunctor::map_object(manifold);
        self.symplectic_manifolds.push(manifold.clone());
        self.agents.push(agent);
        self.agents.last().unwrap()
    }

    /// Compose two agents in parallel.
    pub fn parallel_compose(&self, i: usize, j: usize) -> Option<MonoidalAgent> {
        let a = self.agents.get(i)?;
        let b = self.agents.get(j)?;
        Some(MonoidalAgent::tensor(a, b))
    }

    /// Observe an agent with a probe.
    pub fn observe(&self, agent_idx: usize, probe_direction: &DVector<f64>) -> Option<f64> {
        let agent = self.agents.get(agent_idx)?;
        let probe = ObservationProbe::new(agent, probe_direction);
        Some(probe.measure(&agent.state))
    }

    /// Profile all agents using Yoneda.
    pub fn profile_all(&self) -> Vec<AgentProfile> {
        self.agents.iter()
            .map(|a| AgentProfile::profile(a, &self.agents))
            .collect()
    }

    /// Compute the Hamiltonian energy of an agent.
    pub fn energy(&self, agent_idx: usize, hamiltonian: &Hamiltonian) -> Option<f64> {
        let agent = self.agents.get(agent_idx)?;
        Some(hamiltonian.evaluate(&agent.state))
    }
}

/// Build a sample Lau system with symplectic agents.
pub fn sample_system() -> LauSystem {
    let mut system = LauSystem::new("sample-lau");

    // Phase space for a harmonic oscillator
    let phase_2d = SymplecticManifold::canonical(2, "T*R");
    system.add_physical_agent(&phase_2d);

    // Phase space for a 2D oscillator
    let phase_4d = SymplecticManifold::canonical(4, "T*R²");
    system.add_physical_agent(&phase_4d);

    // Add a pure computational agent
    let comp_agent = Agent::new(
        4,
        vec![
            Capability::new("compute", 2, 2),
            Capability::new("communicate", 1, 1),
        ],
        "compute-agent",
    ).with_state(DVector::from_vec(vec![1.0, 0.0, 0.0, 1.0]));
    system.agents.push(comp_agent);

    system
}

/// The full categorical pipeline: from symplectic manifold to agent behavior.
pub fn categorical_pipeline() -> PipelineResult {
    let manifold = SymplecticManifold::canonical(2, "T*R");
    let agent = HamiltonianFunctor::map_object(&manifold);

    // Verify functoriality
    let functor_ok = HamiltonianFunctor::verify_functoriality_identity(&manifold);

    // Natural transformation: conservation law
    let momentum = NoetherTheorem::linear_momentum(&manifold);
    let state = DVector::from_vec(vec![1.0, 5.0]); // q=1, p=5

    // Presheaf of observables
    let obs = ObservablePresheaf::new(vec![
        Observable::vector(DMatrix::identity(2, 2), 2, "full"),
    ]);
    let id = AgentMorphism::identity(&agent);
    let presheaf_ok = obs.verify_identity(&id);

    // Yoneda: profile
    let profile = AgentProfile::profile(&agent, &[agent.clone()]);

    // Adjunction
    let adj = ObsCtrlAdjunction::new(2);
    let state_space = StateSpace::new(2, "S");
    let (tri1, tri2) = adj.verify_triangle_identities(&state_space, &agent);

    // Monoidal
    let a2 = Agent::new(2, vec![], "A2");
    let mono = MonoidalAgent::tensor(&agent, &a2);

    // Compact closed
    let snake_ok = CompactClosed::verify_snake_identity(&agent);

    // Dual
    let dual = DualAgent::dual(&agent);

    PipelineResult {
        manifold_label: manifold.label,
        agent_label: agent.label,
        functoriality_verified: functor_ok,
        conserved_momentum: momentum.evaluate(&state),
        presheaf_verified: presheaf_ok,
        profile_features: profile.features.clone(),
        triangle_identities: (tri1, tri2),
        monoidal_dim: mono.dimension(),
        snake_identity: snake_ok,
        dual_label: dual.dual_label,
    }
}

/// Result of running the full categorical pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineResult {
    pub manifold_label: String,
    pub agent_label: String,
    pub functoriality_verified: bool,
    pub conserved_momentum: f64,
    pub presheaf_verified: bool,
    pub profile_features: DVector<f64>,
    pub triangle_identities: (bool, bool),
    pub monoidal_dim: usize,
    pub snake_identity: bool,
    pub dual_label: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_system_creation() {
        let system = sample_system();
        assert_eq!(system.agents.len(), 3);
        assert_eq!(system.symplectic_manifolds.len(), 2);
    }

    #[test]
    fn test_physical_agent_from_manifold() {
        let mut system = LauSystem::new("test");
        let m = SymplecticManifold::canonical(2, "T*R");
        let agent = system.add_physical_agent(&m);
        assert_eq!(agent.state_dimension, 2);
        assert_eq!(agent.capabilities.len(), 3);
    }

    #[test]
    fn test_parallel_compose() {
        let system = sample_system();
        let mono = system.parallel_compose(0, 1).unwrap();
        assert_eq!(mono.dimension(), 6); // 2 + 4
    }

    #[test]
    fn test_observe_agent() {
        let mut system = LauSystem::new("test");
        let m = SymplecticManifold::canonical(2, "T*R");
        let idx = system.agents.len();
        let agent = Agent::new(2, vec![], "obs-agent")
            .with_state(DVector::from_vec(vec![3.0, 4.0]));
        system.agents.push(agent);
        let val = system.observe(idx, &DVector::from_vec(vec![1.0, 0.0])).unwrap();
        assert!((val - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_profile_all() {
        let system = sample_system();
        let profiles = system.profile_all();
        assert_eq!(profiles.len(), 3);
    }

    #[test]
    fn test_energy_computation() {
        let mut system = LauSystem::new("test");
        let m = SymplecticManifold::canonical(2, "T*R");
        let agent = Agent::new(2, vec![], "osc")
            .with_state(DVector::from_vec(vec![1.0, 1.0]));
        system.agents.push(agent);
        let h = Hamiltonian::quadratic(DMatrix::identity(2, 2), "H");
        let e = system.energy(0, &h).unwrap();
        assert!((e - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_full_pipeline() {
        let result = categorical_pipeline();
        assert!(result.functoriality_verified);
        assert!(result.presheaf_verified);
        assert!(result.snake_identity);
        assert!(result.triangle_identities.0);
        assert!(result.triangle_identities.1);
        assert!((result.conserved_momentum - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_serialization() {
        let result = categorical_pipeline();
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("functoriality_verified"));
        assert!(json.contains("T*R"));
    }

    #[test]
    fn test_multi_agent_system() {
        let mut system = LauSystem::new("multi");
        for i in 0..5 {
            let m = SymplecticManifold::canonical(2, format!("agent_{}", i));
            system.add_physical_agent(&m);
        }
        assert_eq!(system.agents.len(), 5);
        // Pairwise parallel composition
        for i in 0..4 {
            let mono = system.parallel_compose(i, i + 1).unwrap();
            assert_eq!(mono.dimension(), 4);
        }
    }

    #[test]
    fn test_categorical_interoperability() {
        // Agents from different manifolds should compose
        let m2 = SymplecticManifold::canonical(2, "2d");
        let m4 = SymplecticManifold::canonical(4, "4d");
        let a2 = HamiltonianFunctor::map_object(&m2);
        let a4 = HamiltonianFunctor::map_object(&m4);
        let mono = MonoidalAgent::tensor(&a2, &a4);
        assert_eq!(mono.dimension(), 6);

        // They can be braided
        let sigma = Braiding::swap(2, 4);
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let swapped = sigma.apply(&state);
        assert_eq!(swapped.len(), 6);
    }

    #[test]
    fn test_conservation_across_composition() {
        let m = SymplecticManifold::canonical(2, "T*R");
        let a = HamiltonianFunctor::map_object(&m);
        let b = HamiltonianFunctor::map_object(&m);
        let mono = MonoidalAgent::tensor(&a, &b);

        // Conservation should hold in both components
        let mom = NoetherTheorem::linear_momentum(&m);
        let left_state = DVector::from_vec(vec![1.0, 5.0]);
        assert!((mom.evaluate(&left_state) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_yoneda_distinguishes_agents() {
        let a = Agent::new(2, vec![], "small");
        let b = Agent::new(4, vec![], "large");
        let probe = Agent::new(2, vec![], "probe");
        assert!(!YonedaEmbedding::agents_equal_by_relationships(&a, &b, &[probe]));
    }

    #[test]
    fn test_observation_control_duality() {
        let adj = ObsCtrlAdjunction::new(2);
        let state = StateSpace::new(2, "S");
        let agent = Agent::new(2, vec![Capability::new("observe", 0, 1)], "obs");
        let unit = adj.unit(&state);
        let counit = adj.counit(&agent);
        // Unit goes from state to controlled space
        assert_eq!(unit.source_dim, 2);
        // Counit goes from observed to agent
        assert_eq!(counit.target_dim, 2);
    }
}
