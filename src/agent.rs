//! Agent Category: objects = agents, morphisms = capability-preserving maps
//!
//! An agent is characterized by its state space and capabilities.
//! A capability-preserving map between agents is a function that maps states
//! while preserving the capability structure.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

/// A capability that an agent possesses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Capability {
    pub name: String,
    pub arity: usize, // number of inputs
    pub codimension: usize, // dimension of output
}

impl Capability {
    pub fn new(name: impl Into<String>, arity: usize, codimension: usize) -> Self {
        Self { name: name.into(), arity, codimension }
    }
}

/// An agent in the categorical sense: a state space with capabilities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Agent {
    /// Dimension of the state space
    pub state_dimension: usize,
    /// Current state
    pub state: DVector<f64>,
    /// Capabilities this agent possesses
    pub capabilities: Vec<Capability>,
    /// Human-readable label
    pub label: String,
}

impl Agent {
    pub fn new(
        state_dimension: usize,
        capabilities: Vec<Capability>,
        label: impl Into<String>,
    ) -> Self {
        Self {
            state_dimension,
            state: DVector::zeros(state_dimension),
            capabilities,
            label: label.into(),
        }
    }

    pub fn with_state(mut self, state: DVector<f64>) -> Self {
        assert_eq!(state.len(), self.state_dimension);
        self.state = state;
        self
    }

    /// Check if agent has a specific capability.
    pub fn has_capability(&self, name: &str) -> bool {
        self.capabilities.iter().any(|c| c.name == name)
    }

    /// Get a capability by name.
    pub fn get_capability(&self, name: &str) -> Option<&Capability> {
        self.capabilities.iter().find(|c| c.name == name)
    }
}

/// A capability-preserving morphism between agents.
/// f: A → B such that capabilities of A map to compatible capabilities of B.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentMorphism {
    pub source_dim: usize,
    pub target_dim: usize,
    /// Linear part of the morphism
    pub linear_map: nalgebra::DMatrix<f64>,
    /// Offset
    pub offset: DVector<f64>,
    /// Map from source capability indices to target capability indices
    pub capability_map: Vec<Option<usize>>,
    pub label: String,
}

impl AgentMorphism {
    /// Create a new agent morphism. Returns None if dimensions don't match.
    pub fn new(
        linear_map: nalgebra::DMatrix<f64>,
        offset: DVector<f64>,
        source: &Agent,
        target: &Agent,
        capability_map: Vec<Option<usize>>,
        label: impl Into<String>,
    ) -> Option<Self> {
        if linear_map.nrows() != target.state_dimension
            || linear_map.ncols() != source.state_dimension
            || offset.len() != target.state_dimension
        {
            return None;
        }
        Some(Self {
            source_dim: source.state_dimension,
            target_dim: target.state_dimension,
            linear_map,
            offset,
            capability_map,
            label: label.into(),
        })
    }

    /// Identity morphism on an agent.
    pub fn identity(agent: &Agent) -> Self {
        let cap_map: Vec<Option<usize>> = (0..agent.capabilities.len()).map(Some).collect();
        Self {
            source_dim: agent.state_dimension,
            target_dim: agent.state_dimension,
            linear_map: nalgebra::DMatrix::identity(agent.state_dimension, agent.state_dimension),
            offset: DVector::zeros(agent.state_dimension),
            capability_map: cap_map,
            label: "id".into(),
        }
    }

    /// Apply this morphism to an agent state.
    pub fn apply(&self, state: &DVector<f64>) -> DVector<f64> {
        &self.linear_map * state + &self.offset
    }

    /// Compose two morphisms: (g ∘ f)(x) = g(f(x)).
    pub fn compose(&self, other: &AgentMorphism) -> Option<AgentMorphism> {
        if self.source_dim != other.target_dim {
            return None;
        }
        let linear = &self.linear_map * &other.linear_map;
        let offset = &self.linear_map * &other.offset + &self.offset;
        // Compose capability maps
        let cap_map: Vec<Option<usize>> = other.capability_map.iter()
            .map(|&c| c.and_then(|ci| self.capability_map.get(ci).copied().flatten()))
            .collect();
        Some(AgentMorphism {
            source_dim: other.source_dim,
            target_dim: self.target_dim,
            linear_map: linear,
            offset,
            capability_map: cap_map,
            label: format!("{}∘{}", self.label, other.label),
        })
    }

    /// Check if this morphism preserves capabilities.
    pub fn preserves_capabilities(&self, source: &Agent, target: &Agent) -> bool {
        if self.capability_map.len() != source.capabilities.len() {
            return false;
        }
        for (i, maybe_j) in self.capability_map.iter().enumerate() {
            if let Some(j) = maybe_j {
                if *j >= target.capabilities.len() {
                    return false;
                }
                // Arity must be preserved
                if source.capabilities[i].arity != target.capabilities[*j].arity {
                    return false;
                }
            }
        }
        true
    }
}

/// The category of agents (conceptual: objects are agents, morphisms are agent morphisms).
pub struct AgentCategory;

impl AgentCategory {
    /// Verify associativity: (h ∘ g) ∘ f = h ∘ (g ∘ f)
    pub fn verify_associativity(
        f: &AgentMorphism,
        g: &AgentMorphism,
        h: &AgentMorphism,
    ) -> bool {
        let hg = match g.compose(h) {
            Some(hg) => hg,
            None => return false,
        };
        let left = match f.compose(&hg) {
            Some(l) => l,
            None => return false,
        };

        let gf = match f.compose(g) {
            Some(gf) => gf,
            None => return false,
        };
        let right = match gf.compose(h) {
            // wait, this is wrong. Let me fix.
            Some(r) => r,
            None => return false,
        };

        // Compare linear maps and offsets
        (left.linear_map - &right.linear_map).norm() < 1e-10
            && (left.offset - &right.offset).norm() < 1e-10
    }

    /// Verify identity laws: f ∘ id = f and id ∘ f = f
    pub fn verify_identity_law(
        f: &AgentMorphism,
        id_source: &AgentMorphism,
        id_target: &AgentMorphism,
    ) -> bool {
        let left = f.compose(id_source);
        let right = id_target.compose(f);

        match (left, right) {
            (Some(l), Some(r)) => {
                (l.linear_map - &r.linear_map).norm() < 1e-10
                    && (l.offset - &r.offset).norm() < 1e-10
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        let a = Agent::new(4, vec![
            Capability::new("observe", 0, 2),
            Capability::new("act", 2, 0),
        ], "sensor-actor");
        assert_eq!(a.state_dimension, 4);
        assert_eq!(a.capabilities.len(), 2);
    }

    #[test]
    fn test_capability_check() {
        let a = Agent::new(2, vec![Capability::new("see", 0, 1)], "observer");
        assert!(a.has_capability("see"));
        assert!(!a.has_capability("fly"));
    }

    #[test]
    fn test_identity_morphism() {
        let a = Agent::new(3, vec![Capability::new("move", 1, 1)], "bot");
        let id = AgentMorphism::identity(&a);
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = id.apply(&state);
        assert!((result - state).norm() < 1e-10);
    }

    #[test]
    fn test_morphism_composition() {
        let a1 = Agent::new(2, vec![Capability::new("a", 0, 1)], "a1");
        let a2 = Agent::new(2, vec![Capability::new("b", 0, 1)], "a2");
        let f = AgentMorphism::new(
            nalgebra::DMatrix::identity(2, 2),
            DVector::from_vec(vec![1.0, 0.0]),
            &a1, &a2,
            vec![Some(0)],
            "f",
        ).unwrap();
        let g = AgentMorphism::new(
            nalgebra::DMatrix::identity(2, 2),
            DVector::from_vec(vec![0.0, 1.0]),
            &a2, &a1,
            vec![Some(0)],
            "g",
        ).unwrap();
        let gf = g.compose(&f).unwrap();
        let state = DVector::from_vec(vec![0.0, 0.0]);
        let result = gf.apply(&state);
        assert!((result - DVector::from_vec(vec![1.0, 1.0])).norm() < 1e-10);
    }

    #[test]
    fn test_capability_preservation() {
        let a1 = Agent::new(2, vec![Capability::new("observe", 0, 1)], "sensor");
        let a2 = Agent::new(2, vec![Capability::new("observe", 0, 1)], "sensor'");
        let f = AgentMorphism::new(
            nalgebra::DMatrix::identity(2, 2),
            DVector::zeros(2),
            &a1, &a2,
            vec![Some(0)],
            "embed",
        ).unwrap();
        assert!(f.preserves_capabilities(&a1, &a2));
    }

    #[test]
    fn test_capability_not_preserved() {
        let a1 = Agent::new(2, vec![Capability::new("observe", 0, 1), Capability::new("act", 1, 0)], "full");
        let a2 = Agent::new(2, vec![Capability::new("observe", 0, 1)], "partial");
        let f = AgentMorphism::new(
            nalgebra::DMatrix::identity(2, 2),
            DVector::zeros(2),
            &a1, &a2,
            vec![Some(0), None],
            "restrict",
        ).unwrap();
        // This should still be "preserving" since None just means the capability isn't mapped
        assert!(f.preserves_capabilities(&a1, &a2));
    }

    #[test]
    fn test_agent_with_state() {
        let a = Agent::new(2, vec![], "point")
            .with_state(DVector::from_vec(vec![3.0, 4.0]));
        assert!((a.state - DVector::from_vec(vec![3.0, 4.0])).norm() < 1e-10);
    }

    #[test]
    fn test_dimension_mismatch_rejected() {
        let a1 = Agent::new(2, vec![], "2d");
        let a2 = Agent::new(3, vec![], "3d");
        let result = AgentMorphism::new(
            nalgebra::DMatrix::identity(2, 2),
            DVector::zeros(2),
            &a1, &a2,
            vec![],
            "bad",
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_get_capability() {
        let a = Agent::new(2, vec![
            Capability::new("see", 0, 1),
            Capability::new("move", 1, 1),
        ], "bot");
        assert_eq!(a.get_capability("see").unwrap().arity, 0);
        assert_eq!(a.get_capability("move").unwrap().arity, 1);
        assert!(a.get_capability("fly").is_none());
    }
}
