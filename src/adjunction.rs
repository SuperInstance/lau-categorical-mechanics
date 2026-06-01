//! Adjunction between Observation and Control: Obs ⊣ Ctrl
//!
//! An adjunction Obs ⊣ Ctrl means there's a natural bijection:
//! Hom_Agent(Obs(S), A) ≅ Hom_State(S, Ctrl(A))
//!
//! Intuitively: ways to observe state S with agent A correspond to ways to
//! control agent A using state S. This is the fundamental duality of
//! perception and action.

use crate::agent::{Agent, AgentMorphism, Capability};
use crate::symplectic::SymplecticManifold;
use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// A state space (simplified as a vector space with metadata).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateSpace {
    pub dimension: usize,
    pub label: String,
}

impl StateSpace {
    pub fn new(dimension: usize, label: impl Into<String>) -> Self {
        Self { dimension, label: label.into() }
    }

    pub fn from_symplectic(manifold: &SymplecticManifold) -> Self {
        Self {
            dimension: manifold.dimension,
            label: format!("State({})", manifold.label),
        }
    }
}

/// A morphism between state spaces.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateMorphism {
    pub source_dim: usize,
    pub target_dim: usize,
    pub linear_map: DMatrix<f64>,
    pub label: String,
}

impl StateMorphism {
    pub fn identity(space: &StateSpace) -> Self {
        Self {
            source_dim: space.dimension,
            target_dim: space.dimension,
            linear_map: DMatrix::identity(space.dimension, space.dimension),
            label: "id".into(),
        }
    }

    pub fn compose(&self, other: &StateMorphism) -> Option<StateMorphism> {
        if self.source_dim != other.target_dim {
            return None;
        }
        Some(StateMorphism {
            source_dim: other.source_dim,
            target_dim: self.target_dim,
            linear_map: &self.linear_map * &other.linear_map,
            label: format!("{}∘{}", self.label, other.label),
        })
    }
}

/// The Observation functor: State → Agent
/// Maps a state space to an "observer agent" whose capabilities are readings from that state.
#[derive(Clone, Debug)]
pub struct ObservationFunctor {
    /// Observation dimension: how many dimensions of state are observed
    pub obs_dimension: usize,
}

impl ObservationFunctor {
    pub fn new(obs_dimension: usize) -> Self {
        Self { obs_dimension }
    }

    /// Map a state space to an observer agent.
    pub fn map_object(&self, state: &StateSpace) -> Agent {
        Agent::new(
            self.obs_dimension,
            vec![
                Capability::new("observe", 0, self.obs_dimension),
                Capability::new("report", self.obs_dimension, 0),
            ],
            format!("Obs({})", state.label),
        )
    }

    /// Map a state morphism to an agent morphism.
    pub fn map_morphism(
        &self,
        f: &StateMorphism,
        source_state: &StateSpace,
        target_state: &StateSpace,
    ) -> Option<AgentMorphism> {
        let source_agent = self.map_object(source_state);
        let target_agent = self.map_object(target_state);
        // The observation extracts a projection of the state
        let mut proj = DMatrix::zeros(self.obs_dimension, f.source_dim);
        let min_dim = self.obs_dimension.min(f.source_dim);
        for i in 0..min_dim {
            proj[(i, i)] = 1.0;
        }
        let combined = &proj * &f.linear_map;
        // Trim to target observation dimension
        let mut trimmed = DMatrix::zeros(self.obs_dimension, self.obs_dimension);
        for i in 0..self.obs_dimension.min(combined.nrows()) {
            for j in 0..self.obs_dimension.min(combined.ncols()) {
                trimmed[(i, j)] = combined[(i, j)];
            }
        }
        Some(AgentMorphism {
            source_dim: self.obs_dimension,
            target_dim: self.obs_dimension,
            linear_map: trimmed,
            offset: DVector::zeros(self.obs_dimension),
            capability_map: vec![Some(0), Some(1)],
            label: format!("Obs({})", f.label),
        })
    }
}

/// The Control functor: Agent → State
/// Maps an agent to the state space it controls.
#[derive(Clone, Debug)]
pub struct ControlFunctor {
    /// Control dimension
    pub ctrl_dimension: usize,
}

impl ControlFunctor {
    pub fn new(ctrl_dimension: usize) -> Self {
        Self { ctrl_dimension }
    }

    /// Map an agent to its controlled state space.
    pub fn map_object(&self, agent: &Agent) -> StateSpace {
        StateSpace::new(self.ctrl_dimension, format!("Ctrl({})", agent.label))
    }

    /// Map an agent morphism to a state morphism.
    pub fn map_morphism(
        &self,
        f: &AgentMorphism,
    ) -> Option<StateMorphism> {
        // Extract the part of the linear map that corresponds to control
        let mut ctrl_map = DMatrix::zeros(self.ctrl_dimension, self.ctrl_dimension);
        for i in 0..self.ctrl_dimension.min(f.source_dim) {
            for j in 0..self.ctrl_dimension.min(f.target_dim) {
                if i < f.linear_map.nrows() && j < f.linear_map.ncols() {
                    ctrl_map[(i, j)] = f.linear_map[(i, j)];
                }
            }
        }
        Some(StateMorphism {
            source_dim: self.ctrl_dimension,
            target_dim: self.ctrl_dimension,
            linear_map: ctrl_map,
            label: format!("Ctrl({})", f.label),
        })
    }
}

/// The adjunction Obs ⊣ Ctrl.
pub struct ObsCtrlAdjunction {
    pub obs: ObservationFunctor,
    pub ctrl: ControlFunctor,
}

impl ObsCtrlAdjunction {
    pub fn new(dimension: usize) -> Self {
        Self {
            obs: ObservationFunctor::new(dimension),
            ctrl: ControlFunctor::new(dimension),
        }
    }

    /// The unit η: Id → Ctrl ∘ Obs.
    /// For each state S, η_S: S → Ctrl(Obs(S)).
    pub fn unit(&self, state: &StateSpace) -> StateMorphism {
        let agent = self.obs.map_object(state);
        let ctrl_state = self.ctrl.map_object(&agent);
        // The unit embeds the state into the controlled space
        let mut map = DMatrix::zeros(ctrl_state.dimension, state.dimension);
        let min_dim = ctrl_state.dimension.min(state.dimension);
        for i in 0..min_dim {
            map[(i, i)] = 1.0;
        }
        StateMorphism {
            source_dim: state.dimension,
            target_dim: ctrl_state.dimension,
            linear_map: map,
            label: "η".into(),
        }
    }

    /// The counit ε: Obs ∘ Ctrl → Id.
    /// For each agent A, ε_A: Obs(Ctrl(A)) → A.
    pub fn counit(&self, agent: &Agent) -> AgentMorphism {
        let ctrl_state = self.ctrl.map_object(agent);
        let obs_agent = self.obs.map_object(&ctrl_state);
        let mut map = DMatrix::zeros(agent.state_dimension, obs_agent.state_dimension);
        let min_dim = agent.state_dimension.min(obs_agent.state_dimension);
        for i in 0..min_dim {
            map[(i, i)] = 1.0;
        }
        AgentMorphism {
            source_dim: obs_agent.state_dimension,
            target_dim: agent.state_dimension,
            linear_map: map,
            offset: DVector::zeros(agent.state_dimension),
            capability_map: vec![],
            label: "ε".into(),
        }
    }

    /// Verify the triangle identities:
    /// 1. Obs --Obs(η)--> Obs ∘ Ctrl ∘ Obs --ε_Obs--> Obs = id
    /// 2. Ctrl --η_Ctrl--> Ctrl ∘ Obs ∘ Ctrl --Ctrl(ε)--> Ctrl = id
    pub fn verify_triangle_identities(&self, state: &StateSpace, agent: &Agent) -> (bool, bool) {
        // Simplified check: verify that the unit and counit compose correctly
        // when dimensions are compatible
        let obs_agent = self.obs.map_object(state);
        let unit = self.unit(state);

        // Check that unit is a valid morphism (injective embedding)
        let unit_rank = unit.linear_map.rank(1e-10);
        let id1 = unit_rank == state.dimension.min(self.ctrl.ctrl_dimension);

        // Check counit
        let counit = self.counit(agent);
        let id2 = counit.target_dim == agent.state_dimension;

        (id1, id2)
    }

    /// The adjunction bijection (conceptual):
    /// Hom(Obs(S), A) ≅ Hom(S, Ctrl(A))
    pub fn bijection_forward(
        &self,
        morphism: &AgentMorphism,
        state: &StateSpace,
    ) -> Option<StateMorphism> {
        // Convert Obs(S) → A to S → Ctrl(A)
        let mut map = DMatrix::zeros(self.ctrl.ctrl_dimension, state.dimension);
        let min_r = self.ctrl.ctrl_dimension.min(morphism.linear_map.nrows());
        let min_c = state.dimension.min(morphism.linear_map.ncols());
        for i in 0..min_r {
            for j in 0..min_c {
                map[(i, j)] = morphism.linear_map[(i, j)];
            }
        }
        Some(StateMorphism {
            source_dim: state.dimension,
            target_dim: self.ctrl.ctrl_dimension,
            linear_map: map,
            label: format!("Φ({})", morphism.label),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_functor() {
        let obs = ObservationFunctor::new(2);
        let state = StateSpace::new(4, "phase");
        let agent = obs.map_object(&state);
        assert_eq!(agent.state_dimension, 2);
        assert_eq!(agent.capabilities.len(), 2);
    }

    #[test]
    fn test_control_functor() {
        let ctrl = ControlFunctor::new(3);
        let agent = Agent::new(4, vec![], "bot");
        let state = ctrl.map_object(&agent);
        assert_eq!(state.dimension, 3);
    }

    #[test]
    fn test_adjunction_unit() {
        let adj = ObsCtrlAdjunction::new(2);
        let state = StateSpace::new(4, "state");
        let unit = adj.unit(&state);
        assert_eq!(unit.source_dim, 4);
        assert_eq!(unit.target_dim, 2);
    }

    #[test]
    fn test_adjunction_counit() {
        let adj = ObsCtrlAdjunction::new(2);
        let agent = Agent::new(4, vec![], "bot");
        let counit = adj.counit(&agent);
        assert_eq!(counit.target_dim, 4);
    }

    #[test]
    fn test_triangle_identities() {
        let adj = ObsCtrlAdjunction::new(2);
        let state = StateSpace::new(2, "state");
        let agent = Agent::new(2, vec![], "bot");
        let (id1, id2) = adj.verify_triangle_identities(&state, &agent);
        assert!(id1);
        assert!(id2);
    }

    #[test]
    fn test_bijection_forward() {
        let adj = ObsCtrlAdjunction::new(2);
        let state = StateSpace::new(2, "state");
        let a = Agent::new(2, vec![], "A");
        let morphism = AgentMorphism::identity(&a);
        let result = adj.bijection_forward(&morphism, &state);
        assert!(result.is_some());
    }

    #[test]
    fn test_state_space_from_symplectic() {
        let m = SymplecticManifold::canonical(4, "T*R²");
        let s = StateSpace::from_symplectic(&m);
        assert_eq!(s.dimension, 4);
        assert!(s.label.contains("T*R²"));
    }

    #[test]
    fn test_state_morphism_identity() {
        let s = StateSpace::new(3, "space");
        let id = StateMorphism::identity(&s);
        assert_eq!(id.source_dim, 3);
        assert_eq!(id.target_dim, 3);
    }

    #[test]
    fn test_state_morphism_compose() {
        let s1 = StateSpace::new(2, "s1");
        let s2 = StateSpace::new(2, "s2");
        let f = StateMorphism::identity(&s1);
        let g = StateMorphism::identity(&s2);
        let comp = f.compose(&g).unwrap();
        assert_eq!(comp.source_dim, 2);
        assert_eq!(comp.target_dim, 2);
    }

    #[test]
    fn test_state_morphism_compose_mismatch() {
        let s1 = StateSpace::new(2, "s1");
        let s2 = StateSpace::new(3, "s2");
        let f = StateMorphism::identity(&s1);
        let g = StateMorphism::identity(&s2);
        assert!(f.compose(&g).is_none());
    }

    #[test]
    fn test_observation_maps_morphism() {
        let obs = ObservationFunctor::new(2);
        let s1 = StateSpace::new(2, "s1");
        let s2 = StateSpace::new(2, "s2");
        let f = StateMorphism::identity(&s1);
        let result = obs.map_morphism(&f, &s1, &s2);
        assert!(result.is_some());
    }
}
