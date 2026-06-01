//! Yoneda Lemma Application: an agent is determined by its relationships
//!
//! The Yoneda lemma states that for any functor F: C → Set and object c ∈ C,
//! Nat(C(c, -), F) ≅ F(c).
//!
//! In our context: an agent A is completely determined by how all other agents
//! relate to it (the morphisms into A). You don't need to know the internals;
//! the pattern of relationships is sufficient.

use crate::agent::{Agent, AgentMorphism, Capability};
use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// A representable functor Hom(-, A) for a fixed agent A.
/// For each agent X, this gives the set of morphisms X → A.
#[derive(Clone, Debug)]
pub struct RepresentableFunctor {
    pub target: Agent,
}

impl RepresentableFunctor {
    pub fn new(target: Agent) -> Self {
        Self { target }
    }

    /// Evaluate Hom(X, A) — compute all valid morphisms from X to A.
    /// In practice, we represent this as the space of valid linear maps.
    pub fn hom(&self, source: &Agent) -> HomSet {
        HomSet {
            source_dim: source.state_dimension,
            target_dim: self.target.state_dimension,
            label: format!("Hom({}, {})", source.label, self.target.label),
        }
    }
}

/// A hom-set Hom(A, B) represented as a vector space of morphisms.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HomSet {
    pub source_dim: usize,
    pub target_dim: usize,
    pub label: String,
}

impl HomSet {
    /// The dimension of this hom-set (as a vector space).
    pub fn dimension(&self) -> usize {
        self.source_dim * self.target_dim
    }

    /// Create a specific morphism from a matrix representation.
    pub fn element(&self, matrix: DMatrix<f64>) -> Option<AgentMorphism> {
        if matrix.nrows() != self.target_dim || matrix.ncols() != self.source_dim {
            return None;
        }
        Some(AgentMorphism {
            source_dim: self.source_dim,
            target_dim: self.target_dim,
            linear_map: matrix,
            offset: DVector::zeros(self.target_dim),
            capability_map: vec![],
            label: "hom_element".into(),
        })
    }
}

/// The Yoneda embedding: maps an agent A to the representable functor Hom(-, A).
/// This is a fully faithful functor, meaning it preserves all structure.
pub struct YonedaEmbedding;

impl YonedaEmbedding {
    /// Embed an agent into the category of presheaves.
    pub fn embed(agent: Agent) -> RepresentableFunctor {
        RepresentableFunctor::new(agent)
    }

    /// Yoneda lemma: Nat(Hom(-, A), F) ≅ F(A)
    /// In our setting, natural transformations from the representable functor
    /// to any presheaf F correspond to elements of F(A).
    pub fn yoneda_lemma<F>(
        agent: &Agent,
        _presheaf: F,
    ) -> YonedaCorrespondence
    where
        F: Fn(&Agent) -> DVector<f64>,
    {
        let element = _presheaf(agent);
        YonedaCorrespondence {
            agent_label: agent.label.clone(),
            presheaf_value: element,
        }
    }

    /// Verify the Yoneda lemma for a simple case.
    /// If two agents have the same hom-sets from all other agents, they're isomorphic.
    pub fn agents_equal_by_relationships(
        a: &Agent,
        b: &Agent,
        test_agents: &[Agent],
    ) -> bool {
        // Check if Hom(X, A) ≅ Hom(X, B) for all test agents X
        let repr_a = RepresentableFunctor::new(a.clone());
        let repr_b = RepresentableFunctor::new(b.clone());

        for x in test_agents {
            let hom_a = repr_a.hom(x);
            let hom_b = repr_b.hom(x);
            // Same dimension means isomorphic as vector spaces
            if hom_a.dimension() != hom_b.dimension() {
                return false;
            }
        }
        true
    }
}

/// The Yoneda correspondence: an element of F(A) ↔ a natural transformation Hom(-, A) → F.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct YonedaCorrespondence {
    pub agent_label: String,
    pub presheaf_value: DVector<f64>,
}

impl YonedaCorrespondence {
    /// The dimension of the correspondence (should match the presheaf value).
    pub fn dimension(&self) -> usize {
        self.presheaf_value.len()
    }
}

/// Application: agent profiling through Yoneda.
/// An agent's "personality" is the natural transformation from its representable functor
/// to some profiling presheaf. Two agents with the same profiles are "the same" categorically.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentProfile {
    pub agent_label: String,
    /// Feature vector extracted via the Yoneda correspondence
    pub features: DVector<f64>,
}

impl AgentProfile {
    /// Profile an agent using the Yoneda lemma approach.
    pub fn profile(agent: &Agent, observers: &[Agent]) -> Self {
        let mut features = Vec::new();
        for obs in observers {
            // The "observation" is the dimension of the hom-space
            let hom_dim = obs.state_dimension * agent.state_dimension;
            features.push(hom_dim as f64);
        }
        Self {
            agent_label: agent.label.clone(),
            features: DVector::from_vec(features),
        }
    }

    /// Compare two agent profiles.
    pub fn similarity(&self, other: &AgentProfile) -> f64 {
        if self.features.len() != other.features.len() {
            return 0.0;
        }
        let diff = &self.features - &other.features;
        let norm = diff.norm();
        if norm < 1e-10 {
            1.0
        } else {
            1.0 / (1.0 + norm)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_representable_functor() {
        let a = Agent::new(3, vec![Capability::new("act", 1, 1)], "target");
        let repr = RepresentableFunctor::new(a);
        let source = Agent::new(2, vec![], "source");
        let hom = repr.hom(&source);
        assert_eq!(hom.dimension(), 6); // 2 × 3
    }

    #[test]
    fn test_hom_set_element() {
        let hom = HomSet {
            source_dim: 2,
            target_dim: 2,
            label: "test".into(),
        };
        let m = DMatrix::identity(2, 2);
        let elem = hom.element(m).unwrap();
        let v = DVector::from_vec(vec![1.0, 2.0]);
        let result = elem.apply(&v);
        assert!((result - v).norm() < 1e-10);
    }

    #[test]
    fn test_hom_set_rejects_wrong_dim() {
        let hom = HomSet {
            source_dim: 2,
            target_dim: 3,
            label: "test".into(),
        };
        let m = DMatrix::identity(2, 2);
        assert!(hom.element(m).is_none());
    }

    #[test]
    fn test_yoneda_embedding() {
        let a = Agent::new(4, vec![Capability::new("see", 0, 2)], "observer");
        let repr = YonedaEmbedding::embed(a.clone());
        let source = Agent::new(4, vec![], "source");
        let hom = repr.hom(&source);
        assert_eq!(hom.dimension(), 16);
    }

    #[test]
    fn test_yoneda_lemma() {
        let a = Agent::new(2, vec![], "test");
        let presheaf = |agent: &Agent| DVector::from_vec(vec![agent.state_dimension as f64]);
        let corr = YonedaEmbedding::yoneda_lemma(&a, presheaf);
        assert_eq!(corr.dimension(), 1);
        assert!((corr.presheaf_value[(0, 0)] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_agents_equal_by_relationships() {
        let a = Agent::new(2, vec![Capability::new("x", 0, 1)], "A");
        let b = Agent::new(2, vec![Capability::new("y", 0, 1)], "B");
        let probe = Agent::new(3, vec![], "probe");
        // Same dimensions → same hom-space dimensions → "equal" by Yoneda
        assert!(YonedaEmbedding::agents_equal_by_relationships(&a, &b, &[probe]));
    }

    #[test]
    fn test_agents_not_equal_by_relationships() {
        let a = Agent::new(2, vec![], "A");
        let b = Agent::new(4, vec![], "B");
        let probe = Agent::new(3, vec![], "probe");
        // Different dimensions → different hom-space dimensions
        assert!(!YonedaEmbedding::agents_equal_by_relationships(&a, &b, &[probe]));
    }

    #[test]
    fn test_agent_profile() {
        let a = Agent::new(3, vec![], "agent");
        let obs1 = Agent::new(2, vec![], "obs1");
        let obs2 = Agent::new(4, vec![], "obs2");
        let profile = AgentProfile::profile(&a, &[obs1, obs2]);
        assert_eq!(profile.features.len(), 2);
        assert!((profile.features[(0, 0)] - 6.0).abs() < 1e-10); // 2 * 3
        assert!((profile.features[(1, 0)] - 12.0).abs() < 1e-10); // 4 * 3
    }

    #[test]
    fn test_profile_similarity_identical() {
        let p1 = AgentProfile {
            agent_label: "a".into(),
            features: DVector::from_vec(vec![1.0, 2.0, 3.0]),
        };
        let p2 = AgentProfile {
            agent_label: "b".into(),
            features: DVector::from_vec(vec![1.0, 2.0, 3.0]),
        };
        assert!((p1.similarity(&p2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_profile_similarity_different() {
        let p1 = AgentProfile {
            agent_label: "a".into(),
            features: DVector::from_vec(vec![1.0, 0.0]),
        };
        let p2 = AgentProfile {
            agent_label: "b".into(),
            features: DVector::from_vec(vec![0.0, 1.0]),
        };
        let sim = p1.similarity(&p2);
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_hom_set_label() {
        let a = Agent::new(2, vec![], "A");
        let b = Agent::new(3, vec![], "B");
        let repr = RepresentableFunctor::new(b);
        let hom = repr.hom(&a);
        assert!(hom.label.contains("A"));
        assert!(hom.label.contains("B"));
    }
}
