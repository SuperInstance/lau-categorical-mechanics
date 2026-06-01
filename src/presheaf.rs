//! Presheaf of Observables: contravariant functor from agent states to measurements
//!
//! An observable is a measurement we can make on an agent's state. The presheaf
//! structure captures how observations pull back along state transformations:
//! if we have a map f: A → B, then observing B pulls back to observing A.

use crate::agent::{Agent, AgentMorphism};
use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// A measurement type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MeasurementType {
    Scalar,
    Vector(usize),
    Matrix(usize, usize),
}

/// An observable: a function from agent states to measurements.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Observable {
    pub label: String,
    pub measurement_type: MeasurementType,
    /// Linear map from state space to measurement space
    pub measurement_map: DMatrix<f64>,
    /// Offset
    pub offset: DVector<f64>,
}

impl Observable {
    pub fn scalar(map: DMatrix<f64>, label: impl Into<String>) -> Self {
        assert_eq!(map.nrows(), 1);
        Self {
            label: label.into(),
            measurement_type: MeasurementType::Scalar,
            measurement_map: map,
            offset: DVector::zeros(1),
        }
    }

    pub fn vector(map: DMatrix<f64>, dim: usize, label: impl Into<String>) -> Self {
        assert_eq!(map.nrows(), dim);
        Self {
            label: label.into(),
            measurement_type: MeasurementType::Vector(dim),
            measurement_map: map,
            offset: DVector::zeros(dim),
        }
    }

    /// Observe/measure a state.
    pub fn observe(&self, state: &DVector<f64>) -> DVector<f64> {
        &self.measurement_map * state + &self.offset
    }

    /// Pull back this observable along a morphism f: A → B.
    /// Given obs: B → Measurement, produce obs ∘ f: A → Measurement.
    /// This is the contravariant action of the presheaf.
    pub fn pull_back(&self, f: &AgentMorphism) -> Observable {
        let new_map = &self.measurement_map * &f.linear_map;
        let new_offset = &self.measurement_map * &f.offset + &self.offset;
        Observable {
            label: format!("{}∘{}", self.label, f.label),
            measurement_type: self.measurement_type.clone(),
            measurement_map: new_map,
            offset: new_offset,
        }
    }
}

/// The presheaf of observables on a category of agents.
/// This is a contravariant functor: Obs: Agent^op → Vect
pub struct ObservablePresheaf {
    pub observables: Vec<Observable>,
}

impl ObservablePresheaf {
    pub fn new(observables: Vec<Observable>) -> Self {
        Self { observables }
    }

    /// The contravariant functorial action: given f: A → B, produce Obs(f): Obs(B) → Obs(A).
    pub fn functorial_action(&self, f: &AgentMorphism) -> ObservablePresheaf {
        let pulled: Vec<Observable> = self.observables.iter()
            .map(|obs| obs.pull_back(f))
            .collect();
        ObservablePresheaf::new(pulled)
    }

    /// Verify contravariant functoriality: pulling back along id = identity, and
    /// pulling back along g ∘ f = (pull back along f) then (pull back along g).
    pub fn verify_contravariant(
        &self,
        f: &AgentMorphism,
        g: &AgentMorphism,
    ) -> bool {
        // Pull back along f, then along g
        let pulled_f = self.functorial_action(f);
        let pulled_gf = pulled_f.functorial_action(g);

        // Pull back along g ∘ f
        let gf = match g.compose(f) {
            Some(gf) => gf,
            None => return false,
        };
        let pulled_gf_direct = self.functorial_action(&gf);

        // They should be the same
        if pulled_gf.observables.len() != pulled_gf_direct.observables.len() {
            return false;
        }
        for (a, b) in pulled_gf.observables.iter().zip(pulled_gf_direct.observables.iter()) {
            if (a.measurement_map.clone() - &b.measurement_map).norm() > 1e-8 {
                return false;
            }
            if (a.offset.clone() - &b.offset).norm() > 1e-8 {
                return false;
            }
        }
        true
    }

    /// Verify identity: pulling back along identity does nothing.
    pub fn verify_identity(&self, id: &AgentMorphism) -> bool {
        let pulled = self.functorial_action(id);
        for (a, b) in self.observables.iter().zip(pulled.observables.iter()) {
            if (a.measurement_map.clone() - &b.measurement_map).norm() > 1e-8 {
                return false;
            }
            if (a.offset.clone() - &b.offset).norm() > 1e-8 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_observable() {
        let obs = Observable::scalar(
            DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
            "position",
        );
        let state = DVector::from_vec(vec![3.0, 5.0]);
        let result = obs.observe(&state);
        assert!((result[(0, 0)] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_observable() {
        let obs = Observable::vector(
            DMatrix::identity(2, 2),
            2,
            "full_state",
        );
        let state = DVector::from_vec(vec![1.0, 2.0]);
        let result = obs.observe(&state);
        assert!((result - state).norm() < 1e-10);
    }

    #[test]
    fn test_pull_back_identity() {
        let obs = Observable::vector(
            DMatrix::identity(2, 2),
            2,
            "state",
        );
        let a = Agent::new(2, vec![], "a");
        let id = AgentMorphism::identity(&a);
        let pulled = obs.pull_back(&id);
        let state = DVector::from_vec(vec![3.0, 4.0]);
        assert!((pulled.observe(&state) - obs.observe(&state)).norm() < 1e-10);
    }

    #[test]
    fn test_pull_back_with_offset() {
        let obs = Observable::vector(
            DMatrix::identity(2, 2),
            2,
            "state",
        );
        let a1 = Agent::new(2, vec![], "a1");
        let a2 = Agent::new(2, vec![], "a2");
        let f = AgentMorphism::new(
            DMatrix::identity(2, 2),
            DVector::from_vec(vec![10.0, 20.0]),
            &a1, &a2, vec![], "shift",
        ).unwrap();
        let pulled = obs.pull_back(&f);
        let state = DVector::from_vec(vec![1.0, 2.0]);
        let result = pulled.observe(&state);
        // obs(f(x)) = f(x) = x + offset
        assert!((result - DVector::from_vec(vec![11.0, 22.0])).norm() < 1e-10);
    }

    #[test]
    fn test_presheaf_contravariant_identity() {
        let obs = ObservablePresheaf::new(vec![
            Observable::vector(DMatrix::identity(2, 2), 2, "state"),
        ]);
        let a = Agent::new(2, vec![], "a");
        let id = AgentMorphism::identity(&a);
        assert!(obs.verify_identity(&id));
    }

    #[test]
    fn test_presheaf_contravariant_composition() {
        let obs = ObservablePresheaf::new(vec![
            Observable::vector(DMatrix::identity(2, 2), 2, "state"),
        ]);
        let a1 = Agent::new(2, vec![], "a1");
        let a2 = Agent::new(2, vec![], "a2");
        let a3 = Agent::new(2, vec![], "a3");
        let f = AgentMorphism::new(
            DMatrix::identity(2, 2),
            DVector::from_vec(vec![1.0, 0.0]),
            &a1, &a2, vec![], "f",
        ).unwrap();
        let g = AgentMorphism::new(
            DMatrix::identity(2, 2),
            DVector::from_vec(vec![0.0, 1.0]),
            &a2, &a3, vec![], "g",
        ).unwrap();
        assert!(obs.verify_contravariant(&f, &g));
    }

    #[test]
    fn test_position_observable() {
        // Observe only position (first half of state)
        let dim = 4;
        let n = 2;
        let mut pos_map = DMatrix::zeros(n, dim);
        for i in 0..n {
            pos_map[(i, i)] = 1.0;
        }
        let obs = Observable::vector(pos_map, 2, "position");
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = obs.observe(&state);
        assert!((result[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((result[(1, 0)] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_momentum_observable() {
        // Observe only momentum (second half of state)
        let dim = 4;
        let n = 2;
        let mut mom_map = DMatrix::zeros(n, dim);
        for i in 0..n {
            mom_map[(i, n + i)] = 1.0;
        }
        let obs = Observable::vector(mom_map, 2, "momentum");
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = obs.observe(&state);
        assert!((result[(0, 0)] - 3.0).abs() < 1e-10);
        assert!((result[(1, 0)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_observable_with_offset() {
        let mut obs = Observable::scalar(
            DMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
            "sum",
        );
        obs.offset = DVector::from_vec(vec![-1.0]);
        let state = DVector::from_vec(vec![3.0, 4.0]);
        let result = obs.observe(&state);
        assert!((result[(0, 0)] - 6.0).abs() < 1e-10);
    }
}
