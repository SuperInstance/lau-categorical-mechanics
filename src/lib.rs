//! # Lau Categorical Mechanics
//!
//! Unifies symplectic mechanics and categorical agents using the language of category theory.
//!
//! **Key insight:** Hamiltonian mechanics is a functor from the category of symplectic manifolds
//! to the category of agent behaviors. Conservation laws are natural transformations.

pub mod symplectic;
pub mod agent;
pub mod functor;
pub mod natural;
pub mod presheaf;
pub mod yoneda;
pub mod adjunction;
pub mod monoidal;
pub mod compact;
pub mod application;

pub use symplectic::*;
pub use agent::*;
pub use functor::*;
pub use natural::*;
pub use presheaf::*;
pub use yoneda::*;
pub use adjunction::*;
pub use monoidal::*;
pub use compact::*;
pub use application::*;
