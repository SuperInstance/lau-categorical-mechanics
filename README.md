# lau-categorical-mechanics

> Categorical foundations unifying symplectic mechanics and agent behaviors

## What This Does

Categorical foundations unifying symplectic mechanics and agent behaviors. Part of the PLATO/LAU ecosystem — a mathematically rigorous framework for building educational agents that learn, teach, and evolve.

## The Key Idea

This crate implements the core abstractions needed for its domain, with a focus on correctness, composability, and conservation guarantees. Every public type is serializable (serde), every algorithm is tested, and every invariant is verified.

## Install

```bash
cargo add lau-categorical-mechanics
```

## Quick Start

See the API Reference below for complete usage. Key entry points:

```rust
use lau_categorical_mechanics::*;
// See types and methods below for complete usage
```

## API Reference

```rust
pub struct HamiltonianFunctor;
pub struct Hamiltonian 
    pub fn quadratic(matrix: DMatrix<f64>, label: impl Into<String>) -> Self 
    pub fn evaluate(&self, x: &DVector<f64>) -> f64 
    pub fn vector_field(&self, x: &DVector<f64>, omega: &DMatrix<f64>) -> Option<DVector<f64>> 
    pub fn map_object(manifold: &SymplecticManifold) -> Agent 
    pub fn map_morphism(
    pub fn verify_functoriality_identity(manifold: &SymplecticManifold) -> bool 
    pub fn verify_functoriality_composition(
pub struct LauSystem 
    pub fn new(label: impl Into<String>) -> Self 
    pub fn add_physical_agent(&mut self, manifold: &SymplecticManifold) -> &Agent 
    pub fn parallel_compose(&self, i: usize, j: usize) -> Option<MonoidalAgent> 
    pub fn observe(&self, agent_idx: usize, probe_direction: &DVector<f64>) -> Option<f64> 
    pub fn profile_all(&self) -> Vec<AgentProfile> 
    pub fn energy(&self, agent_idx: usize, hamiltonian: &Hamiltonian) -> Option<f64> 
pub fn sample_system() -> LauSystem 
pub fn categorical_pipeline() -> PipelineResult 
pub struct PipelineResult 
pub struct Capability 
    pub fn new(name: impl Into<String>, arity: usize, codimension: usize) -> Self 
pub struct Agent 
    pub fn new(
    pub fn with_state(mut self, state: DVector<f64>) -> Self 
    pub fn has_capability(&self, name: &str) -> bool 
    pub fn get_capability(&self, name: &str) -> Option<&Capability> 
pub struct AgentMorphism 
    pub fn new(
    pub fn identity(agent: &Agent) -> Self 
    pub fn apply(&self, state: &DVector<f64>) -> DVector<f64> 
    pub fn compose(&self, other: &AgentMorphism) -> Option<AgentMorphism> 
    pub fn preserves_capabilities(&self, source: &Agent, target: &Agent) -> bool 
pub struct AgentCategory;
    pub fn verify_associativity(
    pub fn verify_identity_law(
pub struct DualAgent 
    pub fn dual(agent: &Agent) -> Self 
    pub fn double_dual(agent: &Agent) -> Self 
pub struct Cup 
    pub fn new(agent: &Agent) -> Self 
    pub fn state(&self) -> DVector<f64> 
pub struct Cap 
    pub fn new(agent: &Agent) -> Self 
    pub fn evaluate(&self, state: &DVector<f64>) -> f64 
pub struct CompactClosed;
    pub fn verify_snake_identity(agent: &Agent) -> bool 
    pub fn trace(agent: &Agent, f: &AgentMorphism) -> f64 
    pub fn verify_double_dual(agent: &Agent) -> bool 
    pub fn compact_compose(
pub struct ObservationProbe 
    pub fn new(agent: &Agent, direction: &DVector<f64>) -> Self 
    pub fn measure(&self, agent_state: &DVector<f64>) -> f64 
    pub fn dual_probe(&self) -> ObservationProbe 
pub struct StateSpace 
    pub fn new(dimension: usize, label: impl Into<String>) -> Self 
    pub fn from_symplectic(manifold: &SymplecticManifold) -> Self 
pub struct StateMorphism 
    pub fn identity(space: &StateSpace) -> Self 
    pub fn compose(&self, other: &StateMorphism) -> Option<StateMorphism> 
pub struct ObservationFunctor 
```

## How It Works

Read the source in `src/` for full implementation details. All algorithms are documented with inline comments explaining the mathematical foundations.

## The Math

This crate implements formal mathematical constructs. See the source documentation for theorem statements and proofs of correctness.

## Testing

**107 tests** covering construction, serialization, correctness properties, edge cases, and composability with other lau-* crates.

## License

MIT
