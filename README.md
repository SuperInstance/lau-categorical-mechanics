# lau-categorical-mechanics

**Categorical foundations unifying symplectic mechanics and agent behaviors.**

[![Tests](https://img.shields.io/badge/tests-107-passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)]()

---

## What This Does

Hamiltonian mechanics has a deep categorical structure: symplectic manifolds form a category, Hamiltonian flows are functors, and conservation laws are natural transformations. This crate makes that structure explicit and computational.

The central idea: **the Hamiltonian functor** maps the category of symplectic manifolds (Sympl) to the category of agents. A phase space becomes an agent; a symplectomorphism becomes a capability-preserving morphism; a conserved quantity becomes a natural transformation.

You can:
- **Build symplectic manifolds** with canonical forms and verify symplectomorphisms
- **Define agents** with state spaces and named capabilities
- **Apply the Hamiltonian functor** to turn physics into agent behavior
- **Express conservation laws** as natural transformations and verify naturality squares
- **Use the Yoneda lemma** to profile agents by their relationships
- **Set up adjunctions** between observation and control functors
- **Compose agents in parallel** via a monoidal (tensor) product with braiding
- **Work with compact closed categories** for dual agents and evaluation/coevaluation
- **Run a full categorical pipeline** that verifies all structure at once

---

## Key Idea

| Classical Mechanics | Categorical Translation |
|---|---|
| Symplectic manifold (M, ω) | Object in **Sympl** |
| Canonical transformation | Symplectomorphism (morphism in Sympl) |
| Hamiltonian flow | **Hamiltonian functor** F: Sympl → Agent |
| Conserved quantity (Noether) | **Natural transformation** η: F → G |
| Observable | **Presheaf** (contravariant functor on Agent) |
| Agent equivalence | **Yoneda lemma** (determined by relationships) |
| Parallel composition | **Monoidal tensor** A ⊗ B |
| Observation ↔ Control | **Adjunction** Obs ⊣ Ctrl |
| Dual agent (input ↔ output) | **Compact closed** structure |

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-categorical-mechanics = { git = "https://github.com/SuperInstance/lau-categorical-mechanics" }
```

**Dependencies:** `serde` (with `derive`), `serde_json`, `nalgebra` (with `serde-serialize`).

---

## Quick Start

### Create a Symplectic Manifold

```rust
use lau_categorical_mechanics::SymplecticManifold;

// Phase space T*R² (4-dimensional: 2 position + 2 momentum)
let phase = SymplecticManifold::canonical(4, "T*R²");
assert_eq!(phase.config_dimension(), 2); // 2 position coordinates

// Symplectic pairing: ⟨q, p⟩_ω
let q = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
let p = DVector::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
let pairing = phase.symplectic_pairing(&q, &p);
```

### Hamiltonian Functor: Physics → Agents

```rust
use lau_categorical_mechanics::{HamiltonianFunctor, Hamiltonian};

// Map a symplectic manifold to an agent
let agent = HamiltonianFunctor::map_object(&phase);
// agent has capabilities: observe_position, observe_momentum, hamiltonian_flow

// Define a harmonic oscillator H = ½(q² + p²)
let h = Hamiltonian::quadratic(DMatrix::identity(2, 2), "oscillator");
let energy = h.evaluate(&DVector::from_vec(vec![1.0, 0.0])); // = 0.5

// Compute Hamiltonian vector field X_H = ω⁻¹∇H
let xh = h.vector_field(&state, &phase.omega);
```

### Conservation Laws as Natural Transformations

```rust
use lau_categorical_mechanics::{NoetherTheorem, ConservationLaw};

// Energy conservation via Noether's theorem
let energy = NoetherTheorem::energy(&manifold);
let conserved = energy.verify(&state1, &state2); // same energy after flow

// Linear and angular momentum
let mom = NoetherTheorem::linear_momentum(&manifold);
let ang = NoetherTheorem::angular_momentum(&manifold, center);
```

### Agent Profiling via Yoneda

```rust
use lau_categorical_mechanics::{YonedaEmbedding, AgentProfile};

// An agent is determined by how others relate to it
let profile = AgentProfile::profile(&agent, &observers);
let similarity = profile.similarity(&other_profile); // 0.0 to 1.0

// Two agents with same relationship structure are categorically equivalent
let equal = YonedaEmbedding::agents_equal_by_relationships(&a, &b, &probes);
```

### Monoidal Composition (Parallel Agents)

```rust
use lau_categorical_mechanics::{MonoidalAgent, Braiding};

// Compose two agents in parallel: A ⊗ B
let combined = MonoidalAgent::tensor(&agent_a, &agent_b);
assert_eq!(combined.dimension(), a_dim + b_dim);

// Entangled composition
let entangled = MonoidalAgent::tensor_entangled(&a, &b, entanglement_matrix);

// Swap: σ_{A,B}: A ⊗ B → B ⊗ A
let sigma = Braiding::swap(a_dim, b_dim);
let swapped = sigma.apply(&combined_state);
```

### Full Categorical Pipeline

```rust
use lau_categorical_mechanics::categorical_pipeline;

let result = categorical_pipeline();
// Verifies: functoriality, naturality, Yoneda, adjunction triangle identities,
// snake identities (compact closed), conservation laws, serialization
assert!(result.functoriality_verified);
assert!(result.presheaf_verified);
assert!(result.snake_identity);
```

---

## API Reference

### Symplectic Category (`symplectic`)

| Type | Description |
|---|---|
| `SymplecticManifold` | Even-dimensional vector space with canonical symplectic form ω |
| `Symplectomorphism` | Morphism preserving ω (A^T ω₂ A = ω₁) |

```rust
SymplecticManifold::canonical(dim, label)   // ω = [[0,I],[-I,0]]
SymplecticManifold::from_omega(ω, label)    // Custom form (validates antisymmetry)
m.config_dimension()                        // dim/2
m.symplectic_pairing(&u, &v)                // u^T ω v

Symplectomorphism::new(A, src, tgt, label)  // Validates A^T ω₂ A = ω₁
Symplectomorphism::identity(&m)
sym.compose(&other)                         // g ∘ f
sym.inverse()
sym.apply(&point)
```

### Agent Category (`agent`)

| Type | Description |
|---|---|
| `Agent` | State space + capabilities |
| `AgentMorphism` | Capability-preserving linear map between agents |
| `Capability` | Named action with arity and output dimension |

```rust
Agent::new(dim, capabilities, label)
agent.has_capability("observe")
agent.get_capability("act")

AgentMorphism::new(matrix, offset, src, tgt, cap_map, label)
AgentMorphism::identity(&agent)
morph.apply(&state)
morph.compose(&other)                       // g ∘ f
morph.inverse()
```

### Hamiltonian Functor (`functor`)

| Type | Description |
|---|---|
| `HamiltonianFunctor` | F: Sympl → Agent |
| `Hamiltonian` | H(x) = ½x^TQx + c^Tx, with vector field X_H = ω⁻¹∇H |

```rust
HamiltonianFunctor::map_object(&manifold)           // → Agent
HamiltonianFunctor::map_morphism(&sym, &a, &b)      // → AgentMorphism
HamiltonianFunctor::verify_functoriality_identity(&m)    // F(id) = id
HamiltonianFunctor::verify_functoriality_composition(..) // F(g∘f) = F(g)∘F(f)

Hamiltonian::quadratic(Q, label)
h.evaluate(&x)                                      // H(x)
h.vector_field(&x, &omega)                          // X_H = ω⁻¹∇H
```

### Natural Transformations (`natural`)

| Type | Description |
|---|---|
| `NaturalTransformation` | Components η_M with naturality verification |
| `NatTransformComponent` | η_M: F(M) → G(M) as a matrix |
| `NoetherTheorem` | Conservation laws from symmetries |
| `ConservationLaw` | A specific conserved quantity |

```rust
// Noether's theorem
NoetherTheorem::energy(&manifold)
NoetherTheorem::linear_momentum(&manifold)
NoetherTheorem::angular_momentum(&manifold, center)
law.evaluate(&state)                        // Conserved value
law.verify(&s1, &s2)                        // Same value?

// General natural transformations
NaturalTransformation::new(label, components)
nat.verify_naturality(&eta_m, &eta_n, &f, &g)  // η_N ∘ F(f) = G(f) ∘ η_M
nat.vertical_compose(&other)                     // β ∘ α
```

### Presheaves (`presheaf`)

| Type | Description |
|---|---|
| `Observable` | Linear observation map with pullback |
| `ObservablePresheaf` | Contravariant functor on Agent |

```rust
Observable::scalar(matrix, label)           // 1D observation
Observable::vector(matrix, dim, label)      // Multi-dimensional
obs.observe(&state)                         // Apply observation
obs.pull_back(&morphism)                    // Contravariant: precompose

ObservablePresheaf::new(observables)
presheaf.verify_identity(&id)               // F(id) = id
presheaf.verify_contravariant(&f, &g)       // F(g∘f) = F(f)∘F(g)
```

### Yoneda Lemma (`yoneda`)

| Type | Description |
|---|---|
| `RepresentableFunctor` | Hom(-, A) for fixed agent A |
| `HomSet` | Hom(A, B) as a vector space of morphisms |
| `YonedaEmbedding` | Fully faithful functor A ↦ Hom(-, A) |
| `AgentProfile` | Agent characterization via Yoneda correspondence |

```rust
YonedaEmbedding::embed(agent)               // → RepresentableFunctor
YonedaEmbedding::yoneda_lemma(&agent, presheaf)  // F(A) ≅ Nat(Hom(-,A), F)
YonedaEmbedding::agents_equal_by_relationships(&a, &b, &probes)

AgentProfile::profile(&agent, &observers)   // Feature vector via hom-dimensions
profile.similarity(&other)                  // 0.0 (different) to 1.0 (identical)
```

### Adjunction (`adjunction`)

| Type | Description |
|---|---|
| `ObservationFunctor` | Obs: State → Agent (extract observables) |
| `ControlFunctor` | Ctrl: Agent → State (control dimensions) |
| `ObsCtrlAdjunction` | Obs ⊣ Ctrl with unit, counit, bijection |
| `StateSpace` / `StateMorphism` | Category of state spaces |

```rust
ObsCtrlAdjunction::new(obs_dim)
adj.unit(&state)                            // η: S → Ctrl(Obs(S))
adj.counit(&agent)                          // ε: Obs(Ctrl(A)) → A
adj.verify_triangle_identities(&state, &agent)
adj.bijection_forward(&morphism, &state)   // Hom(Obs(S), A) → Hom(S, Ctrl(A))
```

### Monoidal Structure (`monoidal`)

| Type | Description |
|---|---|
| `MonoidalAgent` | Tensor product A ⊗ B |
| `MonoidalMorphism` | f ⊗ g: A⊗C → B⊗D |
| `Braiding` | σ_{A,B}: A ⊗ B → B ⊗ A |
| `TrivialAgent` | Unit object I for the monoidal structure |

```rust
MonoidalAgent::tensor(&a, &b)               // Parallel composition
MonoidalAgent::tensor_entangled(&a, &b, E)  // With entanglement matrix
mono.dimension()                            // a_dim + b_dim

Braiding::swap(a_dim, b_dim)               // Swap map
braiding.apply(&state)
Braiding::verify_hexagon(..)                // Monoidal coherence

TrivialAgent::new()                         // Unit object I
```

### Compact Closed (`compact`)

| Type | Description |
|---|---|
| `DualAgent` | A* — the categorical dual |
| `Cup` | Evaluation map ε_A: A* ⊗ A → I |
| `Cap` | Coevaluation map η_A: I → A ⊗ A* |
| `SnakeResult` | Result of verifying snake/zigzag identities |

```rust
DualAgent::new(&agent)
let cup = Cup::new(dim);
let cap = Cap::new(dim);
cup.apply(&state)                           // Contract dual ⊗ original → scalar
cap.apply()                                 // Create entangled state from scalar
SnakeResult::verify(&cup, &cap, dim)        // Snake identity: (ε⊗id)∘(id⊗η) = id
```

### Application Pipeline (`application`)

| Type | Description |
|---|---|
| `LauSystem` | Multi-agent system with symplectic structure |
| `CategoricalPipelineResult` | Full verification result (serializable) |

```rust
let mut system = LauSystem::new("name");
system.add_physical_agent(&manifold)
system.observe(agent_idx, &direction)
system.energy(agent_idx, &hamiltonian)
system.parallel_compose(i, j)
system.profile_all()

let result = categorical_pipeline();
// result.functoriality_verified, .naturality_verified, .presheaf_verified,
// .snake_identity, .triangle_identities, .conserved_momentum, ...
```

---

## How It Works

### Layer 1: Symplectic Category (`symplectic`)

A **symplectic manifold** (M, ω) is an even-dimensional vector space with a closed, non-degenerate 2-form. In code, ω is an antisymmetric matrix with ω² = −I (up to sign). The canonical form on ℝ²ⁿ is:

$$\omega = \begin{pmatrix} 0 & I_n \\ -I_n & 0 \end{pmatrix}$$

A **symplectomorphism** φ: (M₁, ω₁) → (M₂, ω₂) satisfies φ*ω₂ = ω₁, checked as A^T ω₂ A = ω₁. These form the morphisms of the category **Sympl**.

### Layer 2: Agent Category (`agent`)

An **agent** is a state space ℝⁿ with named capabilities. An **agent morphism** is an affine map f(x) = Ax + b that preserves capability structure via an index mapping. This forms the category **Agent**.

### Layer 3: Hamiltonian Functor (`functor`)

The **Hamiltonian functor** F: Sympl → Agent is the bridge:
- **On objects**: F(M, ω) = agent with state space M, capabilities for observing position/momentum and flowing along Hamiltonian vector fields
- **On morphisms**: F(φ) = agent morphism mapping states via φ, preserving capabilities
- **Functoriality**: F(id) = id and F(g∘f) = F(g)∘F(f), verified numerically

A **Hamiltonian** H(x) = ½x^TQx + c^Tx generates the vector field X_H = ω⁻¹∇H, which gives Hamilton's equations of motion.

### Layer 4: Natural Transformations = Conservation Laws (`natural`)

**Noether's theorem**: every continuous symmetry of the Hamiltonian produces a conserved quantity. Categorically, a conserved quantity is a **natural transformation** between functors, where the naturality square encodes invariance under Hamiltonian flow:

$$\eta_N \circ F(\varphi) = G(\varphi) \circ \eta_M$$

The crate provides energy, linear momentum, and angular momentum conservation laws, each verified by checking that the conserved quantity is the same before and after flow.

### Layer 5: Presheaves = Observables (`presheaf`)

An **observable** is a contravariant functor (presheaf) on Agent. Given f: A → B, the pullback maps observations of B to observations of A. This contravariance is the categorical reason why "observing after transforming" differs from "transforming after observing." The crate verifies the presheaf axioms: F(id) = id and F(g∘f) = F(f)∘F(g).

### Layer 6: Yoneda Lemma (`yoneda`)

The **Yoneda lemma** states: Nat(Hom(-, A), F) ≅ F(A). In plain terms: an agent is completely determined by how all other agents relate to it. You don't need to inspect internals — the pattern of incoming morphisms is sufficient.

The crate uses this for **agent profiling**: compute hom-space dimensions from probe agents to build a feature vector. Agents with the same profile are categorically equivalent.

### Layer 7: Adjunction = Observation-Control Duality (`adjunction`)

An **adjunction** Obs ⊣ Ctrl between observation and control functors captures the duality:
- **Observation** extracts information from a state
- **Control** injects actions into a state
- The **unit** η: S → Ctrl(Obs(S)) embeds a state into its observable-controlled form
- The **counit** ε: Obs(Ctrl(A)) → A evaluates observations on controlled agents
- The **bijection** Hom(Obs(S), A) ≅ Hom(S, Ctrl(A)) is the adjunction isomorphism
- **Triangle identities** ensure coherence

### Layer 8: Monoidal Structure = Parallel Composition (`monoidal`)

The category of agents is **monoidal**:
- **Tensor product** A ⊗ B: parallel composition (concatenated state spaces)
- **Unit** I: trivial agent (dimension 0)
- **Braiding** σ: A ⊗ B → B ⊗ A (swap)
- **Entanglement**: non-trivial braiding when agents are coupled
- **Coherence**: associator α and hexagon identities verified

### Layer 9: Compact Closed = Duality (`compact`)

A **compact closed** category has duals for every object. The dual agent A* satisfies:
- **Evaluation** (cup): A* ⊗ A → I (contract to scalar)
- **Coevaluation** (cap): I → A ⊗ A* (create entangled pair)
- **Snake identity**: (ε ⊗ id_A) ∘ (id_A ⊗ η) = id_A — the zigzag equals the identity

This is the categorical underpinning of input/output duality and quantum-like entanglement.

### Layer 10: Application Pipeline (`application`)

The `LauSystem` ties everything together: a multi-agent system built from symplectic manifolds, supporting observation, energy computation, parallel composition, profiling, and a full categorical pipeline that verifies all structural properties simultaneously.

---

## The Math

### Symplectic Geometry

A symplectic form ω on M is a closed (dω = 0), non-degenerate 2-form. In coordinates on ℝ²ⁿ:

$$\omega\left(\begin{pmatrix} q \\ p \end{pmatrix}, \begin{pmatrix} q' \\ p' \end{pmatrix}\right) = p \cdot q' - q \cdot p'$$

Darboux's theorem: all symplectic manifolds of the same dimension are locally equivalent.

### Hamiltonian Mechanics

Given H: M → ℝ, the **Hamiltonian vector field** X_H is defined by ω(X_H, ·) = dH. In coordinates:

$$X_H = \omega^{-1} \nabla H$$

Hamilton's equations: q̇ = ∂H/∂p, ṗ = −∂H/∂q.

### Noether's Theorem

If a one-parameter family of symplectomorphisms preserves H, then the function I = ω(X_H, ξ) (where ξ is the infinitesimal generator) is a conserved quantity: dI/dt = 0 along Hamiltonian flow.

### Yoneda Lemma

For any functor F: C → Set and object c ∈ C:

$$\text{Nat}(\text{Hom}(c, -), F) \cong F(c)$$

The isomorphism is: η ↦ η_c(id_c). Fully faithful means the Yoneda embedding C → [C^op, Set] is injective on objects and morphisms.

### Adjunctions

F ⊣ G means Hom(FA, B) ≅ Hom(A, GB) naturally. The unit η: id → GF and counit ε: FG → id satisfy the triangle identities: εF ∘ Fη = id_F and Gε ∘ ηG = id_G.

### Monoidal Categories

A monoidal category (C, ⊗, I, α, λ, ρ) has a tensor product with associator α_{A,B,C}: (A⊗B)⊗C → A⊗(B⊗C), left/right unitors, and coherence conditions (pentagon, triangle). A **braiding** σ_{A,B}: A⊗B → B⊗A satisfies hexagon coherence.

### Compact Closed Categories

Every object A has a dual A* with cup η: I → A⊗A* and cap ε: A*⊗A → I satisfying:

$$(\varepsilon \otimes \text{id}_A) \circ (\text{id}_A \otimes \eta) = \text{id}_A$$
$$(\text{id}_{A^*} \otimes \varepsilon) \circ (\eta \otimes \text{id}_{A^*}) = \text{id}_{A^*}$$

---

## Test Suite

107 tests across 10 modules:

| Module | Tests | Coverage |
|---|---|---|
| `symplectic` | 10 | Canonical form, pairing, antisymmetry, symplectomorphisms, composition, inverse |
| `agent` | 9 | Construction, capabilities, morphisms, composition, inverse, dimension checks |
| `functor` | 8 | Hamiltonian evaluation, vector field, functoriality (identity + composition) |
| `natural` | 8 | Conservation laws (energy, momentum), naturality verification, vertical composition |
| `presheaf` | 9 | Observables, pullback, contravariant identity and composition, position/momentum |
| `yoneda` | 11 | Representable functors, hom-sets, Yoneda lemma, agent equality, profiling, similarity |
| `adjunction` | 11 | Observation/control functors, unit, counit, triangle identities, bijection, state morphisms |
| `monoidal` | 15 | Tensor, entanglement, braiding, associator, hexagon, unitors, pentagon, coherence |
| `compact` | 13 | Dual agents, cup/cap, snake identity, dimension checks, composition |
| `application` | 13 | LauSystem, pipeline, serialization, multi-agent, interoperability, conservation |

Run all tests:

```bash
cargo test
```

---

## License

MIT
