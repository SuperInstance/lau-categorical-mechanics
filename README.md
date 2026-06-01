# lau-categorical-mechanics

Categorical foundations unifying symplectic mechanics and agent behaviors.

**Key insight:** Hamiltonian mechanics is a functor from the category of symplectic manifolds to the category of agent behaviors. Conservation laws are natural transformations.

## Architecture

| Module | Description |
|--------|-------------|
| `symplectic` | Symplectic category: objects = symplectic manifolds, morphisms = symplectomorphisms |
| `agent` | Agent category: objects = agents, morphisms = capability-preserving maps |
| `functor` | Hamiltonian functor: maps symplectic → agent category |
| `natural` | Natural transformations: conservation laws as naturality squares |
| `presheaf` | Presheaf of observables: contravariant functor from agent states to measurements |
| `yoneda` | Yoneda lemma: an agent is determined by its relationships, not its internals |
| `adjunction` | Adjunction Obs ⊣ Ctrl between observation and control |
| `monoidal` | Monoidal structure: tensor product = parallel agent composition, braiding = swapping |
| `compact` | Compact closed: every agent has a dual (observation ↔ control) |
| `application` | Categorical foundation for the Lau ecosystem |

## Usage

```rust
use lau_categorical_mechanics::*;

// Create a symplectic manifold (phase space)
let phase_space = SymplecticManifold::canonical(2, "T*R");

// The Hamiltonian functor maps it to an agent
let agent = HamiltonianFunctor::map_object(&phase_space);

// Conservation laws are natural transformations
let momentum = NoetherTheorem::linear_momentum(&phase_space);

// Compose agents in parallel
let system = MonoidalAgent::tensor(&agent, &agent);

// Every agent has a dual
let dual = DualAgent::dual(&agent);
```

## Dependencies

- `nalgebra` — linear algebra
- `serde` / `serde_json` — serialization

## Tests

107 tests covering all categorical structures, coherence conditions, and their applications.

## License

MIT
