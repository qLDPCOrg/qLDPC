# Workarounds for slow paths of Stim

This guide lists the code in qLDPC that exists only because a path of [Stim](https://github.com/quantumlib/Stim) is slow, with one section for each limitation of Stim.
Code belongs here if it would be deleted or simplified mechanically once Stim is fast.
Algorithmic choices, ordinary uses of Stim's API, and caches that would remain useful do not.

Each site of a workaround carries a marker comment, or the same sentence in a docstring:

```python
# TEMPORARY until <trigger>: see circuits/stim_workarounds.md, "<section title>".
```

So `git grep "TEMPORARY until"` lists every site.
A trigger `stim>=X` holds once the dependency on Stim in qLDPC's `pyproject.toml` requires at least version `X`.
Any other trigger is a behavior of Stim, which its section shows how to check.
When the migration of a section lands, delete the section along with its markers.

## Resets with random outcomes in stim.TableauSimulator

**Slow path:**
`stim.TableauSimulator` collapses a qubit by transposing its whole tableau, so a reset instruction whose outcome is random costs time quadratic in the number `n` of qubits, even if the qubit is unentangled.
In stim 1.16.0 and 1.17.dev1793242987, such a reset on 2,000 qubits takes about 2.5 ms, while a single-qubit gate takes about 0.3 µs.

**Trigger:**
`stim.TableauSimulator` resets an unentangled qubit in time at most linear in `n`.
The following script prints about 16 while such resets take quadratic time, and at most about 4 once they take linear time.

```python
import time
import stim

def reset_time(num_qubits: int) -> float:
    simulator = stim.TableauSimulator()
    simulator.set_num_qubits(num_qubits)
    resets = stim.Circuit("RX 0\nR 0")  # each reset has a random outcome
    start = time.perf_counter()
    for _ in range(10):
        simulator.do(resets)
    return time.perf_counter() - start

print(reset_time(4000) / reset_time(1000))
```

**Sites:**
In `qldpc.circuits.encoding`, the marked `if`/`else` of `get_state_stabilizers` simulates circuits on at least `_MIN_QUBITS_FOR_CHEAP_RESETS` qubits with `_do_with_cheap_resets`, which replaces the reset of each qubit in an eigenstate of another Pauli by single-qubit gates that it looks up in `_RESET_PAULIS`, `_PAULI_FLIPS`, and `_PAULI_SWAPS`.
In `qldpc.circuits.encoding_test`, `test_state_stabilizers_after_replaced_resets`, `test_cheap_resets_on_large_circuits`, and the `min_qubits` parametrization of `test_state_stabilizers_after_resets` exist only for this replacement.

**Migration:**

1. In `get_state_stabilizers`, replace the marked `if`/`else`, with its marker, by `simulator.do(circuit)`.
2. Delete `_do_with_cheap_resets`, `_MIN_QUBITS_FOR_CHEAP_RESETS`, `_RESET_PAULIS`, `_PAULI_FLIPS`, and `_PAULI_SWAPS`, with the comments and markers above them.
3. Delete `test_state_stabilizers_after_replaced_resets` and `test_cheap_resets_on_large_circuits`, with the markers above them.
4. In `test_state_stabilizers_after_resets`, delete the marker and parametrization above it, its `monkeypatch` and `min_qubits` arguments, the `if min_qubits is not None:` block that sets `_MIN_QUBITS_FOR_CHEAP_RESETS`, and the docstring's sentences about replaced and kept resets.
   Keep its circuit and expected stabilizers, which check resets after a measurement with feedback, repeated targets, and a REPEAT block.
5. Delete this section.

**Check:**
Time `get_state_stabilizers(circuit, 2000)` before and after the migration, where `circuit = stim.Circuit("\n".join(f"RX {q}\nCX {q} {q + 1}\nTICK" for q in range(0, 2000, 2)))` prepares 1,000 Bell pairs with one reset instruction each.
With the workaround and stim 1.17.dev1793242987, it takes about 0.2 s, and about 3 s without the workaround.
It should take no longer after the migration.
