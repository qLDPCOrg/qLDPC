# SPDX-License-Identifier: Apache-2.0

"""Acquire, build, and check the circuits of the certification experiment.

Everything in ``state_prep_certification/`` is a prototype, subject to change: its code, interfaces,
inputs, outputs, and documentation.

    python circuits.py --fetch                          (workstation)
    python circuits.py --build                          (workstation)
    python circuits.py --check                          (both machines)

``--fetch`` downloads the archive of MQT QECC at the commit of ``config.json``, checks its
SHA-256, and copies the state preparation circuits ``zero_ft_*.stim`` and ``zero_non_ft_*.stim``,
the cat-state circuits ``ft_ghz_<w>_<t>.stim``, and MQT's license into ``circuit_data/mqt/``.

``--build`` and ``--check`` make and verify the certification inputs; see ``build_inputs`` and
``check_inputs``.  The experiment's README lists the folders of its data directory,
``circuit_data/``, which Git ignores, and says which of them are inputs that no script makes.  This
script makes ``mqt/`` and ``inputs/``, which holds the inputs and ``circuits.json``.  ``--build``
reads ``mqt/``, ``translated/``, ``codes/``, and the verified circuits that the ``provenance.json``
of each folder of synthesized circuits records as synthesized; each such folder gives a family of
its own (``SYNTHESES``).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import stim

from experiments.state_prep_certification.certification import baseline, code_files, counting
from experiments.state_prep_certification.certification.settings import (
    CIRCUITS_JSON,
    CONFIG,
    DATA,
    INPUTS,
    SYNTHESES,
    VERIFIED_FAMILIES,
    certify_options,
    noisy_circuit,
)
from experiments.state_prep_certification.certification.states import same_state
from experiments.state_prep_certification.common import store
from qldpc import circuits as qldpc_circuits

MQT_DIR = DATA / "mqt"
SOURCES = CONFIG["workstation_sources"]
TRANSLATED = DATA / "translated"
# The licenses that travel with the inputs, by their name beside the inputs.
LICENSES = {
    "MQT_QECC_LICENSE": MQT_DIR / "LICENSE",
    "VFTQECC_LICENSE": TRANSLATED / "VFTQECC_LICENSE",
}
MQT_PREFIX = "scripts/ft_stateprep/eval/circuits/"
CAT_PREFIX = "scripts/cat_states/circuits/"


def main() -> None:
    """Run the selected command."""
    args = parse_args()
    if args.fetch:
        fetch()
    elif args.build:
        build_inputs()
    elif args.check:
        check_inputs()
    else:
        raise SystemExit("choose a command; see --help")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--fetch", action="store_true", help="download MQT's circuits")
    commands.add_argument("--build", action="store_true", help="build the certification inputs")
    commands.add_argument(
        "--check",
        action="store_true",
        help="check the inputs' hashes, DEM parameters, states, output qubits, K = t + 1, noise, "
        "and code files, and that the states are CSS; where circuit_data/mqt/ is present, also "
        "rebuild the inputs from its sources",
    )
    return parser.parse_args()


####################################################################################################
# fetching


def fetch() -> None:
    """Download MQT QECC at the pinned commit and copy its circuits and license."""
    url = (
        "https://codeload.github.com/munich-quantum-toolkit/qecc/tar.gz/"
        + SOURCES["mqt_qecc_commit"]
    )
    with urllib.request.urlopen(url, timeout=300) as response:
        archive = response.read()
    digest = hashlib.sha256(archive).hexdigest()
    if digest != SOURCES["mqt_qecc_archive_sha256"]:
        raise SystemExit(f"MQT's archive has SHA-256 {digest}, not the pinned one")
    count = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            path = member.name.split("/", 1)[1] if "/" in member.name else ""
            name = path.rsplit("/", 1)[-1]
            wanted = (
                (
                    path.startswith(MQT_PREFIX)
                    and name.startswith("zero_")
                    and name.endswith(".stim")
                )
                or (
                    path.startswith(CAT_PREFIX)
                    and name.startswith("ft_ghz_")
                    and name.endswith(".stim")
                )
                or path == "LICENSE"
            )
            if wanted and member.isfile():
                target = MQT_DIR / path
                target.parent.mkdir(parents=True, exist_ok=True)
                file = tar.extractfile(member)
                assert file is not None
                target.write_bytes(file.read())
                count += 1
    print(f"fetched {count} files of MQT QECC at {SOURCES['mqt_qecc_commit']} into {MQT_DIR}")


####################################################################################################
# building the certification inputs

# The codes of MQT's state preparation circuits, by folder: [[n, k, d]].  The folder names of the
# color codes without a distance hold their distance-5 circuits, which ``build_inputs`` checks by n.
A_CODES = {
    "steane": (7, 1, 3),
    "shor": (9, 1, 3),
    "tetrahedral": (15, 1, 3),
    "hamming": (15, 7, 3),
    "carbon": (12, 2, 4),
    "rotated_surface_d3": (9, 1, 3),
    "rotated_surface_d5": (25, 1, 5),
    "cc_4_8_8": (17, 1, 5),
    "cc_6_6_6": (19, 1, 5),
    "cc_4_8_8_d7": (31, 1, 7),
}


ALLOWED_GATES = {"H", "CX", "R", "RX", "M", "MR", "MX", "MRX", "CZ", "TICK"}
RESET_GATES = {"R", "RX"}


def build_inputs() -> None:
    """Build every certification input, and the weakened variants, and write ``circuits.json``.

    See ``make_inputs``; this writes the inputs, the licenses, and ``circuits.json`` to
    ``circuit_data/inputs/``.
    """
    data = make_inputs(INPUTS)
    for entry in data["circuits"]:
        print(f"{entry['id']}: {entry['status']} {entry.get('reason', '')}")
    CIRCUITS_JSON.write_text(_dump(data))
    counts: dict[str, int] = {}
    for entry in data["circuits"]:
        key = f"{entry['family']} {entry['status']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps(counts, indent=1, sort_keys=True))


def make_inputs(inputs: pathlib.Path) -> dict[str, Any]:
    """Build every certification input, and the weakened variants; return ``circuits.json``.

    Each input of ``VERIFIED_FAMILIES`` that is not excluded gets one weakened variant
    (``_make_weakened``).  Each input is written noiseless to ``<inputs>/<id>.stim``; the node adds
    the noise and certify options that ``circuits.json`` records.  Every input is validated by
    building its DEM with the certifier, untimed and without an oracle, which also gives the DEM
    parameters of ``circuits.json`` (``dem_parameters``); a cat circuit whose added detectors qLDPC
    rejects is recorded as excluded.  The licenses of ``LICENSES`` are copied beside the inputs.
    The record of the sources gives the MQT QECC releases that synthesized the circuits of
    ``SYNTHESES``, from their ``provenance.json``, and each of these circuits records its release.
    Each input of ``VERIFIED_FAMILIES`` records its code file and the file's SHA-256
    (``code_files.code_file_fields``).
    """
    inputs.mkdir(parents=True, exist_ok=True)
    for name, path in LICENSES.items():
        shutil.copyfile(path, inputs / name)
    entries = []
    for source in _sources():
        made = [_make_entry(source, inputs)]
        if made[0]["status"] == "ok":
            made[0] |= code_files.code_file_fields(made[0])
        if source["family"] in VERIFIED_FAMILIES and made[0]["status"] == "ok":
            made.append(_make_weakened(made[0], inputs))
        entries += made
    releases = sorted(
        {
            call["versions"]["mqt.qecc"]
            for _, folder in SYNTHESES.values()
            for call in _synthesized_circuits(folder)
        }
    )
    return {"circuits": entries, "sources": SOURCES | {"mqt_qecc_synthesis_releases": releases}}


def dem_parameters(circuit: Any, options: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """The DEM parameters of a noiseless input with the noise and certify options of a circuit.

    The certifier runs untimed with ``find_smallest_violation`` replaced by a recorder of its
    arguments, so that it builds the DEM and runs no oracle.  The parameters are N
    (``num_mechanisms``), p (``num_post_selection_detectors``), N_0, z, r, K' = min(K, r + 1),
    U = sum over 1 <= m <= K of C(N_0, m), whether U is within the brute-force cap of
    ``config.json`` (``brute_force``), and whether the ground truth applies (``ground_truth``):
    whether n and the sum over 0 <= m <= K of C(N_0, m) are within the limits of ``ground_truth``
    of ``config.json``.  The README's section Terms defines each symbol.

    Raises:
        ValueError: If qLDPC rejects the circuit, such as for a nondeterministic detector.
    """
    recorded: dict[str, list[int]] = {}

    def recorder(flags: Any, syndromes: Any, *args: Any, **kwargs: Any) -> None:
        recorded["flags"], recorded["syndromes"] = list(flags), list(syndromes)

    noisy = noisy_circuit(circuit, options["noise"])
    with baseline.fault_search_replaced(recorder):
        certificate = qldpc_circuits.certify_bounded_fault_propagation(
            noisy, method="brouwer_zimmermann", **certify_options(options)
        )
    reduced = counting.reduce_signatures(recorded["flags"], recorded["syndromes"])
    max_weight = options["max_weight"]
    size = baseline.enumeration_size(reduced.num_reduced, max_weight)
    rule = CONFIG["ground_truth"]
    ground_truth = (
        len(certificate.output_qubits) <= rule["max_qubits"]
        and 1 + size <= rule["max_combinations"]
    )
    return certificate, {
        "num_mechanisms": certificate.detector_error_model.num_errors,
        "num_post_selection_detectors": len(certificate.post_selection_detectors),
        "N0": reduced.num_reduced,
        "z": reduced.num_zero_flag,
        "r": reduced.rank,
        "K_prime": reduced.max_fault_weight(max_weight),
        "U": size,
        "brute_force": size <= CONFIG["brute_force"]["max_combinations"],
        "ground_truth": ground_truth,
    }


def _sources() -> list[dict[str, Any]]:
    """Every source circuit of the fixed list, with its family, identity, and design t."""
    sources: list[dict[str, Any]] = []
    root = MQT_DIR / MQT_PREFIX
    for folder in sorted(A_CODES):
        n, k, d = A_CODES[folder]
        for path in sorted((root / folder).glob("zero_*.stim")):
            ft = path.name.startswith("zero_ft_")
            variant = path.stem.removeprefix("zero_ft_" if ft else "zero_non_ft_")
            sources.append(
                {
                    "id": f"{'aft' if ft else 'aunv'}.{folder}.{variant}",
                    "family": "A-ft" if ft else "A-unv",
                    "path": path,
                    "repository": SOURCES["mqt_qecc_repository"],
                    "commit": SOURCES["mqt_qecc_commit"],
                    "source_path": MQT_PREFIX + f"{folder}/{path.name}",
                    "code": [n, k, d],
                    "t": (d - 1) // 2,
                }
            )
    for path in sorted((MQT_DIR / CAT_PREFIX).glob("ft_ghz_*.stim")):
        w, t = (int(part) for part in path.stem.removeprefix("ft_ghz_").split("_"))
        sources.append(
            {
                "id": f"bcat.w{w}.t{t}",
                "family": "B-cat",
                "path": path,
                "repository": SOURCES["mqt_qecc_repository"],
                "commit": SOURCES["mqt_qecc_commit"],
                "source_path": CAT_PREFIX + path.name,
                "code": None,
                "cat_width": w,
                "t": t,
            }
        )
    sources.append(
        {
            "id": "cgoto",
            "family": "C-goto",
            "path": TRANSLATED / "goto_steane.stim",
            "repository": "qLDPC",
            "commit": None,
            "source_path": "examples/logical_error_rates/5_fault_tolerance_certificates.ipynb",
            "code": [7, 1, 3],
            "t": 1,
        }
    )
    for family, (prefix, synthesized) in SYNTHESES.items():
        for call in _synthesized_circuits(synthesized):
            n, k, d = call["parameters"]
            sources.append(
                {
                    "id": f"{prefix}.{call['code']}.{call['step']}",
                    "family": family,
                    "path": synthesized / call["stim_file"],
                    "repository": f"MQT QECC {call['versions']['mqt.qecc']} synthesis",
                    "mqt_qecc_release": call["versions"]["mqt.qecc"],
                    "commit": None,
                    "source_path": f"circuit_data/{synthesized.name}/{call['stim_file']}",
                    "code": [n, k, d],
                    "t": (d - 1) // 2,
                }
            )
    sources.append(
        {
            "id": "echen",
            "family": "E-chen",
            "path": TRANSLATED / "chen_cat8.stim",
            "repository": SOURCES["vftqecc_repository"],
            "commit": SOURCES["vftqecc_commit"],
            "source_path": "QuantumSE.jl/example/CatPreparation.jl",
            "code": None,
            "cat_width": 8,
            "t": 2,
        }
    )
    return sources


def _synthesized_circuits(folder: pathlib.Path) -> list[dict[str, Any]]:
    """The records of the verification calls that synthesized a circuit, from the provenance."""
    provenance = json.loads((folder / "provenance.json").read_text())
    return [
        call
        for call in provenance["calls"]
        if call["step"] != "prep" and call["outcome"] == "synthesized"
    ]


def _make_entry(source: dict[str, Any], inputs: pathlib.Path) -> dict[str, Any]:
    """Derive, validate, and write the certification input of one source circuit."""
    text = source["path"].read_text()
    entry = {key: value for key, value in source.items() if key != "path"}
    entry["source_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    try:
        circuit = stim.Circuit(text)
    except ValueError as error:
        return entry | {"status": "excluded", "reason": f"Stim cannot read the source: {error}"}
    gates = {instruction.name for instruction in circuit.flattened()}
    if not gates <= ALLOWED_GATES:
        raise SystemExit(f"{source['id']}: unexpected instructions {sorted(gates - ALLOWED_GATES)}")

    # prepend resets, and for cat circuits compare consecutive measurement results
    circuit, reset_qubits = _prepend_resets(circuit)
    cat = source["family"] == "B-cat"
    if cat:
        for j in range(1, circuit.num_measurements):
            circuit.append("DETECTOR", [stim.target_rec(-j), stim.target_rec(-j - 1)])
    options = {
        "add_flags": not cat,
        "post_select": True if cat else None,
        "output_qubits": None,
        "css_weight": None,
        "max_weight": source["t"] + 1,
        "noise": CONFIG["noise"],
    }
    entry |= {
        "derivation": {
            "prepended_resets": reset_qubits,
            "added_detectors": circuit.num_detectors if cat else 0,
        },
        "options": options,
    }
    entry.update(_validate_and_write(source["id"], circuit, options, inputs))
    if entry["status"] == "ok" and source["code"] is not None:
        n = len(entry["output_qubits"])
        if n != source["code"][0]:
            raise SystemExit(f"{source['id']}: {n} output qubits, not n = {source['code'][0]}")
    return entry


def _make_weakened(parent: dict[str, Any], inputs: pathlib.Path) -> dict[str, Any]:
    """The weakened variant of a verified circuit: delete the interactions of one measured qubit.

    The selected qubit is that of the last target of the last measurement, then the qubit measured
    just before it, and so on, until qLDPC accepts the variant with the parent's output qubits and
    the variant prepares the parent's state, as ``_validate_and_write`` checks.  This is the order
    "last measured first" of the deletion tasks (``applications.measured_qubits``) except in its
    ties: qubits whose last measurements share an instruction come here in the reverse order of its
    targets, and there in the order of their first measurements.
    """
    circuit = stim.Circuit((inputs / f"{parent['id']}.stim").read_text())
    options = dict(parent["options"], output_qubits=parent["output_qubits"])
    tried = []
    for qubit in _measured_qubits_backwards(circuit):
        variant, deleted, measured = _delete_qubit(circuit, qubit)
        result = _validate_and_write(
            f"w.{parent['id']}", variant, options, inputs, parent["stabilizers"]
        )
        tried.append({"qubit": qubit, "outcome": result["status"]})
        if result["status"] == "ok":
            return {
                "id": f"w.{parent['id']}",
                "family": "W",
                "parent": parent["id"],
                "code": parent["code"],
                "t": parent["t"],
                "derivation": {
                    "selected_qubit": qubit,
                    "deleted_target_groups": deleted,
                    "deleted_measurements": measured,
                    "tried": tried,
                },
                "options": options,
                **result,
            }
    (inputs / f"w.{parent['id']}.stim").unlink(missing_ok=True)
    return {
        "id": f"w.{parent['id']}",
        "family": "W",
        "parent": parent["id"],
        "status": "not constructed",
        "derivation": {"tried": tried},
    }


def _dump(data: Any) -> str:
    """The JSON text of ``circuits.json`` or of one of its records."""
    return json.dumps(data, indent=1, sort_keys=True) + "\n"


def _prepend_resets(circuit: Any) -> tuple[Any, list[int]]:
    """Prepend ``R`` on every qubit whose first operation is not a reset."""
    first: dict[int, str] = {}
    for instruction in circuit.flattened():
        for target in instruction.targets_copy():
            if target.is_qubit_target and target.value not in first:
                first[target.value] = instruction.name
    qubits = sorted(qubit for qubit, name in first.items() if name not in RESET_GATES)
    prefix = stim.Circuit()
    if qubits:
        prefix.append("R", qubits)
    return prefix + circuit, qubits


def _measured_qubits_backwards(circuit: Any) -> list[int]:
    """Distinct measured qubits, from the last target of the last measurement backwards."""
    order: list[int] = []
    for instruction in reversed(circuit.flattened()):
        if stim.gate_data(instruction.name).produces_measurements:
            for target in reversed(instruction.targets_copy()):
                if target.value not in order:
                    order.append(target.value)
    return order


def _delete_qubit(circuit: Any, qubit: int) -> tuple[Any, int, int]:
    """Delete every target group that involves the qubit; return the circuit and the counts.

    Raises:
        ValueError: If the circuit has detectors or observables, whose measurement records the
            deletion of a measurement would shift.
    """
    result = stim.Circuit()
    deleted = measured = 0
    for instruction in circuit.flattened():
        if instruction.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            raise ValueError(f"Cannot delete a qubit of a circuit with {instruction.name}")
        if instruction.name == "TICK":
            result.append(instruction)
            continue
        kept = []
        for group in instruction.target_groups():
            if any(target.value == qubit for target in group):
                deleted += 1
                measured += stim.gate_data(instruction.name).produces_measurements
            else:
                kept.extend(group)
        if kept:
            result.append(instruction.name, kept, instruction.gate_args_copy())
    return result, deleted, measured


def _validate_and_write(
    circuit_id: str,
    circuit: Any,
    options: dict[str, Any],
    inputs: pathlib.Path,
    parent_stabilizers: list[str] | None = None,
) -> dict[str, Any]:
    """Build the circuit's DEM untimed (``dem_parameters``); write the input if qLDPC accepts it.

    For a weakened variant, the prepared state must also equal the parent's: every stabilizer of
    the variant has expectation +1 in the parent's state.
    """
    try:
        certificate, parameters = dem_parameters(circuit, options)
    except ValueError as error:
        return {"status": "excluded", "reason": str(error)[:500]}
    stabilizers = [str(stabilizer) for stabilizer in certificate.state_stabilizers]
    if parent_stabilizers is not None and not same_state(parent_stabilizers, stabilizers):
        return {"status": "excluded", "reason": "the prepared state differs from the parent's"}
    path = inputs / f"{circuit_id}.stim"
    path.write_text(str(circuit) + "\n")
    return {
        "status": "ok",
        "input_sha256": store.sha256_file(path),
        "num_qubits": circuit.num_qubits,
        "output_qubits": list(certificate.output_qubits),
        "stabilizers": stabilizers,
        "css_weight": certificate.css_weight,
        **parameters,
    }


####################################################################################################
# checking the inputs (both machines)


def _check_input(entry: dict[str, Any]) -> list[str]:
    """The problems of one record of ``circuits.json``, checked against its input file.

    The file must match the SHA-256 that ``circuits.json`` records for it.  Its DEM parameters,
    recomputed as ``--build`` computes them (``dem_parameters``), must equal those of
    ``circuits.json``.  So must the input's prepared state and output qubits.  The input must
    prepare a CSS state: with ``css_weight`` None in its options, qLDPC sets ``css_weight`` true
    exactly for a CSS state, so the options must leave it None, and qLDPC's choice and the recorded
    ``css_weight`` must both be true.  The record's K must be t + 1, and its noise that of
    ``config.json``.  An input of ``VERIFIED_FAMILIES`` must have a code file that passes
    ``code_files.check_code_file``, which checks, among other things, that the input's state is the
    code's |0...0>, that the record's n agrees with the code file, and that its t does where the
    family takes t from the distance (``settings.T_FROM_DISTANCE``) and the file gives the distance.
    """
    name = entry["id"]
    path = INPUTS / f"{name}.stim"
    if not path.exists():
        return [f"{name}: missing {path}"]
    problems = []
    if store.sha256_file(path) != entry["input_sha256"]:
        problems.append(f"{name}: SHA-256 mismatch")
    certificate, parameters = dem_parameters(stim.Circuit(path.read_text()), entry["options"])
    for key, value in parameters.items():
        if entry.get(key) != value:
            problems.append(f"{name}: {key} is {value}, not {entry.get(key)}")
    stabilizers = [str(stabilizer) for stabilizer in certificate.state_stabilizers]
    if not same_state(entry["stabilizers"], stabilizers):
        problems.append(f"{name}: the input prepares a state other than the recorded one")
    output_qubits = list(certificate.output_qubits)
    if output_qubits != entry["output_qubits"]:
        problems.append(f"{name}: output_qubits is {output_qubits}, not {entry['output_qubits']}")
    if entry["options"]["css_weight"] is not None:
        problems.append(f"{name}: its options set css_weight")
    elif certificate.css_weight is not True:
        problems.append(f"{name}: the input does not prepare a CSS state")
    if entry["css_weight"] is not True:
        problems.append(f"{name}: css_weight is {entry['css_weight']}, not True")
    if entry["options"]["max_weight"] != entry["t"] + 1:
        problems.append(f"{name}: max_weight is {entry['options']['max_weight']}, not t + 1")
    if entry["options"]["noise"] != CONFIG["noise"]:
        problems.append(f"{name}: its noise is not that of config.json")
    if "code_file" in entry:
        problems += code_files.check_code_file(entry)
    elif entry["family"] in VERIFIED_FAMILIES:
        problems.append(f"{name}: no code_file")
    return problems


def check_inputs() -> None:
    """Check every input against ``circuits.json``, and the code file of each circuit that has one.

    Every input of ``circuits.json`` must pass ``_check_input``, and the licenses of ``LICENSES``
    must lie beside the inputs.  Where the fetched sources are present, in ``circuit_data/mqt/`` on
    the workstation, the inputs are also rebuilt from the sources in a scratch directory, and the
    rebuilt ``circuits.json``, with the SHA-256 of every input, and the licenses must equal the
    recorded ones.
    """
    recorded = json.loads(CIRCUITS_JSON.read_text())
    entries = [entry for entry in recorded["circuits"] if entry["status"] == "ok"]
    problems = [f"missing {name}" for name in LICENSES if not (INPUTS / name).exists()]
    for entry in entries:
        problems += _check_input(entry)
    checked = sum((INPUTS / f"{entry['id']}.stim").exists() for entry in entries)
    rebuilt = MQT_DIR.exists()
    if rebuilt:
        problems += _check_rebuilt(recorded)
    for problem in problems:
        print(problem)
    how = ", and rebuilt them from the sources" if rebuilt else ""
    print(f"checked {checked} inputs{how}; {len(problems)} problems")
    if problems:
        raise SystemExit(1)


def _check_rebuilt(recorded: dict[str, Any]) -> list[str]:
    """Rebuild the inputs from the sources in a scratch directory; list how they differ."""
    with tempfile.TemporaryDirectory(dir=DATA, prefix="rebuild-") as scratch:
        rebuilt = make_inputs(pathlib.Path(scratch))
        problems = [
            f"{name} differs from its source"
            for name in LICENSES
            if not (INPUTS / name).exists()
            or (INPUTS / name).read_bytes() != (pathlib.Path(scratch) / name).read_bytes()
        ]
    if rebuilt["sources"] != recorded["sources"]:
        problems.append("the record of the sources differs when rebuilt")
    old = {entry["id"]: _dump(entry) for entry in recorded["circuits"]}
    new = {entry["id"]: _dump(entry) for entry in rebuilt["circuits"]}
    problems += [
        f"{circuit_id}: differs when rebuilt from the sources"
        for circuit_id in sorted(old.keys() | new.keys())
        if old.get(circuit_id) != new.get(circuit_id)
    ]
    return problems


if __name__ == "__main__":
    main()
