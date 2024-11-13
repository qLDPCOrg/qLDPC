"""Tools for simulating error-correcting codes to identify logical error rates

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pathlib
from collections.abc import Sequence

import beliefmatching
import matplotlib.pyplot as plt
import numpy as np
import sinter
import stim

from qldpc import codes, objects


def get_syndrome_extraction_circuit(
    code: codes.CSSCode,
    error_rate: float,
    stabilizer_pauli: objects.PauliXZ = objects.Pauli.X,
    rounds: int = 1,
    gate_order: Sequence[tuple[int, int]] | None = None,
) -> stim.Circuit:
    """Get the syndrome extraction circuit for a CSS qubit code.

    Args:
        code: the CSS code in question
        stabilizer_pauli: the Pauli type of the stabilizers to measure
        error_rate: the probability of SPAM and two-qubit gate errors
        rounds: the number of syndrome extraction rounds
        gate_order (optional): the order in which to apply syndrome extraction gates
    Returns:
        a stim circuit

    The gate_order is a sequence of tuples (ancilla_qubit_index, data_qubit_index), where the
    ancilla_qubit_index must be in range(code.num_checks), and the data_qubit_index must be in
    range(code.num_qubits).  There should be exactly one tuple for each edge in the Tanner graph of
    the code.
    """
    if not code.field.order == 2:
        raise ValueError("Syndrome extraction circuits only supported for qubit CSS codes.")

    # identify data and ancilla qubits by index
    data_qubits = list(range(code.num_qubits))
    ancillas_x = [len(data_qubits) + qq for qq in range(code.num_checks_x)]
    ancillas_z = [len(data_qubits) + len(ancillas_x) + qq for qq in range(code.num_checks_z)]
    ancillas_xz = ancillas_x + ancillas_z

    if gate_order is None:
        gate_order = [
            (ancilla, data_qubit)
            for ancillas, matrix in [(ancillas_x, code.matrix_x), (ancillas_z, code.matrix_z)]
            for ancilla, row in zip(ancillas, matrix)
            for data_qubit in np.nonzero(row)[0]
        ]
    else:
        # assert that there are the correct number of gates addressing the correct qubits
        num_gates_x = len(np.nonzero(code.matrix_x)[0])
        num_gates_z = len(np.nonzero(code.matrix_z)[0])
        assert len(gate_order) == num_gates_x + num_gates_z
        assert all(
            (
                code.matrix_x[ancilla, data_qubit]
                if ancilla < code.num_checks_x
                else code.matrix_z[ancilla, data_qubit]
            )
            for ancilla, data_qubit in gate_order
        )

    # initialize data qubits
    circuit = stim.Circuit()
    circuit.append(f"R{stabilizer_pauli}", data_qubits)
    circuit.append(f"{~stabilizer_pauli}_ERROR", data_qubits, error_rate)

    # initialize ancillas, deferring SPAM errors until later
    circuit.append("RX", ancillas_xz)

    # construct circuit to extract syndromes once
    single_round_circuit = stim.Circuit()

    single_round_circuit.append("Z_ERROR", ancillas_xz, error_rate)
    for ancilla, data_qubit in gate_order:
        gate = "CX" if ancilla < (len(data_qubits) + len(ancillas_x)) else "CZ"
        single_round_circuit.append(gate, [ancilla, data_qubit])
        single_round_circuit.append("DEPOLARIZE2", [ancilla, data_qubit], error_rate)

    # noisy syndrome measurement + ancilla reset
    single_round_circuit.append("Z_ERROR", ancillas_xz, error_rate)
    single_round_circuit.append("MRX", ancillas_xz)

    # append first round of syndrome extraction
    circuit += single_round_circuit

    # initial ancilla detectors
    ancilla_recs = {ancilla: -code.num_checks + qq for qq, ancilla in enumerate(ancillas_xz)}
    for ancilla in ancillas_x if stabilizer_pauli is objects.Pauli.X else ancillas_z:
        circuit.append("DETECTOR", stim.target_rec(ancilla_recs[ancilla]))

    # additional rounds of syndrome extraction
    if rounds > 1:
        repeat_circuit = single_round_circuit.copy()
        for ancilla in ancillas_xz:
            recs = [ancilla_recs[ancilla], ancilla_recs[ancilla] - code.num_checks]
            repeat_circuit.append("DETECTOR", [stim.target_rec(rec) for rec in recs])
        circuit += (rounds - 1) * repeat_circuit

    # measure out data qubits
    circuit.append(f"{~stabilizer_pauli}_ERROR", data_qubits, error_rate)
    circuit.append(f"M{stabilizer_pauli}", data_qubits)

    # check stabilizer parity
    data_qubit_recs = [-code.num_qubits + qq for qq in data_qubits]
    ancilla_recs = {
        ancilla: -code.num_qubits - code.num_checks + qq for qq, ancilla in enumerate(ancillas_xz)
    }
    ancillas = ancillas_x if stabilizer_pauli is objects.Pauli.X else ancillas_z
    matrix = code.matrix_x if stabilizer_pauli is objects.Pauli.X else code.matrix_z
    for ancilla, row in zip(ancillas, matrix):
        recs = [ancilla_recs[ancilla]] + [data_qubit_recs[qq] for qq in np.nonzero(row)[0]]
        circuit.append("DETECTOR", [stim.target_rec(rec) for rec in recs])

    # check logical observable parity
    for logical_op_idx, logical_op in enumerate(code.get_logical_ops(stabilizer_pauli)):
        recs = [stim.target_rec(data_qubit_recs[qq]) for qq in np.nonzero(logical_op)[0]]
        circuit.append("OBSERVABLE_INCLUDE", recs, logical_op_idx)

    return circuit


class CustomDecoder(sinter.Decoder):
    """
    Initializes a CustomDecoder object.

    Args
    ----------
    decoder: A decoder object
    """

    def __init__(self, decoder_function):
        self.decoder_function = decoder_function

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:
        """Performs decoding by reading problems from, and writing solutions to, file paths.
        Args:
            num_shots: The number of times the circuit was sampled. The number of problems
                to be solved.
            num_dets: The number of detectors in the circuit. The number of detection event
                bits in each shot.
            num_obs: The number of observables in the circuit. The number of predicted bits
                in each shot.
            dem_path: The file path where the detector error model should be read from,
                e.g. using `stim.DetectorErrorModel.from_file`. The error mechanisms
                specified by the detector error model should be used to configure the
                decoder.
            dets_b8_in_path: The file path that detection event data should be read from.
                Note that the file may be a named pipe instead of a fixed size object.
                The detection events will be in b8 format (see
                https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md ). The
                number of detection events per shot is available via the `num_dets`
                argument or via the detector error model at `dem_path`.
            obs_predictions_b8_out_path: The file path that decoder predictions must be
                written to. The predictions must be written in b8 format (see
                https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md ). The
                number of observables per shot is available via the `num_obs` argument or
                via the detector error model at `dem_path`.
            tmp_dir: Any temporary files generated by the decoder during its operation MUST
                be put into this directory. The reason for this requirement is because
                sinter is allowed to kill the decoding process without warning, without
                giving it time to clean up any temporary objects. All cleanup should be done
                via sinter deleting this directory after killing the decoder.
        """
        self.dem = stim.DetectorErrorModel.from_file(dem_path)
        self.initialize_decoder()

        shots = stim.read_shot_data_file(path=dets_b8_in_path, format="b8", num_detectors=num_dets)

        predictions = np.zeros((num_shots, num_obs), dtype=bool)

        for i in range(num_shots):
            predictions[i, :] = self.full_decode(shots[i, :])

        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=num_obs,
        )

    def full_decode(self, syndrome: np.ndarray) -> np.ndarray:
        corr = self.decoder.decode(syndrome)
        return (self.dem_matrices.observables_matrix @ corr) % 2

    def initialize_decoder(self):
        self.dem_matrices = beliefmatching.detector_error_model_to_check_matrices(
            self.dem, allow_undecomposed_hyperedges=True
        )
        self.decoder = self.decoder_function(self.dem_matrices)


def run_simulation(
    code, distance, sector, noise_range, shots, decoder_func, code_name: str, overwrite=True
) -> list["sinter.TaskStats"]:
    file_name = f"{('_'.join(code_name.split())).lower()}_{sector}"
    filename = f"bposd_{file_name}.csv"

    if overwrite:
        if pathlib.Path(filename).is_file():
            pathlib.Path(filename).unlink()

    def generate_example_tasks():
        for noise in noise_range:
            stim_circuit = get_syndrome_extraction_circuit(code, noise, sector, distance)
            yield sinter.Task(
                circuit=stim_circuit,
                detector_error_model=stim_circuit.detector_error_model(decompose_errors=False),
                json_metadata={"noise": noise, "d": distance, "repetitions": distance},
            )

    samples = sinter.collect(
        num_workers=10,
        max_shots=shots,
        max_errors=100,
        tasks=generate_example_tasks(),
        decoders=["bposd"],
        custom_decoders={"bposd": CustomDecoder(decoder_func)},
        print_progress=True,
        save_resume_filepath=filename,
    )
    return samples


def print_results(samples: list["sinter.TaskStats"]) -> None:
    # Print samples as CSV data.
    print(sinter.CSV_HEADER)
    for sample in samples:
        print(sample.to_csv_line())


def plot_results(samples: list["sinter.TaskStats"], code_name, sector, noise_range):
    # Render a matplotlib plot of the data.
    fig, axis = plt.subplots(1, 1, sharey=True, figsize=(8, 6))
    sinter.plot_error_rate(
        ax=axis,
        stats=samples,
        group_func=lambda stat: f"{code_name} d={stat.json_metadata['d']}",
        filter_func=lambda stat: stat.decoder == "bposd",
        x_func=lambda stat: stat.json_metadata["noise"],
    )

    axis.set_ylabel("Logical Error Rate")
    axis.set_title(f"{code_name} threshold with BPOSD for {sector} stabilizers")

    axis.plot(noise_range, noise_range, "--", color="k")
    axis.loglog()
    axis.grid()
    axis.set_xlabel("Physical Error Rate")
    axis.legend()

    # Save to file and also open in a window.
    file_name = f"{('_'.join(code_name.split())).lower()}_{sector}"
    fig.savefig(f"{file_name}_plot.png")
    plt.show()
