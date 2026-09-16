"""
QuTSAE: Quantum Time Series Autoencoder
=========================================
Modernised port of the Qiskit implementation behind:
  Cybulski, J.L. & Zajac, S. (2024) "Design Considerations for Denoising
  Quantum Time Series Autoencoder", ICCS 2024, LNCS vol. 14837, pp. 252-267.

Ported from the original notebooks in
  github.com/ironfrown/ts_anomaly_detection_by_denoising
  (qiskit_qae/ts_qiskit_qae_angles_v3_03_testing_ideas.ipynb)

The port preserves the original design (angle encoding around |+>, full-QAE
training/testing circuit topology, cost = 1 - P(|0>^n), COBYLA optimisation)
but updates the Qiskit API surface to qiskit>=2.x / qiskit-aer>=0.17 /
qiskit-machine-learning>=0.9 (SamplerV2 primitives, `.compose()` instead of
opaque `.append(to_instruction())`, `real_amplitudes()` function instead of
the deprecated `RealAmplitudes` class).

Additionally implements the TwoLocal (Rx+Ry) ansatz variant with an `aw`
(additional-width / "extra qubits") parameter, as described in the paper's
Fig. 4 and Section 2.4 ("Ansatz size (width and depth)"), which was not
present as a standalone reusable function in the found reference repo.
"""

import random
import time

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import real_amplitudes, TwoLocal
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import SamplerQNN
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score,
)
from sklearn.utils import shuffle as data_shuffle


# ---------------------------------------------------------------------------
# Angle encoding (paper Sec 2.2, Fig 3): H then Ry(value) per qubit
# ---------------------------------------------------------------------------

def sequence_encoder(qubit_no, input_no=None, label="S"):
    """Build the input/output encoding block: H + Ry(value) per qubit.

    Values beyond `input_no` (extra/"aw" qubits) get a fixed Ry(0), i.e. they
    stay in the |+> state and carry no data - this is how the original code
    supports ansatz widths larger than the window size.
    """
    if input_no is None:
        input_no = qubit_no
    qr = QuantumRegister(qubit_no, "q")
    seq = QuantumCircuit(qr, name="sequence")
    used = 0
    for q in range(qubit_no):
        seq.h(q)
        if q >= input_no:
            seq.ry(0, q)
        else:
            p = Parameter(f"{label}({used})")
            seq.ry(p, q)
        used += 1
    return seq, list(seq.parameters)


### Default scaler = pi/2, matching the published paper's text (Sec 2.2):
### "a Ry rotation of value x 0.5*pi x (1 - 2*eps)".
def ts_relang_encode_val(val, scaler=np.pi / 2, err_range=0):
    return val * scaler * (1 - 2 * err_range)


def ts_relang_decode_val(val, scaler=np.pi / 2, err_range=0):
    return val / (scaler * (1 - 2 * err_range))


def ts_relang_encode(wind_set, scaler=np.pi / 2, err_range=0):
    encoded = np.array(
        [[ts_relang_encode_val(v, scaler, err_range) for v in wind] for wind in wind_set]
    )
    return encoded


def ts_relang_decode(wind_set, scaler=np.pi / 2, err_range=0):
    decoded = np.array(
        [[ts_relang_decode_val(v, scaler, err_range) for v in wind] for wind in wind_set]
    )
    return decoded


# ---------------------------------------------------------------------------
# Ansatz (paper Sec 2.4): width = num_qubits (+ aw extra), depth = reps
# ---------------------------------------------------------------------------

def ansatz(num_qubits, reps=2, ent="sca", label="A", rotation="ry"):
    """Build the encoder/decoder ansatz.

    rotation='ry'    -> original RealAmplitudes-equivalent (Ry-only), as used
                         in the reference repo's `ansatz()` helper.
    rotation='rxry'  -> TwoLocal with [Rx, Ry] rotation blocks, as described
                         in the paper (Fig. 4 caption, Sec 2.4) once the
                         Ry-only ansatz was found to limit trainability.
    """
    if rotation == "ry":
        return real_amplitudes(num_qubits, reps=reps, entanglement=ent, parameter_prefix=label)
    elif rotation == "rxry":
        # TwoLocal is a lazily-built BlueprintCircuit; decompose it into a
        # plain QuantumCircuit so it can be `.compose()`d and executed by
        # AerSimulator (which otherwise sees an opaque "TwoLocal" instruction).
        tl = TwoLocal(
            num_qubits,
            rotation_blocks=["rx", "ry"],
            entanglement_blocks="cx",
            entanglement=ent,
            reps=reps,
            parameter_prefix=label,
        )
        return tl.decompose()
    else:
        raise ValueError(f"Unknown rotation type: {rotation}")


# ---------------------------------------------------------------------------
# Full-QAE circuits (paper Fig. 1 = training, Fig. 2 = testing)
# ---------------------------------------------------------------------------

def train_qae(num_latent, num_trash, aw=0, reps=2, ent="sca", rotation="ry",
              in_seq_label="I", out_seq_label="O", enc_label="X", dec_label="Y"):
    """Build the Fig.1 training circuit: Input -> Encoder -> [reset trash] ->
    Decoder(inv) -> Output(inv) -> measure all.

    `aw` extra qubits widen the encoder/decoder ansatz beyond the
    latent+trash (=window) size without taking part in input/output encoding.
    """
    lt_qubits = num_latent + num_trash
    width = lt_qubits + aw
    qr = QuantumRegister(width, "q")
    cr = ClassicalRegister(width, "c")

    in_qc, _ = sequence_encoder(width, input_no=lt_qubits, label=in_seq_label)
    out_qc, _ = sequence_encoder(width, input_no=lt_qubits, label=out_seq_label)
    enc_qc = ansatz(width, reps=reps, ent=ent, label=enc_label, rotation=rotation)
    dec_qc = ansatz(width, reps=reps, ent=ent, label=dec_label, rotation=rotation)

    qc = QuantumCircuit(qr, cr)
    qc.compose(in_qc, qubits=range(width), inplace=True)
    qc.barrier()
    qc.compose(enc_qc, qubits=range(width), inplace=True)
    qc.barrier()
    for i in range(num_trash):
        qc.reset(num_latent + i)
    qc.barrier()
    qc.compose(dec_qc.inverse(), qubits=range(width), inplace=True)
    qc.barrier()
    qc.compose(out_qc.inverse(), qubits=range(width), inplace=True)
    qc.barrier()
    for i in range(width):
        qc.measure(qr[i], cr[i])

    in_out_params = list(in_qc.parameters) + list(out_qc.parameters)
    weight_params = list(enc_qc.parameters) + list(dec_qc.parameters)
    return qc, in_out_params, weight_params, width


def qae_test_circuit(num_latent, num_trash, aw=0, reps=2, ent="sca", rotation="ry",
                      meas_q=None, in_seq_label="I", enc_label="X", dec_label="Y"):
    """Build the Fig.2 testing circuit: Input -> Encoder -> [reset trash] ->
    Decoder(inv) -> optional single-qubit measurement.
    """
    lt_qubits = num_latent + num_trash
    width = lt_qubits + aw
    qr = QuantumRegister(width, "q")
    cr = ClassicalRegister(1, "meas")

    in_qc, in_params = sequence_encoder(width, input_no=lt_qubits, label=in_seq_label)
    enc_qc = ansatz(width, reps=reps, ent=ent, label=enc_label, rotation=rotation)
    dec_qc = ansatz(width, reps=reps, ent=ent, label=dec_label, rotation=rotation)

    qc = QuantumCircuit(qr, cr)
    qc.compose(in_qc, qubits=range(width), inplace=True)
    qc.barrier()
    qc.compose(enc_qc, qubits=range(width), inplace=True)
    qc.barrier()
    for i in range(num_trash):
        qc.reset(num_latent + i)
    qc.barrier()
    qc.compose(dec_qc.inverse(), qubits=range(width), inplace=True)
    if meas_q is not None:
        qc.barrier()
        qc.measure(meas_q, 0)

    weight_params = list(enc_qc.parameters) + list(dec_qc.parameters)
    return qc, in_params, weight_params, width


def single_qubit_angle_meas(qc, backend, shots=10000):
    """Run `qc` (with a single classical bit measured) and return the
    inferred qubit angle relative to |+> (H state = 0), exactly matching the
    reference notebook's formula: meas_angle = 2*arccos(amp0) - pi/2.
    Rotations left (negative) / right (positive) of |+> map to negative /
    positive values, per the paper's angle-encoding convention (Fig. 3a)."""
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    c0 = counts.get("0", 0)
    c1 = counts.get("1", 0)
    p0 = c0 / (c0 + c1) if (c0 + c1) else 0.5
    amp0 = np.sqrt(p0)
    meas_angle = 2 * np.arccos(np.clip(amp0, -1, 1)) - np.pi / 2
    return meas_angle


# ---------------------------------------------------------------------------
# Metrics on dictionaries of windows (paper uses these for R2/RMSE/MAE/MAPE)
# ---------------------------------------------------------------------------

def merged_tswind(wind_dict, trim_left=0, trim_right=0):
    out = []
    for k in sorted(wind_dict.keys()):
        w = wind_dict[k][trim_left:]
        w = w[:-trim_right] if trim_right > 0 else w
        out.extend(w)
    return out


def rms_tswin(exp_d, pred_d, **kw):
    e, p = merged_tswind(exp_d, **kw), merged_tswind(pred_d, **kw)
    return float(np.sqrt(mean_squared_error(e, p)))


def mae_tswin(exp_d, pred_d, **kw):
    e, p = merged_tswind(exp_d, **kw), merged_tswind(pred_d, **kw)
    return float(mean_absolute_error(e, p))


def mape_tswin(exp_d, pred_d, **kw):
    e, p = merged_tswind(exp_d, **kw), merged_tswind(pred_d, **kw)
    return float(mean_absolute_percentage_error(e, p))


def r2_tswin(exp_d, pred_d, **kw):
    e, p = merged_tswind(exp_d, **kw), merged_tswind(pred_d, **kw)
    return float(r2_score(e, p))


# ---------------------------------------------------------------------------
# Cost / training driver (paper Sec 2.3, cost = 1 - P(|0>^n))
# ---------------------------------------------------------------------------

def cost_zero_prob(probs):
    """cost = 1 - P(all qubits measured 0), averaged over the batch."""
    return 1.0 - np.sum(probs[:, 0]) / probs.shape[0]


def detect_device():
    """GPU if AerSimulator reports one (requires the `qiskit-aer-gpu` package
    + a working CUDA install), else CPU. Mirrors the reference notebook's
    own "use GPU when present, otherwise CPU" pattern."""
    try:
        devices = AerSimulator().available_devices()
    except Exception:
        devices = ("CPU",)
    return "GPU" if "GPU" in devices else "CPU"


class QuTSAETrainer:
    """Trains a full-QAE denoising model (paper Fig. 1) with COBYLA and
    optionally tracks train/valid MAE during training via periodic testing
    (paper Fig. 2 circuit, single-qubit measurement decode)."""

    def __init__(self, num_latent, num_trash, aw=0, reps=2, ent="sca", rotation="ry",
                 shots=10000, seed=2023, device=None):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.aw = aw
        self.reps = reps
        self.ent = ent
        self.rotation = rotation
        self.shots = shots
        self.seed = seed
        self.device = device or detect_device()

        self.train_qc, self.in_out_params, self.weight_params, self.width = train_qae(
            num_latent, num_trash, aw=aw, reps=reps, ent=ent, rotation=rotation,
        )

        backend_opts = {"seed_simulator": seed, "method": "statevector"}
        if self.device == "GPU":
            backend_opts.update({"device": "GPU", "cuStateVec_enable": True})
        else:
            backend_opts["device"] = "CPU"

        self.sampler = SamplerV2(
            options={"backend_options": backend_opts, "run_options": {"seed": seed}}
        )
        self.backend = AerSimulator(**backend_opts)
        print(f"[QuTSAETrainer] running on device={self.device}")

        self.qnn = SamplerQNN(
            circuit=self.train_qc,
            input_params=self.in_out_params,
            weight_params=self.weight_params,
            interpret=lambda x: x,
            output_shape=2 ** self.width,
            sampler=self.sampler,
        )

        self.objective_func_vals = []
        self.mae_train_vals = []
        self.mae_valid_vals = []
        self.params_hist = []

    def _cost_fun(self, params_values, train_set, test_ctx, rng):
        probs = self.qnn.forward(train_set, params_values)
        cost = cost_zero_prob(probs)
        self.objective_func_vals.append(float(cost))
        self.params_hist.append(params_values)

        if test_ctx is not None:
            train_mae = self._quick_test(params_values, test_ctx, "train", rng)
            valid_mae = self._quick_test(params_values, test_ctx, "valid", rng)
            self.mae_train_vals.append(train_mae)
            self.mae_valid_vals.append(valid_mae)
        return cost

    def _quick_test(self, weight_values, test_ctx, split, rng, sample=10):
        pure = test_ctx[f"{split}_pure"]
        noisy = test_ctx[f"{split}_noisy"]
        idxs = rng.sample(range(len(pure)), min(sample, len(pure)))
        in_org, out_rec = {}, {}
        for idx in idxs:
            wind = noisy[idx]
            out_meas = self._reconstruct_one(weight_values, wind)
            in_org[idx] = list(pure[idx])
            out_rec[idx] = out_meas
        return mae_tswin(in_org, out_rec)

    def _reconstruct_one(self, weight_values, wind):
        """Reconstruct a single window, binding parameters by NAME (each call
        to qae_test_circuit() creates fresh Parameter objects, even with the
        same labels, so binding must not rely on object identity)."""
        name_to_val = {}
        out_meas = []
        for meas_q in range(len(wind)):
            out_qc, in_params, weight_params, _ = qae_test_circuit(
                self.num_latent, self.num_trash, aw=self.aw, reps=self.reps,
                ent=self.ent, rotation=self.rotation, meas_q=meas_q,
            )
            name_to_val = {p.name: v for p, v in zip(in_params, wind)}
            name_to_val.update({p.name: v for p, v in zip(self.weight_params, weight_values)})
            bind = {p: name_to_val[p.name] for p in out_qc.parameters}
            out_qc = out_qc.assign_parameters(bind)
            out_qc = out_qc.decompose(reps=3)
            out_meas.append(single_qubit_angle_meas(out_qc, self.backend, shots=self.shots))
        return out_meas

    def fit(self, y_train_noisy_enc, y_train_enc, epochs=500, shuffle=True,
            test_ctx=None, init_vals=None, seed=None, log_every=50):
        rng = random.Random(seed if seed is not None else self.seed)
        train_set = np.array(
            [list(n) + list(p) for n, p in zip(y_train_noisy_enc, y_train_enc)]
        )
        if shuffle:
            train_set = data_shuffle(train_set, random_state=self.seed)

        if init_vals is None:
            np.random.seed(seed if seed is not None else self.seed)
            init_vals = np.random.random(len(self.weight_params))

        opt = COBYLA(maxiter=epochs)

        def wrapped_cost(w):
            c = self._cost_fun(w, train_set, test_ctx, rng)
            it = len(self.objective_func_vals)
            if log_every and it % log_every == 0:
                msg = f"  iter {it}/{epochs}  cost={c:.4f}"
                if test_ctx is not None:
                    msg += f"  train_mae={self.mae_train_vals[-1]:.4f}  valid_mae={self.mae_valid_vals[-1]:.4f}"
                print(msg)
            return c

        t0 = time.time()
        result = opt.minimize(fun=wrapped_cost, x0=init_vals)
        elapsed = time.time() - t0
        return {
            "optimum_parameters": result.x,
            "minimum_cost": float(result.fun),
            "elapsed_sec": elapsed,
            "n_weight_params": len(self.weight_params),
            "objective_func_vals": self.objective_func_vals,
            "mae_train_vals": self.mae_train_vals,
            "mae_valid_vals": self.mae_valid_vals,
        }

    def reconstruct(self, weight_values, windows_noisy):
        """Full reconstruction of a set of noisy windows using the trained
        weights (paper Fig. 2 circuit, per-qubit measurement decode)."""
        return np.array([self._reconstruct_one(weight_values, wind) for wind in windows_noisy])
