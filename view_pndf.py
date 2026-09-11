"""
View-PNDF core module
=====================

View-specific Pattern Neuron Detection and Fine-tuning for consistent
radiology report generation (RRG).

This module implements, for a medical VLM whose language decoder is Phi-3
(``Mini-InternVL2-4B-DA-Medical`` = InternViT vision tower + Phi3ForCausalLM),
the three building blocks of View-PNDF:

  * VND  -- View-specific Neuron Detection.  We run image+prompt samples of a
            given view (frontal / lateral) through the *whole* VLM and collect,
            per decoder layer and per projection, the activation magnitude of
            every neuron (paper Eq. 6, ``ActivationScore = ||X_l(p, w)||``).
            Per-sample top-ratio neurons are intersected across samples
            (paper Eq. 7) to obtain the view's pattern neurons.

  * VNV  -- View-specific Neuron Verification.  We zero the detected neurons
            (and, for the control, an equal number of random neurons) and
            observe the change in generated reports.

  * VNF  -- View-specific Neuron Fine-tuning.  Only the detected pattern
            neurons receive gradient updates (see ``mask_grad_to_neurons``).

We do NOT rewrite the model file.  We attach ``forward`` hooks to the
*unmodified*, officially loaded decoder layers.  This keeps weight loading
risk-free and makes the code architecture-agnostic given a small
``ProjectionSpec``.

Phi-3 uses FUSED projections, which the spec below encodes:
  * ``self_attn.qkv_proj`` : [ (n_heads+2*n_kv)*head_dim , hidden ]
        rows  [0 : q_dim)                 -> q neurons
        rows  [q_dim : q_dim+kv_dim)      -> k neurons
        rows  [q_dim+kv_dim : q_dim+2kv)  -> v neurons
  * ``self_attn.o_proj``   : [ hidden , n_heads*head_dim ]
        columns (input dim) -> o neurons
  * ``mlp.gate_up_proj``   : [ 2*intermediate , hidden ]
        rows  [0 : intermediate)             -> gate  (not tracked)
        rows  [intermediate : 2*intermediate)-> up neurons
  * ``mlp.down_proj``      : [ hidden , intermediate ]
        columns (input dim) -> down neurons
"""

import json
import itertools
from dataclasses import dataclass

import numpy as np
import torch


# The six neuron groups, matching the JSON schema used by detect/deactivation.
GROUPS = ["fwd_up", "fwd_down", "attn_q", "attn_k", "attn_v", "attn_o"]


@dataclass
class ProjectionSpec:
    """Describes where each neuron group lives inside a decoder layer.

    ``q_dim`` / ``kv_dim`` are the slice widths inside the fused qkv output;
    ``intermediate`` is the slice width inside the fused gate_up output.
    """

    q_dim: int
    kv_dim: int
    intermediate: int
    num_layers: int
    # kv_factor = n_heads / n_kv_heads.  For Phi-3 (no GQA) this is 1.
    kv_factor: int = 1


def phi3_spec_from_model(language_model) -> ProjectionSpec:
    """Build a ProjectionSpec from a loaded Phi3ForCausalLM config.

    The dimensions are read straight from the HF config, so this also works for
    any decoder that shares Phi-3's *fused* ``qkv_proj`` / ``gate_up_proj``
    layout (e.g. Phi-3.5).  For decoders with *split* projections (LLaMA-style
    q/k/v/gate/up), add a spec + hook variant -- see ``build_projection_spec``.
    """
    cfg = language_model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    q_dim = cfg.num_attention_heads * head_dim
    kv_dim = cfg.num_key_value_heads * head_dim
    return ProjectionSpec(
        q_dim=q_dim,
        kv_dim=kv_dim,
        intermediate=cfg.intermediate_size,
        num_layers=cfg.num_hidden_layers,
        kv_factor=cfg.num_attention_heads // cfg.num_key_value_heads,
    )


def build_projection_spec(model) -> ProjectionSpec:
    """Architecture-agnostic entry point.

    Given a loaded VLM (or a bare CausalLM), return the ProjectionSpec describing
    where each neuron group lives.  Currently wired for the InternVL medical
    backbone (Phi-3 fused projections).  To support another backbone
    (MedGemma / LLaVA-Med / Hulu-Med), branch on ``type(...).__name__`` here and
    return the matching spec; the fused-vs-split layout also dictates which hook
    variant ``ActivationCollector`` should attach.
    """
    lm = getattr(model, "language_model", model)
    return phi3_spec_from_model(lm)


def get_decoder_layers(model):
    """Return the list of Phi-3 decoder layers from an InternVLChatModel
    (or a bare Phi3ForCausalLM)."""
    lm = getattr(model, "language_model", model)
    return lm.model.layers


def merge_image_embeds(model, pixel_values, input_ids):
    """Replicate InternVLChatModel's image/text embedding merge without going
    through its distributed-only training ``forward``.

    Returns ``inputs_embeds`` (B, N, C) with the ``<IMG_CONTEXT>`` token
    positions replaced by projected vision features.  ``model.img_context_token_id``
    must already be set by the caller.
    """
    if pixel_values is not None:
        vit_embeds = model.extract_feature(pixel_values)
        input_embeds = model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        flat_ids = input_ids.reshape(B * N)
        selected = flat_ids == model.img_context_token_id
        assert selected.sum() != 0, "no <IMG_CONTEXT> tokens found in input_ids"
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device, input_embeds.dtype)
        return input_embeds.reshape(B, N, C)
    return model.language_model.get_input_embeddings()(input_ids)


# --------------------------------------------------------------------------- #
# VND: activation-score collection via forward hooks
# --------------------------------------------------------------------------- #
class ActivationCollector:
    """Attaches forward hooks to every decoder layer and accumulates, for the
    tokens of a single forward pass, the per-neuron activation magnitude
    (sum of abs over the sequence dimension) for each of the six groups.

    Usage::

        collector = ActivationCollector(model, spec)
        collector.attach()
        with torch.no_grad():
            model(pixel_values=..., input_ids=..., attention_mask=...)
        scores = collector.pop()          # {group: {layer: np.ndarray}}
        ...
        collector.detach()
    """

    def __init__(self, model, spec: ProjectionSpec):
        self.spec = spec
        self.layers = get_decoder_layers(model)
        self._handles = []
        self._scores = {g: {} for g in GROUPS}

    # -- hook factories ---------------------------------------------------- #
    def _qkv_hook(self, layer_idx):
        s = self.spec

        def hook(module, inputs, output):
            # output: (batch, seq, q_dim + 2*kv_dim)
            act = output.detach().float().abs().sum(dim=1).squeeze(0)  # (proj_out,)
            q = act[: s.q_dim]
            k = act[s.q_dim : s.q_dim + s.kv_dim]
            v = act[s.q_dim + s.kv_dim : s.q_dim + 2 * s.kv_dim]
            self._scores["attn_q"][layer_idx] = q.cpu().numpy()
            self._scores["attn_k"][layer_idx] = k.cpu().numpy()
            self._scores["attn_v"][layer_idx] = v.cpu().numpy()

        return hook

    def _oproj_hook(self, layer_idx):
        def hook(module, inputs, output):
            # o neurons live on the INPUT dim of o_proj (the attn output).
            act = inputs[0].detach().float().abs().sum(dim=1).squeeze(0)
            self._scores["attn_o"][layer_idx] = act.cpu().numpy()

        return hook

    def _gateup_hook(self, layer_idx):
        s = self.spec

        def hook(module, inputs, output):
            # output: (batch, seq, 2*intermediate); up = second half.
            act = output.detach().float().abs().sum(dim=1).squeeze(0)
            up = act[s.intermediate : 2 * s.intermediate]
            self._scores["fwd_up"][layer_idx] = up.cpu().numpy()

        return hook

    def _down_hook(self, layer_idx):
        def hook(module, inputs, output):
            # down neurons live on the INPUT dim of down_proj (intermediate).
            act = inputs[0].detach().float().abs().sum(dim=1).squeeze(0)
            self._scores["fwd_down"][layer_idx] = act.cpu().numpy()

        return hook

    # -- lifecycle --------------------------------------------------------- #
    def attach(self):
        for idx, layer in enumerate(self.layers):
            self._handles.append(layer.self_attn.qkv_proj.register_forward_hook(self._qkv_hook(idx)))
            self._handles.append(layer.self_attn.o_proj.register_forward_hook(self._oproj_hook(idx)))
            self._handles.append(layer.mlp.gate_up_proj.register_forward_hook(self._gateup_hook(idx)))
            self._handles.append(layer.mlp.down_proj.register_forward_hook(self._down_hook(idx)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def pop(self):
        """Return the scores collected during the last forward and reset."""
        scores = self._scores
        self._scores = {g: {} for g in GROUPS}
        return scores


def topk_indices_per_layer(scores, atten_ratio, ffn_ratio):
    """Given one sample's ``{group: {layer: np.ndarray}}`` scores, return the
    top-ratio neuron indices per layer per group (paper Eq. 7 candidate set).

    FFN groups (fwd_up/fwd_down) use ``ffn_ratio``; attention groups use
    ``atten_ratio``.
    """
    selected = {g: {} for g in GROUPS}
    for group in GROUPS:
        ratio = ffn_ratio if group.startswith("fwd") else atten_ratio
        for layer, arr in scores[group].items():
            top_n = int(ratio * len(arr))
            if top_n <= 0:
                selected[group][layer] = set()
            else:
                idx = np.argsort(arr)[-top_n:][::-1]
                selected[group][layer] = set(int(x) for x in idx)
    return selected


def intersect_sample_sets(per_sample_selected):
    """Intersect the candidate sets across all samples (paper Eq. 7, ``∩``).

    ``per_sample_selected`` is a list of ``{group: {layer: set}}`` dicts.
    Returns ``{group: {layer_str: set}}`` (layer keys are strings to match the
    JSON schema consumed by deactivation.py / train.py).
    """
    result = {g: {} for g in GROUPS}
    if not per_sample_selected:
        return result
    for group in GROUPS:
        layers = per_sample_selected[0][group].keys()
        for layer in layers:
            common = None
            for sample in per_sample_selected:
                s = sample[group].get(layer, set())
                common = s if common is None else (common & s)
            result[group][str(layer)] = common if common is not None else set()
    return result


def save_neuron(activate_neurons, path):
    """Serialise ``{group: {layer: set}}`` to JSON (sets -> lists)."""
    out = {}
    for group, layers in activate_neurons.items():
        out[group] = {str(k): sorted(int(x) for x in v) for k, v in layers.items()}
    with open(path, "w") as f:
        json.dump(out, f)


# --------------------------------------------------------------------------- #
# VNV: deactivation (zero neurons) for Phi-3 fused projections
# --------------------------------------------------------------------------- #
def build_deactivate_indices_dict(detected_neuron, spec: ProjectionSpec):
    """Map detected neuron groups to (weight name, dim, indices) for zeroing.

    Handles Phi-3 fused ``qkv_proj`` / ``gate_up_proj`` by offsetting indices
    into the fused rows.
    """
    q = detected_neuron["attn_q"]
    k = detected_neuron["attn_k"]
    v = detected_neuron["attn_v"]
    o = detected_neuron["attn_o"]
    up = detected_neuron["fwd_up"]
    down = detected_neuron["fwd_down"]

    dd = {}
    for idx in range(spec.num_layers):
        li = str(idx)
        prefix = f"language_model.model.layers.{idx}"

        # Fused qkv_proj rows (dim 0): q | k | v.
        qkv_rows = []
        if li in q:
            qkv_rows += [int(j) for j in q[li]]
        if li in k:
            qkv_rows += [spec.q_dim + int(j) // spec.kv_factor for j in k[li]]
        if li in v:
            qkv_rows += [spec.q_dim + spec.kv_dim + int(j) // spec.kv_factor for j in v[li]]
        if qkv_rows:
            dd[f"{prefix}.self_attn.qkv_proj.weight"] = {"dim": 0, "indices": sorted(set(qkv_rows))}

        # o_proj input dim (dim 1).
        if li in o and o[li]:
            dd[f"{prefix}.self_attn.o_proj.weight"] = {"dim": 1, "indices": sorted({int(j) for j in o[li]})}

        # Fused gate_up_proj rows (dim 0): up lives in the second half.
        if li in up and up[li]:
            rows = sorted({spec.intermediate + int(j) for j in up[li]})
            dd[f"{prefix}.mlp.gate_up_proj.weight"] = {"dim": 0, "indices": rows}

        # down_proj input dim (dim 1).
        if li in down and down[li]:
            dd[f"{prefix}.mlp.down_proj.weight"] = {"dim": 1, "indices": sorted({int(j) for j in down[li]})}

    return dd


def apply_zero_mask_to_model(model, deactivate_dict):
    """Zero the specified rows/columns in-place on a loaded model."""
    state_dict = model.state_dict()
    new_state_dict = {}
    for name, param in state_dict.items():
        if name in deactivate_dict:
            info = deactivate_dict[name]
            param = param.clone()
            if info["dim"] == 0:
                param[info["indices"], :] = 0
            else:
                param[:, info["indices"]] = 0
        new_state_dict[name] = param
    model.load_state_dict(new_state_dict)


# --------------------------------------------------------------------------- #
# VNF: restrict gradients to the detected pattern neurons
# --------------------------------------------------------------------------- #
def build_grad_mask(model, detected_neuron, spec: ProjectionSpec, device):
    """Return ``{param_name: boolean mask}`` that is True only on the detected
    view-specific neuron slices.  Used to zero out gradients everywhere else so
    that fine-tuning updates ONLY the pattern neurons (paper VNF)."""
    dd = build_deactivate_indices_dict(detected_neuron, spec)
    masks = {}
    param_shapes = {n: p.shape for n, p in model.named_parameters()}
    for name, info in dd.items():
        if name not in param_shapes:
            continue
        mask = torch.zeros(param_shapes[name], dtype=torch.bool, device=device)
        if info["dim"] == 0:
            mask[info["indices"], :] = True
        else:
            mask[:, info["indices"]] = True
        masks[name] = mask
    return masks


def register_grad_mask_hooks(model, masks):
    """Attach backward hooks so each parameter keeps gradients only where its
    mask is True.  Returns the hook handles."""
    handles = []
    name_to_param = dict(model.named_parameters())
    for name, mask in masks.items():
        p = name_to_param.get(name)
        if p is None:
            continue
        p.requires_grad_(True)

        def make_hook(m):
            return lambda grad: grad * m

        handles.append(p.register_hook(make_hook(mask)))
    return handles


def read_neuron(path):
    """Load a neuron JSON file into ``{group: {layer_str: set}}``."""
    with open(path, "r") as f:
        data = json.load(f)
    for group in data:
        data[group] = {k: set(v) if isinstance(v, list) else v for k, v in data[group].items()}
    return data
