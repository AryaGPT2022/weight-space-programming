"""Model I/O for (possibly structurally edited) BERT cross-encoders.

transformers 5.x removed BERT head pruning, but BertSelfAttention reshapes with
(-1, head_size), so physically shrinking Q/K/V/O and FFN Linears just works. We record
the surviving structure in `wsp_arch.json` next to the safetensors so a checkpoint can be
rebuilt exactly:

    {
      "keep_layers": [0,1,2,3,4,5],          # original layer ids, in order
      "layers": {                             # keyed by ORIGINAL layer id
        "0": {"heads": [0..11], "ffn": [0..1535], "low_rank": {"attention.self.query": 16}},
        ...
      },
      "vocab_keep": null | int               # rows kept in a remapped word-embedding table
    }
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
from wsp.config import BASE_ID, BASE_REVISION
HEAD_DIM = 32
MAX_LEN = 256


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ----------------------------------------------------------------------------- custom modules
class LowRankLinear(nn.Module):
    """W ≈ B @ A  (out×r)(r×in); keeps the original bias."""

    def __init__(self, in_f, out_f, r, bias=True):
        super().__init__()
        self.in_features, self.out_features, self.rank = in_f, out_f, r
        self.A = nn.Linear(in_f, r, bias=False)
        self.B = nn.Linear(r, out_f, bias=bias)

    def forward(self, x):
        return self.B(self.A(x))

    @property
    def weight(self):  # effective weight, for inspection only
        return self.B.weight @ self.A.weight


class NullSelfAttention(nn.Module):
    """A layer whose heads were all removed: contributes nothing but output.dense's bias."""

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states.new_zeros(*hidden_states.shape[:-1], 0), None


class RemappedEmbedding(nn.Module):
    """Word embeddings with a shrunken table; `remap` maps original token id -> new row."""

    def __init__(self, n_orig, n_keep, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_keep, dim))
        self.register_buffer("remap", torch.zeros(n_orig, dtype=torch.long))

    def forward(self, ids):
        return F.embedding(self.remap[ids], self.weight)


def _cls_only_forward(self, hidden_states, attention_mask=None, *args, **kwargs):
    """Last-layer rewrite: the classifier reads only the [CLS] row of the final layer, so compute
    the final layer's queries, attention, residual and FFN for position 0 only. Keys/values still
    come from every token. Mathematically identical output for the classifier; ~1/L less compute."""
    x, cls = hidden_states, hidden_states[:, :1]
    s = self.attention.self
    B, T, _ = x.shape
    if isinstance(s, NullSelfAttention):
        ctx = x.new_zeros(B, 1, 0)
    else:
        nh = s.num_attention_heads
        q = s.query(cls).view(B, 1, nh, HEAD_DIM).transpose(1, 2)
        k = s.key(x).view(B, T, nh, HEAD_DIM).transpose(1, 2)
        v = s.value(x).view(B, T, nh, HEAD_DIM).transpose(1, 2)
        sc = (q @ k.transpose(2, 3)) * s.scaling
        if attention_mask is not None:
            m = attention_mask[..., :1, :] if attention_mask.dim() == 4 else attention_mask
            sc = sc.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else sc + m
        ctx = (sc.softmax(-1) @ v).transpose(1, 2).reshape(B, 1, nh * HEAD_DIM)
    a = self.attention.output(ctx, cls)
    return self.output(self.intermediate(a), a)


def make_cls_only(layer):
    layer.forward = _cls_only_forward.__get__(layer)


# ----------------------------------------------------------------------------- arch handling
def default_arch(cfg):
    L, H, I = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.intermediate_size
    return {"keep_layers": list(range(L)),
            "layers": {str(l): {"heads": list(range(H)), "ffn": list(range(I)), "low_rank": {}}
                       for l in range(L)},
            "vocab_keep": None}


def _get(mod, dotted):
    for p in dotted.split("."):
        mod = getattr(mod, p)
    return mod


def _set(mod, dotted, new):
    *parents, last = dotted.split(".")
    for p in parents:
        mod = getattr(mod, p)
    setattr(mod, last, new)


def _shape_layer(layer, spec, hidden):
    """Resize a BertLayer's modules (fresh, uninitialised) to match `spec`."""
    nh, nf = len(spec["heads"]), len(spec["ffn"])
    att = layer.attention
    if nh == 0:
        att.self = NullSelfAttention()
    else:
        for n in ("query", "key", "value"):
            setattr(att.self, n, nn.Linear(hidden, nh * HEAD_DIM))
        att.self.num_attention_heads = nh
        att.self.all_head_size = nh * HEAD_DIM
    att.output.dense = nn.Linear(nh * HEAD_DIM, hidden)
    layer.intermediate.dense = nn.Linear(hidden, nf)
    layer.output.dense = nn.Linear(nf, hidden)
    for name, r in spec.get("low_rank", {}).items():
        lin = _get(layer, name)
        _set(layer, name, LowRankLinear(lin.in_features, lin.out_features, r, lin.bias is not None))


def build_from_arch(cfg, arch):
    cfg = copy.deepcopy(cfg)
    cfg.num_hidden_layers = len(arch["keep_layers"])
    cfg._attn_implementation = "eager"
    model = AutoModelForSequenceClassification.from_config(cfg)
    for pos, orig in enumerate(arch["keep_layers"]):
        _shape_layer(model.bert.encoder.layer[pos], arch["layers"][str(orig)], cfg.hidden_size)
    if arch.get("vocab_keep"):
        model.bert.embeddings.word_embeddings = RemappedEmbedding(cfg.vocab_size, arch["vocab_keep"],
                                                                  cfg.hidden_size)
    return model


# ----------------------------------------------------------------------------- load / save
def load(path=None, dev=None):
    """Returns (model, tokenizer, arch). `path` None/BASE_ID -> pristine base model."""
    dev = dev or device()
    if path is None or str(path) == BASE_ID or not (Path(path) / "wsp_arch.json").exists():
        src = BASE_ID if path is None else str(path)
        rev = BASE_REVISION if src == BASE_ID else None   # pin the hub download; local dirs have no revision
        model = AutoModelForSequenceClassification.from_pretrained(src, revision=rev, attn_implementation="eager")
        tok = AutoTokenizer.from_pretrained(src, revision=rev)
        arch = default_arch(model.config)
    else:
        path = Path(path)
        cfg = AutoConfig.from_pretrained(path)  # nosec B615 - local checkpoint directory, not a hub download
        arch = json.loads((path / "wsp_arch.json").read_text())
        model = build_from_arch(cfg, arch)
        sd = load_file(path / "model.safetensors")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.endswith("position_ids")]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
        tok = AutoTokenizer.from_pretrained(path)  # nosec B615 - local checkpoint directory
    if arch.get("cls_only_last"):
        make_cls_only(model.bert.encoder.layer[-1])
    model.eval().to(dev)
    return model, tok, arch


def save(model, tok, arch, path, meta=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, path / "model.safetensors")
    model.config.save_pretrained(path)
    tok.save_pretrained(path)
    (path / "wsp_arch.json").write_text(json.dumps(arch))
    if meta is not None:
        (path / "edit_meta.json").write_text(json.dumps(meta, indent=2))
    return sha256(path / "model.safetensors")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- scoring
@torch.inference_mode()
def score(model, tok, queries, texts, bs=64, max_len=MAX_LEN, want=()):
    """Relevance logits for (query, passage) pairs. `want` ⊆ {"attentions","hidden"} returns
    per-batch lists of numpy arrays as well (only used by the budgeted harness)."""
    dev = next(model.parameters()).device
    out, extra = [], {k: [] for k in want}
    for i in range(0, len(queries), bs):
        enc = tok(list(queries[i:i + bs]), list(texts[i:i + bs]), padding=True, truncation="only_second",
                  max_length=max_len, return_tensors="pt").to(dev)
        o = model(**enc, output_attentions="attentions" in want, output_hidden_states="hidden" in want)
        out.append(o.logits[:, 0].float().cpu())
        if "attentions" in want:
            extra["attentions"].append([a.float().cpu().numpy() for a in o.attentions])
        if "hidden" in want:
            extra["hidden"].append([h.float().cpu().numpy() for h in o.hidden_states])
        if want:
            extra.setdefault("input_ids", []).append(enc["input_ids"].cpu().numpy())
    logits = torch.cat(out).numpy() if out else np.zeros(0)
    return (logits, extra) if want else logits


# ----------------------------------------------------------------------------- size / speed
def param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    emb = sum(p.numel() for n, p in model.named_parameters() if ".embeddings." in n)
    return {"params_total": total, "params_embedding": emb, "params_encoder_head": total - emb}


@torch.inference_mode()
def throughput(model, tok, n_pairs=512, bs=32, seq=MAX_LEN, dev="cpu", warmup=2):
    model = model.to(dev)
    ids = torch.randint(1000, 20000, (bs, seq), device=dev)
    enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
           "token_type_ids": torch.cat([torch.zeros(bs, 32, dtype=torch.long, device=dev),
                                        torch.ones(bs, seq - 32, dtype=torch.long, device=dev)], 1)}
    sync = (lambda: torch.mps.synchronize()) if str(dev) == "mps" else (lambda: None)
    for _ in range(warmup):
        model(**enc)
    sync()
    t = time.perf_counter()
    n = 0
    while n < n_pairs:
        model(**enc)
        n += bs
    sync()
    return n / (time.perf_counter() - t)
