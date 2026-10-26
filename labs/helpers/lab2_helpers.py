"""Provided machinery for Lab 2, Language Model Systems. You do not need to read this file.

The lab notebook downloads it, checks its digest, and imports it. Every function that touches your
own work takes your function as an argument: the grader reads answers with your extract_answer, the
fine-tuning loss and the log-likelihood go through your masked_logprob, and the quantizers are
yours. Everything here was used to produce the lab's cached results, on a Tesla T4.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import statistics
import time
import urllib.request
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

S0_SYSTEM = "Reply with a single number and nothing else."
S1_SYSTEM = 'Solve the problem step by step. End with a line of the form "The answer is N."'
SYSTEMS = {
    "S0": {"system": S0_SYSTEM, "shots": 0, "max_new_tokens": 512},
    "S1": {"system": S1_SYSTEM, "shots": 0, "max_new_tokens": 512},
    "S2": {"system": S1_SYSTEM, "shots": 4, "max_new_tokens": 512},
}
LORA_CONFIG = {"r": 16, "alpha": 32, "dropout": 0.05, "lr": 2e-4, "epochs": 2, "batch": 8, "micro_batch": 2}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SCALE_FACTORS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)

# Computed once from MODEL_ID at REVISION. A CPU-only learner uses these for the
# ungraded memory table and the graded D1 probe, rather than loading a 0.5B model
# merely to count its parameters.
CACHED_TRAIN_TOKENS = 227_072
CACHED_WEIGHT_BYTES = {
    64: {
        "parts": {
            "embedding": {"fp16": 272_269_312, "int4": 272_269_312},
            "attention": {"fp16": 88_135_680, "int4": 23_451_648.0},
            "mlp": {"fp16": 627_572_736, "int4": 166_699_008.0},
            "other": {"fp16": 87_808, "int4": 87_808},
        },
        "fp16": 988_065_536,
        "int4": 462_507_776.0,
        "ratio": 0.4680942297333605,
    },
}


def cached_weight_bytes(group_size):
    """The pilot's weight-byte report for a group size supported by the learner lab."""
    if group_size not in CACHED_WEIGHT_BYTES:
        raise ValueError(f"no cached weight-byte report for group size {group_size}")
    report = CACHED_WEIGHT_BYTES[group_size]
    return {"parts": {part: dict(row) for part, row in report["parts"].items()},
            "fp16": report["fp16"], "int4": report["int4"], "ratio": report["ratio"]}


def _check_code(code):
    if not isinstance(code, int) or not 2000 <= code < 2020:
        raise ValueError("copy the four-digit PANEL_CODE from checkpoint 2")


def personal_panel(data, code, count=200):
    """The fixed test panel issued to one learner.

    Codes 2000 to 2019 define the twenty course variants. Every system in a
    learner's live comparison must use the same ordered IDs so the comparison
    remains paired. The full 1,269-problem record stays separate.
    """
    _check_code(code)
    if not 1 <= count <= len(data["full"]):
        raise ValueError("panel size must fit inside the fixed test set")
    return sorted(random.Random(460_000 + code).sample(list(data["full"]), count))


def panel_fingerprint(ids):
    """A short order-sensitive check that the issued panel was copied correctly."""
    return sum((i + 1) * problem_id for i, problem_id in enumerate(ids)) % 1_000_000


# The code also fixes two choices that no saved record contains, so what the live run measures on them is the
# learner's own: the LoRA training seed, and the 128 calibration solutions behind task 5's Sigma. The saved records
# use seeds 0 to 2 and the pool's first 128 solutions, and no code maps to either.

def personal_seed(code):
    """The LoRA training seed for an issued panel code."""
    _check_code(code)
    return 1000 + code


def personal_calibration(data, code, count=128):
    """The calibration solutions for an issued panel code, drawn from the lab's pool of training problems."""
    _check_code(code)
    pool = data["calibration_pool"]
    picked = sorted(random.Random(470_000 + code).sample(range(len(pool)), count))
    return [pool[k] for k in picked]


def final_epoch_loss(losses, n_examples, cfg=LORA_CONFIG):
    """The mean training loss over the last epoch's steps, as train_lora reports them."""
    per_epoch = math.ceil(n_examples / cfg["batch"])
    return statistics.fmean(losses[-per_epoch:])


# The environment the checkpoint 2 keys were computed in, by the staff key run on the course runtime. None until
# that run is recorded. The keyed values depend on the GPU and on these library versions.
KEY_ENVIRONMENT = None


def environment():
    """The GPU and library versions this session computes with."""
    import importlib.metadata as metadata

    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda, "torch": torch.__version__.split("+")[0],
            "transformers": version("transformers"), "peft": version("peft")}


def environment_differences(env, keyed=None):
    """Where this session differs from the environment the keys were computed in."""
    keyed = KEY_ENVIRONMENT if keyed is None else keyed
    if not keyed:
        return []
    return [f"{k}: {env.get(k)} here, {keyed[k]} for the keys" for k in keyed if env.get(k) != keyed[k]]


def checklist(conditions):
    """Return a pass flag and specific feedback for a notebook contract."""
    unmet = [name for name, ok in conditions.items() if not ok]
    if not unmet:
        return 1, "  every requirement met"
    return 0, "\n".join(f"  not yet: {name}" for name in unmet)


# ------------------------------------------------------------------------------------------------ files

def fetch(name, digest, base, folder):
    """Download one file into folder unless an intact copy is there; stop if its digest differs."""
    path = Path(folder) / name
    if not (path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest):
        path.parent.mkdir(parents=True, exist_ok=True)
        if base.startswith("http"):
            urllib.request.urlretrieve(base + name, path)
        else:
            path.write_bytes((Path(base) / name).read_bytes())
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"{name} did not download intact. Delete {path} and run this cell again.")
    return path


def load_data(path):
    """The problems, the training and calibration solutions, and every cached run, from lab2_data.json.gz."""
    data = json.loads(gzip.decompress(Path(path).read_bytes()).decode("utf-8"))
    data["problems"] = {int(k): v for k, v in data["problems"].items()}
    return data


# ------------------------------------------------------------------------------------------------ scoring

def score(raws, golds, extract_answer):
    """Correct and unreadable flags for each raw answer, read with your extract_answer."""
    preds = [extract_answer(r) for r in raws]
    return {"pred": preds, "correct": [p is not None and p == g for p, g in zip(preds, golds)],
            "unreadable": [p is None for p in preds]}


def cached(data, name, extract_answer, split="full"):
    """A cached run, scored with your extract_answer. split is "full" (1,269 problems), "200" or "dev"."""
    runs = {"full": data["runs"], "200": data["runs_200"], "dev": data["runs_dev"]}[split]
    ids = {"full": data["full"], "200": data["test200"], "dev": data["dev"]}[split]
    run = runs[name]
    golds = [data["problems"][i]["answer"] for i in ids]
    return {"name": name, "ids": ids, **run, **score(run["raw"], golds, extract_answer)}


def summary(run):
    n = len(run["correct"])
    return {"n": n, "right": sum(run["correct"]), "unreadable": sum(run["unreadable"]),
            "wrong": n - sum(run["correct"]) - sum(run["unreadable"]), "accuracy": sum(run["correct"]) / n,
            "prompt_tokens": statistics.mean(run["prompt_tokens"]),
            "output_tokens": statistics.mean(run["output_tokens"])}


def show(runs, n_params=494_032_768, weight_gb=None):
    """The record: one row per run, with FLOPs per problem at 2N per token."""
    print(f"{'system':24s} {'n':>5s} {'right':>6s} {'wrong':>6s} {'unread':>6s} {'acc':>6s} "
          f"{'prompt':>7s} {'output':>7s} {'GFLOPs':>7s}" + (f" {'weights':>8s}" if weight_gb else ""))
    for run in runs:
        s = summary(run)
        flops = 2 * n_params * (s["prompt_tokens"] + s["output_tokens"]) / 1e9
        extra = f" {weight_gb.get(run['name'], weight_gb.get('fp16')):7.3f}G" if weight_gb else ""
        print(f"{run['name']:24s} {s['n']:5d} {s['right']:6d} {s['wrong']:6d} {s['unreadable']:6d} "
              f"{s['accuracy']:6.3f} {s['prompt_tokens']:7.0f} {s['output_tokens']:7.0f} {flops:7.0f}" + extra)


def show_comparison(label, cmp):
    flag = "excludes 0" if cmp["lo"] > 0 or cmp["hi"] < 0 else "includes 0"
    print(f"{label:34s} {cmp['acc_a']:.3f} -> {cmp['acc_b']:.3f}  difference {cmp['diff']:+.3f} "
          f"[{cmp['lo']:+.3f}, {cmp['hi']:+.3f}] ({flag})  A only {cmp['a_only']}, B only {cmp['b_only']}, "
          f"sign test p = {cmp['p']:.2g}")


# ------------------------------------------------------------------------------------------------ model

def device_and_dtype():
    return ("cuda", torch.float16) if torch.cuda.is_available() else ("cpu", torch.float32)


def load_model():
    """The model and tokenizer, with the shipped generation defaults replaced by plain greedy decoding.

    Qwen2.5-Instruct ships sampling on (temperature 0.7, top-p 0.8, top-k 20) and a repetition penalty of 1.1.
    A "greedy" run that kept those defaults would not be greedy, so every run here sets its decoding explicitly.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    device, dtype = device_and_dtype()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=dtype)
    model = model.to(device=device, dtype=dtype).eval()
    shipped = model.generation_config.to_dict()
    eos = shipped.get("eos_token_id") or tok.eos_token_id
    model.generation_config = GenerationConfig(
        do_sample=False, temperature=1.0, top_p=1.0, top_k=50, repetition_penalty=1.0,
        bos_token_id=shipped.get("bos_token_id"), eos_token_id=eos if isinstance(eos, list) else [eos],
        pad_token_id=tok.pad_token_id)
    return model, tok


def free():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def worked_answer(problem):
    """A worked example's or training target's text: the model's own solution if it has one, else GSM8K's."""
    return problem.get("assistant") or f"{problem['solution']}\nThe answer is {problem['answer']}."


def messages(problem, spec, exemplars=()):
    """A chat: the system prompt, each worked example as a user turn and an assistant turn, then the problem."""
    chat = [{"role": "system", "content": spec["system"]}] if spec.get("system") else []
    for ex in exemplars:
        chat += [{"role": "user", "content": ex["question"]}, {"role": "assistant", "content": worked_answer(ex)}]
    return chat + [{"role": "user", "content": problem["question"]}]


def render(tok, chat):
    return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def _generate(model, tok, prompts, max_new_tokens):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0)
    eos = set(model.generation_config.eos_token_id)
    rows = []
    for ids, n_prompt in zip(out[:, enc["input_ids"].shape[1]:].tolist(), enc["attention_mask"].sum(1).tolist()):
        stop = next((i for i, t in enumerate(ids) if t in eos), None)
        kept = ids if stop is None else ids[:stop]
        rows.append({"prompt_tokens": n_prompt, "output_tokens": len(kept) + (stop is not None),
                     "finished": stop is not None, "raw": tok.decode(kept, skip_special_tokens=True)})
    return rows


def run_system(name, model, tok, data, ids, spec, extract_answer, batch_size=64):
    """Run a system greedily on the given problems and score it with your extract_answer."""
    problems = [{"id": i, **data["problems"][i]} for i in ids]
    shots = data["exemplars"][:spec["shots"]]
    batch_size = batch_size // 2 if shots else batch_size
    prompts = {p["id"]: render(tok, messages(p, spec, shots)) for p in problems}
    order = sorted(problems, key=lambda p: (-len(prompts[p["id"]]), p["id"]))
    done, start = {}, time.perf_counter()
    for b in range(0, len(order), batch_size):
        batch = order[b:b + batch_size]
        for p, row in zip(batch, _generate(model, tok, [prompts[p["id"]] for p in batch], spec["max_new_tokens"])):
            done[p["id"]] = row
        print(f"\r  {name}: {min(b + batch_size, len(order))}/{len(order)} problems", end="")
    print(f"  ({time.perf_counter() - start:.0f} s)")
    rows = [done[i] for i in ids]
    run = {key: [r[key] for r in rows] for key in ("raw", "prompt_tokens", "output_tokens", "finished")}
    return {"name": name, "ids": list(ids), **run,
            **score(run["raw"], [data["problems"][i]["answer"] for i in ids], extract_answer)}


# ------------------------------------------------------------------------------------------- log-likelihood

def solution_examples(tok, problems):
    """Token ids and an include mask for each problem's solution after S1's prompt.

    The mask covers the assistant's reply and its end-of-turn token, as a fine-tuning loss mask does.
    """
    end = tok.convert_tokens_to_ids("<|im_end|>")
    out = []
    for p in problems:
        chat = messages(p, SYSTEMS["S1"])
        prompt = render(tok, chat)
        full = tok.apply_chat_template(chat + [{"role": "assistant", "content": worked_answer(p)}], tokenize=False)
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = tok(full, add_special_tokens=False)["input_ids"]
        assert ids[:len(prompt_ids)] == prompt_ids, "the prompt does not tokenize as a prefix"
        last = max(i for i, t in enumerate(ids) if t == end)
        include = [0] * len(prompt_ids) + [1] * (last + 1 - len(prompt_ids)) + [0] * (len(ids) - last - 1)
        out.append({"id": p["id"], "input_ids": ids, "include": include})
    return out


def pad_right(examples, pad_id, device):
    width = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    inc = torch.zeros((len(examples), width), dtype=torch.long)
    for r, e in enumerate(examples):
        n = len(e["input_ids"])
        ids[r, :n] = torch.tensor(e["input_ids"])
        mask[r, :n] = 1
        inc[r, :n] = torch.tensor(e["include"])
    return ids.to(device), mask.to(device), inc.to(device)


def token_logprobs(model, ids, mask):
    """log p(token | the tokens before it), for every token after the first: shape (batch, length - 1)."""
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)


def solution_loglik(model, tok, examples, masked_logprob, batch_size=8):
    """Per-problem (sum, count) of the solutions' log-probabilities, through your masked_logprob."""
    sums, counts = [], []
    with torch.no_grad():
        for b in range(0, len(examples), batch_size):
            ids, mask, inc = pad_right(examples[b:b + batch_size], tok.pad_token_id, model.device)
            s, c = masked_logprob(token_logprobs(model, ids, mask), inc[:, 1:])
            sums += [float(x) for x in s]
            counts += [int(round(float(x))) for x in c]
    return sums, counts


def per_token(sums, counts):
    return sum(sums) / sum(counts)


# ----------------------------------------------------------------------------------------------- LoRA

def train_lora(model, tok, examples, masked_logprob, cfg=LORA_CONFIG, seed=0):
    """Fine-tune LoRA adapters, with the loss on the solution tokens only, through your masked_logprob.

    Adapter weights are kept in fp32; on a GPU the forward pass runs under fp16 autocast with a gradient scaler.
    """
    from peft import LoraConfig, get_peft_model
    random.seed(seed)
    torch.manual_seed(seed)
    model = get_peft_model(model, LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()
    order = list(range(len(examples)))
    steps = math.ceil(len(order) / cfg["batch"]) * cfg["epochs"]
    warm = max(1, round(0.05 * steps))
    opt = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    cuda = next(model.parameters()).is_cuda
    scaler = torch.amp.GradScaler("cuda") if cuda else None
    model.train()
    losses, start = [], time.perf_counter()
    for epoch in range(cfg["epochs"]):
        random.Random(seed * 1000 + epoch).shuffle(order)
        for s in range(0, len(order), cfg["batch"]):
            idx = order[s:s + cfg["batch"]]
            count = sum(sum(examples[i]["include"][1:]) for i in idx)
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for m in range(0, len(idx), cfg["micro_batch"]):
                ids, mask, inc = pad_right([examples[i] for i in idx[m:m + cfg["micro_batch"]]],
                                           tok.pad_token_id, model.device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                    total, _ = masked_logprob(token_logprobs(model, ids, mask), inc[:, 1:])
                    loss = -total.sum() / count
                (scaler.scale(loss) if scaler else loss).backward()
                step_loss += loss.item()
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            losses.append(step_loss)
            print(f"\r  step {len(losses)}/{steps}: loss {step_loss:.3f}", end="")
    print(f"  ({time.perf_counter() - start:.0f} s)")
    model.eval()
    return model, losses


def load_adapter(model, folder):
    from peft import PeftModel
    return PeftModel.from_pretrained(model, str(folder))


# ------------------------------------------------------------------------------------------- quantization

def quantizable(model):
    """Every Linear layer whose weight is not the tied embedding."""
    tied = model.get_input_embeddings().weight
    return [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m.weight is not tied]


def quantize_model(model, quantize_matrix, group_size):
    """Every quantizable weight rounded with your quantize_matrix, in place."""
    with torch.no_grad():
        for _, m in quantizable(model):
            m.weight.copy_(quantize_matrix(m.weight, group_size))
    return model


def _part(name):
    if "embed_tokens" in name or "lm_head" in name:
        return "embedding"
    if "self_attn" in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    return "other"


def weight_bytes(model, group_size, bits=4):
    """Bytes of weights by part: in fp16, and with every quantizable Linear in int4 plus a 16-bit scale per group."""
    qset = {id(m.weight): m.in_features for _, m in quantizable(model)}
    parts = {}
    for name, p in model.named_parameters():  # a tied weight is listed once
        before = 2 * p.numel()
        if id(p) in qset:
            g = qset[id(p)] if group_size == "row" else int(group_size)
            after = p.numel() * bits / 8 + 2 * p.numel() / g
        else:
            after = before
        row = parts.setdefault(_part(name), {"fp16": 0, "int4": 0})
        row["fp16"] += before
        row["int4"] += after
    total_before = sum(r["fp16"] for r in parts.values())
    total_after = sum(r["int4"] for r in parts.values())
    return {"parts": parts, "fp16": total_before, "int4": total_after, "ratio": total_after / total_before}


def show_weight_bytes(wb, prediction):
    print(f"{'part':10s} {'fp16 GB':>8s} {'int4 GB':>8s}")
    for part, row in wb["parts"].items():
        print(f"{part:10s} {row['fp16'] / 1e9:8.3f} {row['int4'] / 1e9:8.3f}")
    print(f"{'total':10s} {wb['fp16'] / 1e9:8.3f} {wb['int4'] / 1e9:8.3f}")
    print(f"\nratio {wb['ratio']:.6f}; you predicted {prediction}")


def group_view(W, group_size):
    rows, cols = W.shape
    g = cols if group_size == "row" else int(group_size)
    if cols % g:
        raise ValueError(f"group size {g} does not divide a row of {cols}")
    return W.float().reshape(rows, cols // g, g)


def quantize_with_scales(W, scale, bits=4):
    """W rounded to int4 with the given scale for each group (shape rows x groups x 1), then back to floats."""
    qmax = 2 ** (bits - 1) - 1
    Wg = W.float().reshape(W.shape[0], scale.shape[1], -1)
    return (torch.clamp(torch.round(Wg / scale), -qmax - 1, qmax) * scale).reshape(W.shape).to(W.dtype)


def output_error(W, Wq, sigma, group_size):
    """e^T Sigma e for each row and group, with e = Wq - W: the mean squared change that group's error makes to the
    layer's output, when sigma holds each group's block of E[x x^T] over the layer's inputs."""
    e = group_view(Wq.float() - W.float(), group_size)
    return torch.einsum("rgi,gij,rgj->rg", e, sigma.float(), e)


def input_moments(model, tok, examples, group_size, batch_size=8):
    """Sigma for every quantizable Linear: per group of input columns, the block of E[x x^T] over the calibration
    tokens, prompt and solution alike, padding left out. Collected by forward hooks on the unquantized model."""
    sums, state = {}, {"mask": None, "tokens": 0}

    def hook(name):
        def collect(module, inputs, output):
            x = group_view(inputs[0][state["mask"]], group_size)
            block = torch.einsum("ngi,ngj->gij", x, x)
            sums[name] = sums[name] + block if name in sums else block
        return collect

    handles = [m.register_forward_hook(hook(n)) for n, m in quantizable(model)]
    try:
        with torch.no_grad():
            for b in range(0, len(examples), batch_size):
                ids, mask, _ = pad_right(examples[b:b + batch_size], tok.pad_token_id, model.device)
                state["mask"] = mask.bool()
                state["tokens"] += int(mask.sum())
                model.get_decoder()(input_ids=ids, attention_mask=mask)
    finally:
        for h in handles:
            h.remove()
    return {n: s / state["tokens"] for n, s in sums.items()}


def quantize_model_calibrated(model, sigmas, group_size, best_scales):
    """Every quantizable weight rounded with the scales your best_scales chooses, in place. Returns the summed
    output error with task 4's scales and with the chosen ones, and the share of groups that clipped."""
    plain_err = chosen_err = 0.0
    clipped = groups = 0
    with torch.no_grad():
        for name, m in quantizable(model):
            sigma = sigmas[name].to(m.weight.device)
            plain = best_scales(m.weight, sigma, group_size, factors=(1.0,))
            scale = best_scales(m.weight, sigma, group_size)
            plain_err += float(output_error(m.weight, quantize_with_scales(m.weight, plain), sigma, group_size).sum())
            Wq = quantize_with_scales(m.weight, scale)
            chosen_err += float(output_error(m.weight, Wq, sigma, group_size).sum())
            clipped += int((scale < plain * 0.999).sum())
            groups += scale.numel()
            m.weight.copy_(Wq)
    return {"error_ratio": chosen_err / plain_err, "clipped_share": clipped / groups}


# ---------------------------------------------------------------------------------------------- cost

def cost_facts(model):
    """E7's figures for this model: FLOPs per token, weight and KV-cache bytes, and where the KV cache wins."""
    cfg = model.config
    n = sum(p.numel() for p in model.parameters())  # a tied weight is counted once
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2
    return {"params": n, "flops_per_token": 2 * n, "lora_train_flops_per_token": 4 * n,
            "weight_bytes_fp16": 2 * n, "kv_bytes_per_token": kv, "kv_crossover_tokens": 2 * n / kv}


# ---------------------------------------------------------------------------------------------- audit

AUDIT_LABELS = ("arithmetic slip", "wrong setup", "incomplete generation", "extraction failure")
AUDIT_FEEDBACK = {
    284: {
        "label": "arithmetic slip",
        "first_error": "20 + 40 + 60 + 240 = 340",
        "diagnosis": "The weekly story counts are set up correctly, but the final sum is 360, not 340.",
        "grader": "The extractor reads 340, so this is a model error that the grader records correctly.",
    },
    1271: {
        "label": "wrong setup",
        "first_error": "The total number of candies ... = 54 + 108 + 27 = 189",
        "diagnosis": "189 is the subtotal of known colors, not the bag total. The next step must subtract 189 from the given total, 232.",
        "grader": "The extractor reads 0 from the model's final answer, so this is a model error that the grader records correctly.",
    },
    1134: {
        "label": "extraction failure",
        "first_error": "There is no reasoning error: the model reaches 28 / 4 = 7 and boxes 7.",
        "diagnosis": "The strict extractor does not read boxed notation. With no 'answer is' marker, it falls back to the last number, $60 from the closing sentence.",
        "grader": "The model answer is correct, but the grader reads 60 and records it as wrong.",
    },
    499: {
        "label": "wrong setup",
        "first_error": "The cost for the first venue is: 200",
        "diagnosis": "The first venue also needs $5 of food per guest, so its cost is 200 + 5n. The comparison should be 200 + 5n = 25n.",
        "grader": "The extractor reads 8, so this is a model error that the grader records correctly.",
    },
    1018: {
        "label": "wrong setup",
        "first_error": "Time spent reading 18 pages = 18 pages x 0.3 pages per minute",
        "diagnosis": "The rate is correctly found as 0.3 pages per minute, but time is pages divided by pages per minute, not their product.",
        "grader": "The extractor reads 5.4, so this is a model error that the grader records correctly.",
    },
    254: {
        "label": "wrong setup",
        "first_error": "30 x 0.30 = 9. So, Anne went down the slide 9 times.",
        "diagnosis": "Nine is the reduction. Thirty percent less than 30 is 30 - 9 = 21 before multiplying by four for Robert.",
        "grader": "The extractor reads 36, so this is a model error that the grader records correctly.",
    },
    1084: {
        "label": "wrong setup",
        "first_error": "Now, we need to find out how much fence is left over for Sam.",
        "diagnosis": "The model has already solved S = 20, which is Sam's share. It invents a second 'left over' quantity and subtracts both shares again.",
        "grader": "The extractor reads 0, so this is a model error that the grader records correctly.",
    },
    233: {
        "label": "wrong setup",
        "first_error": "Another one-third of the remaining penguins ... = one-third x 12 = 4",
        "diagnosis": "'Another one-third' refers to another third of the original 36 penguins. Twelve swim and twelve eat, leaving twelve.",
        "grader": "The extractor reads 32, so this is a model error that the grader records correctly.",
    },
    923: {
        "label": "incomplete generation",
        "first_error": "The answer repeats 15 - 15 = 0 and 15 - 0 = 15 until it reaches the output limit.",
        "diagnosis": "The response never completes a stable calculation or a final answer. Repetition and truncation are different from a finished answer with one bad step.",
        "grader": "The fallback extractor still finds the trailing 15. That makes the answer scoreable, but it does not make the generation complete.",
    },
    559: {
        "label": "wrong setup",
        "first_error": "Daily recommended cups of coffee: 12.",
        "diagnosis": "The problem gives the daily recommendation as 4 cups. Octavia drinks 2, Juan drinks 20, and he must reduce by 16.",
        "grader": "The extractor reads 54, so this is a model error that the grader records correctly.",
    },
}


def _audit_case(data, item):
    if not isinstance(item, int) or not 1 <= item <= len(data["audit"]):
        raise ValueError(f"audit item must be from 1 to {len(data['audit'])}")
    i = data["audit"][item - 1]
    return i, data["full"].index(i), data["problems"][i]

def show_audit(data, extract_answer, item):
    """One of the audit's ten problems: the question, S0's answer, then S1's full answer and what the grader read."""
    i, k, p = _audit_case(data, item)
    print(f"Audit item {item} of {len(data['audit'])} (problem {i}). The right answer is {p['answer']}.\n")
    print(p["question"], "\n")
    print("S0 answered:", data["runs"]["S0"]["raw"][k].strip()[:200], "\n")
    raw = data["runs"]["S1"]["raw"][k]
    print("S1 answered:\n" + raw)
    print(f"\nThe grader read {extract_answer(raw)!r} from S1's answer.")


def show_audit_feedback(data, item):
    """Reveal the editorial diagnosis for an audit case after the learner records one."""
    i, _, _ = _audit_case(data, item)
    feedback = AUDIT_FEEDBACK[i]
    print(f"Audit item {item} review ({i})")
    print("Label:", feedback["label"])
    print("First error:", feedback["first_error"])
    print("Diagnosis:", feedback["diagnosis"])
    print("Grader check:", feedback["grader"])


def subset(run, ids):
    """A run restricted to the given problems, in their order, so two runs over different problem sets can be
    compared on the problems they share."""
    pos = {i: k for k, i in enumerate(run["ids"])}
    keep = [pos[i] for i in ids]
    out = {"name": run["name"], "ids": list(ids)}
    for key, value in run.items():
        if isinstance(value, list) and len(value) == len(run["ids"]) and key != "ids":
            out[key] = [value[k] for k in keep]
    return out



def sign_test(a_only, b_only):
    """Exact two-sided sign test on the problems where two systems disagree (E6)."""
    m, k = a_only + b_only, min(a_only, b_only)
    if m == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)
