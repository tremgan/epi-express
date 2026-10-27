"""Experiment logging: one record per training run, written as it happens.

- runs/runs.jsonl  machine-readable: a start event, one event per epoch, an end event
- EXPERIMENTS.md   human-readable: the same run as a markdown section, including what
                   changed compared to the previous run

Every write is flushed to disk immediately, so an interrupted run (Ctrl-C, crash, even a
killed process) keeps everything up to its last finished epoch.
"""
import inspect
import json
import os
import subprocess
import time
from datetime import datetime

JSONL = os.path.join("runs", "runs.jsonl")
MARKDOWN = "EXPERIMENTS.md"
OPTIMIZER_KEYS = ("lr", "momentum", "weight_decay", "betas", "nesterov")


def describe_run(model, model_config, optimizer, loss_function, train_loader, n_epochs,
                 data=None, **extra):
    """Everything that defines a run. Put anything else you change (scheduler, seed, ...)
    into **extra so it shows up in the log and in the change report."""
    sig = inspect.signature(type(model).__init__)
    params = {
        k: p.default
        for k, p in sig.parameters.items()
        if k != "self" and p.default is not p.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    params.update(model_config)  # resolved constructor args, so changed defaults are caught too
    return {
        "model": {
            "name": type(model).__name__,
            "params": params,
            "n_params": sum(p.numel() for p in model.parameters()),
        },
        "optimizer": {
            "name": type(optimizer).__name__,
            **{k: v for k, v in optimizer.defaults.items() if k in OPTIMIZER_KEYS},
        },
        "loss": type(loss_function).__name__,
        "batch_size": train_loader.batch_size,
        "n_train": len(train_loader.dataset),
        "n_epochs": n_epochs,
        "data": data or {},
        "extra": extra,
    }


def _append(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _event(**kw):
    _append(JSONL, json.dumps(kw, default=str) + "\n")


def _read_events():
    if not os.path.exists(JSONL):
        return []
    events = []
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # half-written line from a hard kill
    return events


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):  # an empty dict contributes no keys
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _git_state():
    try:
        run = lambda *a: subprocess.check_output(  # noqa: E731
            ["git", *a], stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        return {"commit": run("rev-parse", "--short", "HEAD"),
                "uncommitted_code": bool(run("status", "--porcelain", "--", "*.py"))}
    except Exception:
        return {"commit": None, "uncommitted_code": None}


def _summary_lines(c):
    m, o = c["model"], c["optimizer"]
    opt = ", ".join(f"{k}={v}" for k, v in o.items() if k != "name")
    lines = [
        f"- Model: `{m['name']}({', '.join(f'{k}={v}' for k, v in m['params'].items())})`"
        f" — {m['n_params']:,} parameters",
        f"- Optimizer: `{o['name']}({opt})` | loss `{c['loss']}` | batch size {c['batch_size']}"
        f" | {c['n_epochs']} epochs | {c['n_train']} training genes",
    ]
    if c["data"]:
        lines.append("- Data: " + ", ".join(f"{k}={v}" for k, v in c["data"].items()))
    if c["extra"]:
        lines.append("- Extra: " + ", ".join(f"{k}={v}" for k, v in c["extra"].items()))
    return lines


def _best(epochs):
    ok = [e for e in epochs if e.get("val_spearman") is not None and e["val_spearman"] == e["val_spearman"]]
    return max(ok, key=lambda e: e["val_spearman"]) if ok else None


def _result_line(status, epochs, duration_s):
    line = f"- **Result: {status}**"
    if epochs:
        last, best = epochs[-1], _best(epochs)
        line += f" — {len(epochs)} epoch(s) finished, {duration_s / 60:.1f} min"
        if best:
            line += (f"; best val Spearman {best['val_spearman']:.4f} (epoch {best['epoch']}),"
                     f" last {last['val_spearman']:.4f}, last val loss {last['val_loss']:.4f}")
        else:
            line += f"; last train loss {last['train_loss']:.4f}"
    else:
        line += " — no epoch finished"
    return line + "\n"


class RunLogger:
    """with RunLogger(config, note="what I changed and why") as log: ... log.log_epoch(...)"""

    def __init__(self, config, note=""):
        self.config = json.loads(json.dumps(config, default=str))  # same types as when re-read
        self.note = note
        self.epochs = []

    def __enter__(self):
        events = _read_events()
        self._close_orphans(events)
        taken = {e["run_id"] for e in events if e["event"] == "start"}
        base = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id, n = base, 1
        while self.run_id in taken:  # two runs started within the same second
            n += 1
            self.run_id = f"{base}-{n}"
        starts = [e for e in events if e["event"] == "start"]

        new = _flatten(self.config)
        if not starts:
            changes = ["first logged run (baseline)"]
        else:
            old = _flatten(starts[-1]["config"])
            changes = [f"`{k}`: {old.get(k, '—')} → {new.get(k, '—')}"
                       for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
            changes = changes or ["none (identical config — differences are run-to-run noise)"]
        code = _git_state()
        self.t0 = time.time()
        _event(event="start", run_id=self.run_id, time=datetime.now().isoformat(timespec="seconds"),
               note=self.note, config=self.config, code=code, changes=changes)

        head = f"\n## Run {self.run_id}" + (f" — {self.note}" if self.note else "") + "\n"
        head += "- Changes vs previous run:\n" + "".join(f"  - {c}\n" for c in changes)
        if code["commit"]:
            head += f"- Code: commit `{code['commit']}`" + (" + uncommitted changes" if code["uncommitted_code"] else "") + "\n"
        head += "\n".join(_summary_lines(self.config)) + "\n- Epochs:\n"
        _append(MARKDOWN, head)
        return self

    def log_epoch(self, epoch, train_loss, val_loss=None, val_spearman=None):
        e = {"epoch": epoch, "train_loss": float(train_loss),
             "val_loss": None if val_loss is None else float(val_loss),
             "val_spearman": None if val_spearman is None else float(val_spearman)}
        self.epochs.append(e)
        _event(event="epoch", run_id=self.run_id, **e)
        line = f"  - epoch {epoch}: train loss {e['train_loss']:.4f}"
        if e["val_loss"] is not None:
            line += f" | val loss {e['val_loss']:.4f} | val Spearman {e['val_spearman']:.4f}"
        _append(MARKDOWN, line + "\n")

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            status = "completed"
        elif issubclass(exc_type, KeyboardInterrupt):
            status = "interrupted"
        else:
            status = f"failed ({exc_type.__name__}: {exc})"
        duration = time.time() - self.t0
        best = _best(self.epochs)
        _event(event="end", run_id=self.run_id, status=status, duration_s=round(duration, 1),
               best_val_spearman=best and best["val_spearman"], best_epoch=best and best["epoch"])
        _append(MARKDOWN, _result_line(status, self.epochs, duration))
        return False  # never swallow the exception

    @staticmethod
    def _close_orphans(events):
        """A run that has no end event was killed hard (terminal closed, power loss, kill -9)."""
        ended = {e["run_id"] for e in events if e["event"] == "end"}
        for s in (e for e in events if e["event"] == "start" and e["run_id"] not in ended):
            eps = [e for e in events if e["event"] == "epoch" and e["run_id"] == s["run_id"]]
            _event(event="end", run_id=s["run_id"], status="killed", duration_s=None)
            _append(MARKDOWN, f"- **Result: killed** — no final status was recorded "
                              f"(run {s['run_id']}; {len(eps)} epoch(s) finished)\n")
