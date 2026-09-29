"""Ask Qwen2-Audio to answer and explain, on a few validation clips per task (4 its label ranking got right, 4 wrong).

Free-form, greedy. The explanation is the model's own account, not the mechanism behind the log-likelihood ranking,
so it is shown next to the ranked top-3 and the true label. Output: results/alm_explanations.md.
"""

from pathlib import Path

import numpy as np

from alm_qwen import ANSWERS, PROMPTS, generate, load_model, to16
from hw1.data import LABELS, load_audio, load_manifest

HW1 = Path(__file__).resolve().parents[1]
ASK = " Then explain in two or three sentences which sounds in the recording led you to that answer."


def main() -> None:
    proc, model = load_model()
    lines = ["# Qwen2-Audio: answers with explanations (validation clips, plain prompt)", "",
             "Greedy free-form answers. `ranked top-3` is the log-likelihood ranking used for scoring.", ""]
    for key in ("A", "B"):
        z = np.load(HW1 / "features" / f"{key}_alm_plain_validation.npz")
        y = np.array([LABELS[key].index(l) for l in z["label"]])
        order = np.argsort(-z["logp"], 1)
        right, wrong = np.where(order[:, 0] == y)[0][:4], np.where(order[:, 0] != y)[0][:4]
        paths = load_manifest(key).set_index("sample_id")["path"]
        lines += [f"## Dataset {key}", ""]
        for i in np.concatenate([right, wrong]):
            sid = str(z["sample_id"][i])
            text = generate(proc, model, to16(load_audio(paths[sid])), PROMPTS[key]["plain"] + ASK, max_new_tokens=120)
            top3 = ", ".join(ANSWERS[key][LABELS[key][c]] for c in order[i, :3])
            lines += [f"- **{sid}** true: {ANSWERS[key][LABELS[key][y[i]]]} | ranked top-3: {top3}",
                      f"  > {text.strip()}", ""]
            print(sid, text.strip()[:120], flush=True)
    (HW1 / "results" / "alm_explanations.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
