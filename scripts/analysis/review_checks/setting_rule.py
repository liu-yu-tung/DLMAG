import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import boot_ci, fmt, per_prompt

FD_OUT = (Path(__file__).parent / "fd_robust.out").read_text()
PAIRS = [("rescale18", "rescale12"), ("rescale12_phi1", "rescale12"), ("rescale12_best3_c8998", "rescale12"),
         ("rescale12_best3_c8998", "rescale12_phi1"), ("rescale12_best3_c8998", "rescale18")]


def fd_vgg_line(a: str, b: str) -> str:
    match = re.search(rf"^{re.escape(a)} - {re.escape(b)}: (.*)$", FD_OUT, flags=re.M)
    return match.group(1) if match else "not in fd_robust.out"


print("a - b: dCLAP, dKL-PaSST (seed-0 bootstrap, 1000 resamples), and paired FD-VGG from fd_robust.out (CPU bootstrap over the same 100 prompts)")
for a, b in PAIRS:
    x, y = per_prompt(f"dev__{a}"), per_prompt(f"dev__{b}")
    dclap = (x["clap_cosine"] - y["clap_cosine"]).to_numpy()
    dkl = (x["kl_passt_softmax"] - y["kl_passt_softmax"]).to_numpy()
    print(f"{a} - {b}")
    print(f"  dCLAP {fmt(boot_ci(dclap))}")
    print(f"  dKL-PaSST {fmt(boot_ci(dkl))}")
    print(f"  FD-VGG: {fd_vgg_line(a, b)}")
