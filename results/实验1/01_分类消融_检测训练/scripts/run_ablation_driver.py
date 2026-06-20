import yaml
from omegaconf import OmegaConf

from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.eval.ablation.runner import run_ablation, write_results

base = OmegaConf.load("configs/train/classify/config.yaml")
base.data.data_root = "/root/autodl-tmp/data/raw/birds525"
base.data.num_workers = 8
base.data.batch_size = 128
base.model.pretrained = True
base.trainer.accelerator = "gpu"
base.trainer.devices = 1
base.trainer.max_epochs = 80
base.export.enabled = False
base.output_dir = "/root/autodl-tmp/outputs/ablation"
OmegaConf.update(base, "track.aim", True, force_add=True)  # aim 追踪每个实验

grid = yaml.safe_load(open("configs/ablation/classify_grid.yaml"))["grid"]
m = DatasetManifest.load("data/processed/birds525/manifest.json")
print("GRID:", grid)
rows = run_ablation(base, grid, m)
csv_p, md_p = write_results(rows, "/root/autodl-tmp/outputs/ablation")
print("=== ABLATION DONE ===", csv_p)
print(open(md_p).read())
