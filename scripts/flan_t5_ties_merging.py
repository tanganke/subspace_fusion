from _common import *

from scripts._common import DictConfig

log = logging.getLogger(__name__)

from collections import defaultdict

import lightning as L
import lightning.pytorch as pl
from flan_t5_checkpoint_path import finetuned_model_path
from flan_t5_individuals import Program as _Program
from flan_t5_individuals import metric_func
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator

from datasets import DatasetDict, load_dataset, load_from_disk
from src.tasks.arithmetic import state_dict_avg, state_dict_sub, state_dict_sum
from src.ties_merging_utils import *
from src.utils import num_devices

# disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Program(_Program):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        if hasattr(cfg, "seed") and cfg.seed is not None:
            log.info(f"set seed to {cfg.seed}")
            L.seed_everything(cfg.seed)

        if cfg.peft.peft_config is None:
            self.results_dir = RESULTS_DIR / cfg.model.name
        else:
            self.results_dir = RESULTS_DIR / (cfg.model.name + "_" + cfg.peft.name)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.results_dir / "ties_merging.csv"

        self.fabric = L.Fabric(accelerator="cuda", devices=1)
        self.fabric.launch()

    def run(self):
        self.load_models()
        self.load_datasets()

        self.ties_merging()

    def ties_merging(self):
        results = defaultdict(list)
        reference_tv = deepcopy(self.task_vectors[0])
        tv_flat_checks = torch.vstack([state_dict_to_vector(check, remove_keys=[]) for check in track(self.task_vectors, "flattening task vectors")])

        K = 20
        for scaling_coef in track(np.linspace(0, 1, 11), "scaling_coef"):
            merged_tv = ties_merging(tv_flat_checks, reset_thresh=K, merge_func="dis-sum")
            merged_tv = vector_to_state_dict(merged_tv, reference_tv, remove_keys=[])
            model = deepcopy(self.pretrained_model)
            for n, p in merged_tv.items():
                model.state_dict()[n] += scaling_coef * merged_tv[n]
            model = self.fabric.setup_module(model)
            _results = self.eval_model_on_datasets(model)
            results["scaling_coef"].append(scaling_coef)
            for dataset_name, scrore in _results.items():
                results[dataset_name].append(scrore)
            print(df := pd.DataFrame(results))
            df.to_csv(self.results_path, index=False)


@hydra.main(str(CONFIG_DIR), "flan_t5_default", None)
def main(cfg: DictConfig):
    (program := Program(cfg)).run()


if __name__ == "__main__":
    main()
