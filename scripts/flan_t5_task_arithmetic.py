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
        self.results_path = self.results_dir / "task_arithmetic.csv"

        self.fabric = L.Fabric(accelerator="cpu", devices=1)
        self.fabric.launch()

    def run(self):
        self.load_models()
        self.load_datasets()

        self.eval_task_arithmetic()

    def eval_task_arithmetic(self):
        results = defaultdict(list)
        task_vector = state_dict_sum(self.task_vectors)
        for scaling_coef_ in track(np.linspace(0, 1, 11), description="scaling_coef"):
            model = deepcopy(self.pretrained_model)
            for n, p in task_vector.items():
                model.state_dict()[n] += scaling_coef_ * task_vector[n]
            model = self.fabric.setup_module(model)
            _results = self.eval_model_on_datasets(model)
            results["scaling_coef"].append(scaling_coef_)
            for dataset_name, scrore in _results.items():
                results[dataset_name].append(scrore)
            print(df := pd.DataFrame(results))
            df.to_csv(self.results_path, index=False)


@hydra.main(str(CONFIG_DIR), "flan_t5_default", None)
def main(cfg: DictConfig):
    (program := Program(cfg)).run()


if __name__ == "__main__":
    main()
