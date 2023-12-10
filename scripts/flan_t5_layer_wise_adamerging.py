from _common import *

log = logging.getLogger(__name__)

import types
from collections import defaultdict
from typing import Any

import lightning as L
import lightning.pytorch as pl
from flan_t5_checkpoint_path import finetuned_model_path
from flan_t5_individuals import Program as _Program
from flan_t5_individuals import metric_func
from torch.func import functional_call
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator
from transformers.generation import GenerationConfig, GenerationMixin

from datasets import DatasetDict, load_dataset, load_from_disk
from src.adamerging import softmax_entropy
from src.concrete_mask import ConcreteMask
from src.layer_wise_fusion import *
from src.tasks.arithmetic import state_dict_avg, state_dict_sub, state_dict_sum
from src.ties_merging_utils import *
from src.utils import num_devices, num_parameters, timeit_context

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
        self.results_path = self.results_dir / "layer_wise_adamerging.csv"
        self.ckpt_dir = self.results_dir / "layer_wise_adamerging"
        self.ckpt_path = self.ckpt_dir / "ckpt.pt"

        self.fabric = L.Fabric(accelerator="cuda", devices=1)
        self.fabric.launch()

    def run(self):
        self.load_models(task_vector_device=torch.device("cuda:0"))
        self.load_datasets()
        self.initialize_merged_model()

        self.adamerging()

    def eval_model_on_datasets(self, model):
        #! `self.eval_model_on_datasets(self.model)` will call `self.model.generate`, which will call encoder and decoder forward directly, so we need to load the merged state dict first
        self.model.model.load_state_dict(self.model.merged_state_dict, strict=False, assign=True)
        return super().eval_model_on_datasets(model)

    def initialize_merged_model(self):
        pretrained_model = self.pretrained_model
        task_vectors = self.task_vectors

        # Initialize the task-wise weights
        self.init_layer_wise_weights = get_layer_wise_weights(
            num_models=len(task_vectors),
            num_layers=len(task_vectors[0]),
            init_values=0.3,
        )
        model = LayerWiseMergedModel(
            pretrained_model=deepcopy(pretrained_model),
            layer_wise_weight=deepcopy(self.init_layer_wise_weights),
            task_vectors=task_vectors,
        )
        self.model: LayerWiseMergedModel = self.fabric.setup_module(model)

        log.info(f"total number of parameters in the model: {num_parameters(model)}")

    def compute_loss(self):
        losses = 0
        for dataset_name in self.cfg.test_datasets:
            batch = next(self.shuffled_test_loader_iters[dataset_name])
            # truncate the input
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            max_len = input_ids.size(1)
            while torch.all(attention_mask[:, max_len - 1] == 0):
                max_len -= 1
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]

            # T5 uses the `pad_token_id` as the `decoder_start_token_id`
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=torch.ones(input_ids.size(0), 1, dtype=torch.long, device=input_ids.device) * self.tokenizer.pad_token_id,
            )
            logits = outputs.logits[:, 0, :]
            loss = softmax_entropy(logits).mean(0)
            losses += loss
        return losses

    def adamerging(self):
        log.info("start test time adaptation")
        cfg = self.cfg

        results = defaultdict(list)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0)

        self.model.eval()
        self.model.merge_weights()
        _results = self.eval_model_on_datasets(self.model)
        for dataset_name, score in _results.items():
            results["epoch"].append(0)
            results["dataset"].append(dataset_name)
            results["score"].append(score)

        for epoch_idx in tqdm(range(epochs := 1000)):
            self.model.train()
            losses = self.compute_loss()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            self.model.merge_weights()
            print(self.model.layer_wise_weight)

            if ((epoch_idx + 1) % 100) == 0:
                os.makedirs(self.ckpt_dir, exist_ok=True)
                torch.save(
                    {
                        "layer_wise_weight": self.model.layer_wise_weight,
                    },
                    self.ckpt_path,
                )
                _results = self.eval_model_on_datasets(self.model)
                for dataset_name, score in _results.items():
                    results["epoch"].append(epoch_idx + 1)
                    results["dataset"].append(dataset_name)
                    results["score"].append(score)
                pd.DataFrame(results).to_csv(self.results_path, index=False)


@hydra.main(str(CONFIG_DIR), "flan_t5_default", None)
def main(cfg: DictConfig):
    (program := Program(cfg)).run()


if __name__ == "__main__":
    main()
