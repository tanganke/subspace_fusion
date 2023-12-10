from typing import Any

from _common import *

from scripts._common import DictConfig

log = logging.getLogger(__name__)

from collections import defaultdict

import lightning as L
import lightning.fabric
import lightning.pytorch as pl
from flan_t5_checkpoint_path import finetuned_model_path
from flan_t5_individuals import Program as _Program
from flan_t5_individuals import metric_func
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator
from transformers.generation import GenerationConfig, GenerationMixin

from datasets import DatasetDict, load_dataset, load_from_disk
from src.adamerging import softmax_entropy
from src.concrete_mask import ConcreteMask
from src.task_wise_fusion import *
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
        self.results_path = self.results_dir / "concrete_task_arithmetic.csv"
        self.ckpt_dir = self.results_dir / "concrete_task_arithmetic"
        self.ckpt_path = self.ckpt_dir / "ckpt.pt"
        self.individual_results_path = self.results_dir / "concrete_task_arithmetic_individuals.csv"

        self.fabric = L.Fabric(accelerator="cuda", devices=1)
        self.fabric.launch()

    def run(self):
        self.load_models(task_vector_device=torch.device("cuda:1"))
        self.load_datasets()
        self.initialize_merged_model()

        self.meta_train()
        self.eval_individuals()

    def merge_model_weights(self):
        """this method is called every time `shared_mask` is updated"""
        self.model.task_vectors = self.shared_mask.apply_mask(self.task_vectors)
        self.model.merge_weights()

    def initialize_merged_model(self):
        pretrained_model = self.pretrained_model
        task_vectors = self.task_vectors

        # Initialize the task-wise weights
        self.shared_mask = ConcreteMask(
            temperature=0.5,
            state_dict=task_vectors[0],
            init_value=0,
            draw_sample=True,
        )
        self.init_task_wise_weights = get_task_wise_weights(
            num_models=len(task_vectors),
            init_values=0.3,
        )
        model = TaskWiseMergedModel(
            pretrained_model=deepcopy(pretrained_model),
            task_wise_weight=deepcopy(self.init_task_wise_weights),
            task_vectors=task_vectors,
        )
        self.model: TaskWiseMergedModel = self.fabric.setup_module(model)

        log.info(f"total number of parameters in the model: {num_parameters(model)}")

    def eval_model_on_datasets(self, model):
        #! `self.eval_model_on_datasets(self.model)` will call `self.model.generate`, which will call encoder and decoder forward directly, so we need to load the merged state dict first
        self.model.model.load_state_dict(self.model.merged_state_dict, strict=False, assign=True)
        return super().eval_model_on_datasets(model)

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

    def meta_train(self):
        log.info("meta training")
        cfg = self.cfg
        results = defaultdict(list)

        self.model.task_wise_weight.requires_grad_(False)
        meta_optimizer = torch.optim.Adam(self.shared_mask.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0)

        self.merge_model_weights()
        _results = self.eval_model_on_datasets(self.model)
        results["epoch"].append(0)
        for dataset_name, score in _results.items():
            results[dataset_name].append(score)

        for epoch_idx in tqdm(range(epochs := 2000)):
            # task arithmetic is not optimizable, skip

            # update shared mask
            self.model.train()
            losses = self.compute_loss()
            meta_optimizer.zero_grad()
            losses.backward()
            meta_optimizer.step()
            self.merge_model_weights()

            if cfg.fast_dev_run and epoch_idx > 2:
                break

            if ((epoch_idx + 1) % 200) == 0:
                os.makedirs(self.ckpt_dir, exist_ok=True)
                torch.save(
                    {
                        "task_wise_weight": self.model.task_wise_weight,
                        "shared_mask": self.shared_mask,
                    },
                    self.ckpt_path,
                )
                _results = self.eval_model_on_datasets(self.model)
                results["epoch"].append(epoch_idx + 1)
                for dataset_name, score in _results.items():
                    results[dataset_name].append(score)
                pd.DataFrame(results).to_csv(self.results_path, index=False)

    @torch.no_grad()
    def eval_individuals(self):
        log.info("start eval indivuduals")
        cfg = self.cfg
        shared_mask: ConcreteMask = torch.load(self.ckpt_path, map_location="cpu")["shared_mask"].to(tuple(self.task_vectors[0].values())[0].device)
        concrete_masks = shared_mask._draw_mask()

        results = defaultdict(list)
        Total_score = 0
        for dataset_idx, dataset_name in enumerate(
            tqdm(cfg.test_datasets, desc="evaluating individual models"),
        ):
            model = deepcopy(self.pretrained_model)

            # mask task vector
            task_vector = deepcopy(self.task_vectors[dataset_idx])
            task_vector = shared_mask._apply_mask(concrete_masks, task_vector)

            # add task vector to model
            state_dict = model.state_dict()
            for n, p in task_vector.items():
                state_dict[n] += p.to(state_dict[n].device)
            model.load_state_dict(state_dict)
            model = self.fabric.setup_module(model)

            score = metric_func[dataset_name](model, self.test_loaders[dataset_idx], self.tokenizer)
            results[dataset_name].append(score)
            Total_score += score

        log.info("Eval: " + " Avg score:" + str(Total_score / len(cfg.test_datasets)) + "\n")
        pd.DataFrame(results).to_csv(self.individual_results_path, index=False)


@hydra.main(str(CONFIG_DIR), "flan_t5_default", None)
def main(cfg: DictConfig):
    (program := Program(cfg)).run()


if __name__ == "__main__":
    main()
