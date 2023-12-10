from _common import *

log = logging.getLogger(__name__)

from collections import defaultdict

from clip_checkpoint_path import (
    CHECKPOINT_DIR,
    finetuned_model_path,
    pretrained_model_path,
    sam_retraining_model_path,
)

from src.adamerging import softmax_entropy
from src.clip_eval import eval_single_dataset, eval_single_dataset_preprocess_head
from src.concrete_mask import ConcreteMask
from src.datasets.common import maybe_dictionarize
from src.heads import get_classification_head
from src.task_vectors import StateDict, TaskVector
from src.task_wise_fusion import *
from src.task_wise_fusion import check_parameterNamesMatch
from src.tasks.arithmetic import state_dict_sum
from src.utils import num_parameters, timeit_context


class Program:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        cfg.datasets = cfg.seen_datasets + cfg.unseen_datasets
        cfg.save = str(CHECKPOINT_DIR / cfg.model)
        cfg.data_location = str(DATA_DIR)

        result_dir = RESULTS_DIR / cfg.model / cfg.exp_name
        result_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = result_dir / "concrete_task_arithmetic.csv"
        self.ckpt_dir = result_dir / "concrete_task_arithmetic"
        self.ckpt_path = self.ckpt_dir / "ckpt.pt"

        self.fabric = L.Fabric(accelerator="cuda", devices=1)
        self.fabric.launch()

    def run(self):
        self.load_models(task_vector_device=torch.device("cuda"))
        self.load_datasets()
        self.initialize_merged_model()

        self.meta_train()

    def load_models(self, task_vector_device=torch.device("cpu")):
        cfg = self.cfg

        with timeit_context():
            log.info("load models")
            pretrained_model = torch.load(pretrained_model_path(cfg.model), map_location="cpu")
            task_vectors: List[StateDict] = [
                TaskVector(
                    pretrained_checkpoint=pretrained_model_path(cfg.model),
                    finetuned_checkpoint=finetuned_model_path(cfg.model, dataset_name),
                ).vector
                for dataset_name in tqdm(cfg.seen_datasets, "load finetuned checkpoints")
            ]
            # Check if the parameter names in the task vectors match
            check_parameterNamesMatch(task_vectors)

        self.pretrained_model = pretrained_model
        task_vectors = [{k: v.to(task_vector_device, non_blocking=True) for k, v in tv.items()} for tv in task_vectors]
        self.task_vectors = task_vectors

        self.classification_heads = {dataset_name: get_classification_head(cfg, dataset_name).cuda() for dataset_name in cfg.datasets}

    def load_datasets(self):
        from src.datasets.registry import get_dataset

        cfg = self.cfg

        # Load the datasets
        datasets = {
            dataset_name: get_dataset(
                dataset_name,
                self.pretrained_model.val_preprocess,
                location=cfg.data_location,
                batch_size=16,
                num_workers=cfg.num_workers,
            )
            for dataset_name in cfg.datasets
        }
        shuffled_test_loaders = {dataset_name: dataset.test_loader_shuffle for dataset_name, dataset in datasets.items()}
        shuffled_test_loader_iters = {dataset_name: iter(itertools.cycle(dataloader)) for dataset_name, dataloader in shuffled_test_loaders.items()}

        self.datasets = datasets
        self.shuffled_test_loader_iters = shuffled_test_loader_iters

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
        ).to("cuda")

        model.train_preprocess = pretrained_model.train_preprocess
        model.val_preprocess = pretrained_model.val_preprocess
        self.model = model

        log.info(f"total number of parameters in the model: {num_parameters(model)}")

    def meta_train(self):
        log.info("start meta training")
        cfg = self.cfg
        results_path = self.results_path

        results = {"epoch": [], "dataset": [], "acc": []}
        # for task arithmetic, the task-wise weights are not optimizable
        self.model.task_wise_weight.requires_grad_(False)
        meta_optimizer = torch.optim.Adam(
            self.shared_mask.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

        self.merge_model_weights()
        _results = self.eval_model_on_datasets(self.model)
        for dataset_name, score in _results.items():
            results["epoch"].append(0)
            results["dataset"].append(dataset_name)
            results["acc"].append(score)

        for epoch_idx in tqdm(range(epochs := 2000)):
            # task arithmetic is not optimizable, skip

            # meta update
            self.model.train()
            losses = 0
            # * only seen datasets are used for meta training
            for dataset_name in self.cfg.seen_datasets:
                batch = next(self.shuffled_test_loader_iters[dataset_name])
                batch = maybe_dictionarize(batch)
                x = batch["images"].to(cfg.device)

                outputs = self.classification_heads[dataset_name](self.model(x))
                loss = softmax_entropy(outputs).mean(0)
                losses += loss

            meta_optimizer.zero_grad()
            losses.backward()
            meta_optimizer.step()
            self.merge_model_weights()

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
                for dataset_name, score in _results.items():
                    results["epoch"].append(epoch_idx + 1)
                    results["dataset"].append(dataset_name)
                    results["acc"].append(score)
                print(df := pd.DataFrame(results))
                df.to_csv(results_path, index=False)

    @torch.no_grad()
    def eval_model_on_datasets(self, model):
        """
        Evaluates a given model on multiple datasets (both seen tasks and unseen tasks).

        This method sets the model to evaluation mode, then iterates over all datasets specified in the configuration.
        For each dataset, it evaluates the model using the corresponding classification head and logs the accuracy.
        It also calculates and logs the average accuracy across all datasets.

        Args:
            model (torch.nn.Module): The model to be evaluated.

        Returns:
            dict: A dictionary containing two lists: 'dataset' with the names of the datasets, and 'acc' with the corresponding accuracies.
        """
        model.eval()
        results = {}

        Total_ACC = 0
        for dataset_name in tqdm(self.cfg.datasets, "evaluate model on datasets", leave=False):
            classification_head = self.classification_heads[dataset_name]
            metrics = eval_single_dataset_preprocess_head(model, classification_head, dataset_name, self.cfg)
            Total_ACC += metrics["top1"]
            log.info("Eval: init: " + " dataset: " + str(dataset_name) + " ACC: " + str(metrics["top1"]))

            results[dataset_name] = metrics["top1"]

        log.info("Eval: init: " + " Avg ACC:" + str(Total_ACC / len(self.cfg.datasets)) + "\n")
        return results


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="clip_generalization_exp1",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    (program := Program(cfg)).run()


if __name__ == "__main__":
    main()
