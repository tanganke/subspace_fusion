from _common import *

log = logging.getLogger(__name__)

from collections import defaultdict

from clip_checkpoint_path import (
    CHECKPOINT_DIR,
    finetuned_model_path,
    pretrained_model_path,
    sam_retraining_model_path,
)

from src.clip_eval import eval_single_dataset, eval_single_dataset_preprocess_head
from src.heads import get_classification_head
from src.task_vectors import StateDict, TaskVector
from src.task_wise_fusion import check_parameterNamesMatch
from src.tasks.arithmetic import state_dict_sum
from src.utils import timeit_context


class Program:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        cfg.datasets = cfg.seen_datasets + cfg.unseen_datasets
        cfg.save = str(CHECKPOINT_DIR / cfg.model)
        cfg.data_location = str(DATA_DIR)

        result_dir = RESULTS_DIR / cfg.model / cfg.exp_name
        result_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = result_dir / "task_arithmetic.csv"

        self.fabric = L.Fabric(accelerator="cuda", devices=1)
        self.fabric.launch()

    def run(self):
        self.load_models()
        self.load_datasets()

        self.eval_task_arithmetic()

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

    def eval_model_on_datasets(self, model):
        """
        Evaluates a given model on multiple datasets.

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

            results[dataset_name]= metrics["top1"]

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
