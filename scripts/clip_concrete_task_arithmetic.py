from _common import *

log = logging.getLogger(__name__)

from clip_checkpoint_path import (
    CHECKPOINT_DIR,
    finetuned_model_path,
    pretrained_model_path,
    sam_retraining_model_path,
)
from timer import timer

from src.adamerging import softmax_entropy
from src.clip_eval import eval_single_dataset, eval_single_dataset_preprocess_head
from src.concrete_mask import ConcreteMask
from src.datasets.common import maybe_dictionarize
from src.heads import get_classification_head
from src.task_vectors import StateDict, TaskVector
from src.task_wise_fusion import *
from src.task_wise_fusion import check_parameterNamesMatch
from src.utils import num_parameters, timeit_context


class Program:
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        cfg.save = str(CHECKPOINT_DIR / cfg.model)
        cfg.data_location = str(DATA_DIR)

        save_dir = RESULTS_DIR / cfg.model
        if cfg.version is not None:
            save_dir = save_dir / f"version_{cfg.version}"
        os.makedirs(save_dir, exist_ok=True)
        self.results_path = results_path = save_dir / "clip_concrete_task_arithmetic.csv"
        self.ckpt_dir = results_path.parent / os.path.basename(results_path).split(".")[0]
        self.ckpt_path = self.ckpt_dir / os.path.basename(results_path).replace(".csv", ".pt")
        self.individual_results_path = save_dir / "clip_concrete_task_arithmetic_individuals.csv"
        log.info(f'results will be saved to "{self.results_path}"')

    def run(self):
        self.load_models()
        self.load_datasets()
        self.initialize_merged_model()

        self.meta_train()
        # self.eval_binary_mask()
        self.eval_individuals()

    def meta_train(self):
        log.info("start meta training")
        cfg = self.cfg
        results_path = self.results_path

        results = {"epoch": [], "dataset": [], "acc": []}
        # for task arithmetic, the task-wise weights are not optimizable
        self.model.task_wise_weight.requires_grad_(False)
        meta_optimizer = torch.optim.Adam(
            self.shared_mask.parameters(),
            lr=cfg.lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.wd,
        )

        self.merge_model_weights()
        # self.eval_model_on_datasets(epoch_idx=0, results=results)

        for epoch_idx in (pbar := tqdm(range(epochs := 2000))):
            # task arithmetic is not optimizable, skip

            # meta update
            self.model.train()
            losses = 0
            for dataset_name in self.cfg.datasets:
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
            pbar.set_description(f"epoch: {epoch_idx + 1}/{epochs}, loss: {losses.item():.2f}")

            if cfg.fast_dev_run and epoch_idx > 2:
                break

            if ((epoch_idx + 1) % 1000) == 0:
                os.makedirs(self.ckpt_dir, exist_ok=True)
                torch.save(
                    {
                        "task_wise_weight": self.model.task_wise_weight,
                        "shared_mask": self.shared_mask,
                    },
                    self.ckpt_path,
                )
                self.eval_model_on_datasets(epoch_idx=epoch_idx + 1, results=results)
                pd.DataFrame(results).to_csv(results_path, index=False)

    def eval_binary_mask(self):
        log.info("start eval binary mask")
        cfg = self.cfg
        # load shared_mask from `ckpt_path`:
        shared_mask: ConcreteMask = torch.load(self.ckpt_path, map_location="cpu")["shared_mask"].to(tuple(self.task_vectors[0].values())[0].device)
        shared_mask.temperature = 0.001
        concrete_masks = shared_mask._draw_mask(binary_mask=True)
        concrete_masks = {k: v.float() for k, v in concrete_masks.items()}
        self.model.task_vectors = shared_mask.apply_mask(self.task_vectors, concrete_masks)
        self.model.merge_weights()

        self.eval_model_on_datasets(epoch_idx=0, results=None)

    def eval_individuals(self):
        log.info("start eval individuals")
        cfg = self.cfg
        # load shared_mask from `ckpt_path`:
        shared_mask: ConcreteMask = torch.load(self.ckpt_path, map_location="cpu")["shared_mask"].to(tuple(self.task_vectors[0].values())[0].device)
        concrete_masks = shared_mask._draw_mask()

        results = {"dataset": [], "acc": []}
        Total_ACC = 0
        for dataset_idx, dataset_name in enumerate(tqdm(cfg.datasets, desc="evaluating individual models")):
            model = deepcopy(self.pretrained_model)

            # mask task vector
            task_vector = deepcopy(self.task_vectors[dataset_idx])
            task_vector = shared_mask._apply_mask(concrete_masks, task_vector)

            # add task vector to model
            state_dict = model.state_dict()
            for n, p in task_vector.items():
                state_dict[n] += p.to(state_dict[n].device)
            model.load_state_dict(state_dict)
            model = model.cuda()

            metrics = eval_single_dataset_preprocess_head(
                model,
                self.classification_heads[dataset_name],
                dataset_name,
                cfg,
                dataloader=self.test_loaders[dataset_name],
            )
            Total_ACC += metrics["top1"]
            log.info("Eval: " + " dataset: " + str(dataset_name) + " ACC: " + str(metrics["top1"]))

            results["dataset"].append(dataset_name)
            results["acc"].append(metrics["top1"])

        log.info("Eval: " + " Avg ACC:" + str(Total_ACC / len(cfg.datasets)) + "\n")
        pd.DataFrame(results).to_csv(self.individual_results_path, index=False)

    @timer("merge_model_weights", unit="s")
    def merge_model_weights(self):
        """this method is called every time `shared_mask` is updated"""
        self.model.task_vectors = self.shared_mask.apply_mask(self.task_vectors)
        self.model.merge_weights()

    @torch.no_grad()
    def eval_model_on_datasets(
        self,
        epoch_idx: int,
        results: Dict[str, List[float]],
    ):
        self.model.eval()

        Total_ACC = 0
        for dataset_name in self.cfg.datasets:
            classification_head = self.classification_heads[dataset_name]
            metrics = eval_single_dataset_preprocess_head(
                self.model,
                classification_head,
                dataset_name,
                self.cfg,
                dataloader=self.test_loaders[dataset_name],  # must pass corruption loader if cfg.corruption is not None
            )
            Total_ACC += metrics["top1"]
            log.info("Eval: init: " + " dataset: " + str(dataset_name) + " ACC: " + str(metrics["top1"]))

            if results is not None:
                results["epoch"].append(epoch_idx)
                results["dataset"].append(dataset_name)
                results["acc"].append(metrics["top1"])

        log.info("Eval: init: " + " Avg ACC:" + str(Total_ACC / len(self.cfg.datasets)) + "\n")

    def load_models(self):
        cfg = self.cfg

        if cfg.sam_retraining:
            log.info("SAM retrained model is used")
            _finetuned_model_path = sam_retraining_model_path
        else:
            _finetuned_model_path = finetuned_model_path
        with timeit_context():
            log.info("load models")
            pretrained_model: nn.Module = torch.load(pretrained_model_path(cfg.model), map_location="cpu")
            task_vectors: List[StateDict] = [
                TaskVector(
                    pretrained_checkpoint=pretrained_model_path(cfg.model),
                    finetuned_checkpoint=_finetuned_model_path(cfg.model, dataset_name),
                ).vector
                for dataset_name in tqdm(cfg.datasets)
            ]
            # Check if the parameter names in the task vectors match
            check_parameterNamesMatch(task_vectors)

        self.pretrained_model = pretrained_model
        #! if gpu memory is not enough, comment the following line
        if cfg.model == "ViT-B-32" or cfg.model == "ViT-B-16":
            task_vectors = [{k: v.cuda(0) for k, v in tv.items()} for tv in task_vectors]
        # elif cfg.model == "ViT-L-14":
        #     temp = []
        #     for tv in task_vectors[:4]:
        #         temp.append({k: v.cuda(1, non_blocking=True) for k, v in tv.items()})
        #     for tv in task_vectors[4:]:
        #         temp.append({k: v.cuda(2, non_blocking=True) for k, v in tv.items()})
        #     task_vectors = temp
        self.task_vectors = task_vectors

        self.classification_heads = {dataset_name: get_classification_head(cfg, dataset_name).cuda() for dataset_name in cfg.datasets}

    def initialize_merged_model(self):
        pretrained_model = self.pretrained_model
        task_vectors = self.task_vectors
        for p in pretrained_model.parameters():
            p.detach_().requires_grad_(False)

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

    def load_datasets(self):
        if self.cfg.corruption is None:
            from src.datasets.registry import get_dataset
        else:
            from src.datasets.corruption.registry import get_dataset

        cfg = self.cfg

        # Load the datasets
        datasets_kwargs = dict(
            location=cfg.data_location,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        if cfg.corruption is not None:
            datasets_kwargs["corruption"] = cfg.corruption
        datasets = {
            dataset_name: get_dataset(
                dataset_name,
                self.pretrained_model.val_preprocess,
                **datasets_kwargs,
            )
            for dataset_name in cfg.datasets
        }
        shuffled_test_loaders = {dataset_name: dataset.test_loader_shuffle for dataset_name, dataset in datasets.items()}
        shuffled_test_loader_iters = {dataset_name: iter(itertools.cycle(dataloader)) for dataset_name, dataloader in shuffled_test_loaders.items()}

        self.datasets = datasets
        self.test_loaders = {dataset_name: dataset.test_loader for dataset_name, dataset in datasets.items()}
        self.shuffled_test_loader_iters = shuffled_test_loader_iters


@hydra.main(config_path=str(CONFIG_DIR), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    Program(cfg).run()


if __name__ == "__main__":
    main()
