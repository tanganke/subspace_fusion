from _common import *

log = logging.getLogger(__name__)

from src.adamerging import AdaMerging, ModelWrapper, make_functional, softmax_entropy
from src.clip_eval import eval_single_dataset_preprocess_head
from src.heads import ClassificationHead, get_classification_head
from src.modeling import ImageEncoder
from src.task_vectors import StateDict, TaskVector
from src.task_wise_fusion import *
from src.task_wise_fusion import check_parameterNamesMatch
from src.utils import num_parameters, timeit_context
from torch.utils.data import DataLoader

from clip_checkpoint_path import CHECKPOINT_DIR, finetuned_model_path, pretrained_model_path, sam_retraining_model_path


def evaluate(
    cfg,
    image_encoder: ImageEncoder,
    classification_heads: Dict[str, ClassificationHead],
    exam_datasets: List[str],
    results: Dict[str, List[Any]],
    dataloaders: Dict[str, DataLoader] = None,
    epoch_idx: int = 0,
):
    Total_ACC = 0.0
    for dataset_name in exam_datasets:
        classification_head = classification_heads[dataset_name]
        metrics = eval_single_dataset_preprocess_head(
            image_encoder,
            classification_head,
            dataset_name,
            cfg,
            dataloader=dataloaders[dataset_name] if dataloaders is not None else None,
        )
        Total_ACC += metrics["top1"]
        log.info(f"Eval: Epoch: {epoch_idx} dataset: {dataset_name} ACC: {metrics['top1']}")

        results["epoch"].append(epoch_idx)
        results["dataset"].append(dataset_name)
        results["acc"].append(metrics["top1"])

    log.info("Eval: init: " + " Avg ACC:" + str(Total_ACC / len(exam_datasets)) + "\n")


@hydra.main(config_path=str(CONFIG_DIR), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    model = cfg.model
    cfg.data_location = DATA_DIR
    cfg.save = str(CHECKPOINT_DIR / cfg.model)
    exam_datasets: List[str] = cfg.datasets

    if cfg.sam_retraining:
        log.info("SAM retrained model is used")
        _finetuned_model_path = sam_retraining_model_path
    else:
        _finetuned_model_path = finetuned_model_path

    with timeit_context():
        log.info("load models")
        pretrained_model: nn.Module = torch.load(pretrained_model_path(model))
        task_vectors: List[StateDict] = [
            TaskVector(
                pretrained_checkpoint=pretrained_model_path(model),
                finetuned_checkpoint=_finetuned_model_path(model, dataset_name),
            ).vector
            for dataset_name in tqdm(cfg.datasets)
        ]
        check_parameterNamesMatch(task_vectors)

    task_wise_weight = get_task_wise_weights(
        len(task_vectors),
        init_values=0.2 if cfg.sam_retraining else 0.3,
    )
    module = TaskWiseMergedModel(
        pretrained_model=pretrained_model,
        task_wise_weight=task_wise_weight,
        task_vectors=task_vectors,
        clamp_weights=True,
    ).cuda()
    module.train_preprocess = pretrained_model.train_preprocess
    module.val_preprocess = pretrained_model.val_preprocess
    module.merge_weights()

    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)

    from src.datasets.common import maybe_dictionarize
    from src.datasets.registry import get_dataset

    if cfg.sam_retraining:
        save_dir = RESULTS_DIR / "sam_retraining" / cfg.model
    else:
        save_dir = RESULTS_DIR / cfg.model
    os.makedirs(save_dir, exist_ok=True)
    results = {"epoch": [], "dataset": [], "acc": []}
    results_path = save_dir / "task_wise_adamerging.csv"

    classification_heads = {dataset_name: get_classification_head(cfg, dataset_name).cuda() for dataset_name in exam_datasets}
    for key in classification_heads:
        head = classification_heads[key].eval()
        for p in head.parameters():
            p.requires_grad_(False)
        classification_heads[key] = head

    datasets = {
        dataset_name: get_dataset(
            dataset_name,
            pretrained_model.val_preprocess,
            location=cfg.data_location,
            batch_size=16,
            num_workers=cfg.num_workers,
        )
        for dataset_name in exam_datasets
    }
    shuffled_test_loaders = {dataset_name: dataset.test_loader_shuffle for dataset_name, dataset in datasets.items()}
    shuffled_test_loader_iters = {dataset_name: iter(itertools.cycle(dataloader)) for dataset_name, dataloader in shuffled_test_loaders.items()}

    evaluate(
        cfg=cfg,
        image_encoder=module,
        classification_heads=classification_heads,
        exam_datasets=exam_datasets,
        dataloaders=shuffled_test_loaders,
        results=results,
    )
    pd.DataFrame(results).to_csv(results_path, index=False)

    for epoch in tqdm(range(epochs := 1000)):
        losses = 0.0
        module.train()
        for dataset_name in exam_datasets:
            # Execute only one batch for each dataset
            batch = next(shuffled_test_loader_iters[dataset_name])
            batch = maybe_dictionarize(batch)
            x = batch["images"].to(cfg.device)
            # y = data["labels"].to(cfg.device)

            outputs = classification_heads[dataset_name](module(x))
            loss = softmax_entropy(outputs).mean(0)
            losses += loss

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        module.merge_weights()

        if ((epoch + 1) % 100) == 0:
            os.makedirs(results_path.parent / os.path.basename(results_path).split(".")[0], exist_ok=True)
            torch.save(
                module.task_wise_weight.detach().cpu(),
                results_path.parent / os.path.basename(results_path).split(".")[0] / os.path.basename(results_path).replace(".csv", ".pt"),
            )

            evaluate(
                cfg,
                image_encoder=module,
                classification_heads=classification_heads,
                dataloaders=shuffled_test_loaders,
                exam_datasets=exam_datasets,
                results=results,
                epoch_idx=epoch + 1,
            )
            pd.DataFrame(results).to_csv(results_path, index=False)

    pd.DataFrame(results).to_csv(results_path, index=False)


if __name__ == "__main__":
    main()
