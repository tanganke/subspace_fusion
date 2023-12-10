from _common import *

log = logging.getLogger(__name__)

from src.clip_eval import eval_single_dataset
from src.task_vectors import TaskVector

from clip_checkpoint_path import CHECKPOINT_DIR, finetuned_model_path, pretrained_model_path, sam_retraining_model_path


@hydra.main(config_path=str(CONFIG_DIR), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg.save = str(CHECKPOINT_DIR / cfg.model)
    cfg.data_location = str(DATA_DIR)

    if cfg.sam_retraining:
        log.info("SAM retrained model is used")
        _finetuned_model_path = sam_retraining_model_path
    else:
        _finetuned_model_path = finetuned_model_path

    task_vectors = [
        TaskVector(
            pretrained_checkpoint=pretrained_model_path(cfg.model),
            finetuned_checkpoint=_finetuned_model_path(cfg.model, dataset_name),
        )
        for dataset_name in cfg.datasets
    ]

    task_vector_sum: TaskVector = sum(task_vectors)

    results = {"scaling_coef": [], "dataset": [], "acc": []}

    for scaling_coef_ in np.linspace(0, 1, 11):
        image_encoder = task_vector_sum.apply_to(pretrained_model_path(cfg.model), scaling_coef=scaling_coef_)
        log.info("*" * 20 + "scaling_coef:" + str(scaling_coef_) + "*" * 20)

        accs = []
        for dataset in cfg.datasets:
            metrics = eval_single_dataset(image_encoder, dataset, cfg)
            log.info(str(dataset) + ":" + str(metrics.get("top1") * 100) + "%")
            acc = metrics.get("top1")
            accs.append(metrics.get("top1") * 100)

            results["scaling_coef"].append(scaling_coef_)
            results["dataset"].append(dataset)
            results["acc"].append(acc)

        log.info("Avg ACC:" + str(np.mean(accs)) + "%")

    if cfg.sam_retraining:
        save_dir = RESULTS_DIR / "sam_retraining" / cfg.model
    else:
        save_dir = RESULTS_DIR / cfg.model
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(save_dir / "task_arithmetic.csv", index=False)


if __name__ == "__main__":
    main()
