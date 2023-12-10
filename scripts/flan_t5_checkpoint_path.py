from _common import *
from transformers import AutoConfig, AutoModelForSeq2SeqLM

log = logging.getLogger(__name__)

CHECKPOINT_DIR = CACHE_DIR / "checkpoints"
MODELS = ["flan-t5-base", "flan-t5-large"]
DATASETS = ["glue-cola", "glue-mnli", "glue-mrpc", "glue-qnli", "glue-qqp", "glue-rte", "glue-sst2", "glue-stsb", "glue-wnli"]


def finetuned_model_path(model_name: str, dataset_name: str, template_name: str = "glue_v1") -> Path:
    if model_name not in MODELS:
        log.warning(f"Unknown model {model_name}")
    if dataset_name not in DATASETS:
        log.warning(f"Unknown dataset {dataset_name}")
    ckpt_path = CHECKPOINT_DIR / model_name / template_name / f"{dataset_name}.pth"
    return ckpt_path


if __name__ == "__main__":
    config = AutoConfig.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_config(config)
    exit(0)
