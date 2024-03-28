#!/usr/bin/env python3
R"""
This script finetunes a pretrained language model on a text-to-text dataset.

Usage:

```bash
# see config/flan_t5_finetune.yaml for all options
python scripts/flan_t5_finetune.py datasets=glue-mnli model=flan-t5-base
```

"""
from _common import *

log = logging.getLogger(__name__)

import lightning.pytorch as pl
import peft
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from peft.tuners.lora import LoraLayer
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator

from datasets import DatasetDict, load_dataset, load_from_disk
from src.utils import TitledLog, num_devices

log = logging.getLogger(__name__)

# disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Seq2SeqLMModule(pl.LightningModule):
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM | peft.PeftModel,
        tokenizer: AutoTokenizer,
        optim_cfg: DictConfig,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optim_cfg = optim_cfg

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Dict: Dictionary containing the optimizer and learning rate scheduler.
        """
        optim = {}
        if "optimizer" in self.optim_cfg:
            optim["optimizer"]: torch.optim.Optimizer = instantiate(
                self.optim_cfg["optimizer"],
                params=self.parameters(),
            )
        if "lr_scheduler" in self.optim_cfg:
            optim["lr_scheduler"]: torch.optim.lr_scheduler.LRScheduler = instantiate(
                self.optim_cfg["lr_scheduler"],
                optimizer=optim["optimizer"],
            )
        if self.trainer.is_global_zero:
            log.info(f"{'configure_optimizers':=^50}")
            log.info(optim)
        return optim

    def training_step(self, batch, batch_idx: int):
        # batch has keys ["input_ids", "attention_mask", "labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # trim the input_ids and attention_mask
        max_len = input_ids.size(1)
        while torch.all(attention_mask[:, max_len - 1] == 0):
            max_len -= 1
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]

        outputs = self.forward(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def save_trainable_parameters(self):
        """
        Saves the trainable parameters of the model.
        """
        if self.logger.log_dir is not None:
            # save trainable parameters
            ckpt_path = Path(self.trainer.log_dir) / "checkpoints" / f"epoch={self.current_epoch}_step={self.global_step}.pth"
            if not ckpt_path.parent.exists():
                Path.mkdir(ckpt_path.parent, exist_ok=True)
            state_dict = dict((k, p) for k, p in self.model.named_parameters() if p.requires_grad)
            torch.save(state_dict, ckpt_path)
        else:
            log.warning("logger.log_dir is None, skip saving trainable parameters.")

    def on_train_epoch_end(self) -> None:
        self.save_trainable_parameters()


def load_model_from_config(cfg: DictConfig):
    """
    This function loads a model and its tokenizer from a given configuration.

    Args:
        cfg (DictConfig): The configuration dictionary containing the model and tokenizer information.

    Returns:
        dict: A dictionary containing the tokenizer, model, and module.

    If a PEFT (Parameter-Efficient Fine-tuning) configuration is provided in the cfg,
    it will instantiate a PEFT model. If a seed is provided, it will set the seed for reproducibility.
    If the 'linearize' option is set to True in the model configuration, it will partially linearize the PEFT model.
    If no PEFT configuration is found, it will log a message and proceed with full finetuning.
    """
    tokenizer = instantiate(
        cfg.model.tokenizer,
    )
    model = instantiate(
        cfg.model.model,
    )
    if cfg.peft.peft_config is not None:
        log.info(f"peft config found, use peft model.")
        peft_config: peft.PeftConfig = instantiate(cfg.peft.peft_config)
        #  https://github.com/huggingface/peft/issues/567
        peft_config.target_modules = list(peft_config.target_modules)
        if hasattr(cfg.peft, "seed") and cfg.peft.seed is not None:
            log.info(f"set peft seed to {cfg.peft.seed}")
            L.seed_everything(cfg.peft.seed)
        model = peft.get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        log.info(f"no peft config found, use full finetuning.")

    module = Seq2SeqLMModule(model, tokenizer, optim_cfg=cfg.optim)
    return dict(tokenizer=tokenizer, model=model, module=module)


@hydra.main(str(CONFIG_DIR), "flan_t5_finetune", None)
def main(cfg: DictConfig):
    # cfg.PROJECT_DIR = PROJECT_DIR
    if hasattr(cfg, "seed") and cfg.seed is not None:
        log.info(f"set seed to {cfg.seed}")
        L.seed_everything(cfg.seed)

    # setup trainer, save config
    logger = TensorBoardLogger(LOG_DIR / cfg.model.name / cfg.dataset.name, name=cfg.peft.name)
    trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer), logger=logger)
    if trainer.log_dir is not None and trainer.is_global_zero:
        log.info(f"log_dir: {trainer.log_dir}")
        os.makedirs(trainer.log_dir, exist_ok=True)
        OmegaConf.save(cfg, Path(trainer.log_dir) / "config.yaml")

    # print config to console
    if trainer.is_global_zero:
        print(OmegaConf.to_yaml(cfg))

    # load pretrained model and tokenizer
    with TitledLog("load pretrained model and tokenizer", log_fn=log.info):
        _return = load_model_from_config(cfg)
        tokenizer: AutoTokenizer = _return["tokenizer"]
        model: AutoModelForSeq2SeqLM | peft.PeftModel = _return["model"]
        module: Seq2SeqLMModule = _return["module"]

    # load dataset
    with TitledLog("load datasets and dataloaders", log_fn=log.info):
        if (cache_dir := CACHE_DIR / "datasets" / cfg.dataset.name).exists():
            log.info(f"loading dataset {cfg.dataset.name} from cache {cache_dir}")
            datasets: DatasetDict = load_from_disk(cache_dir)
        else:
            datasets: DatasetDict = instantiate(cfg.dataset.datasets)

            # convert the task to text-to-text format
            if hasattr(cfg.dataset, "preprocessor"):
                if hasattr(cfg.dataset.preprocessor, "template_file"):
                    cfg.dataset.preprocessor.template_file = str(TEMPLATE_DIR / cfg.dataset.preprocessor.template_file)
                preprocessor = instantiate(
                    cfg.dataset.preprocessor,
                    tokenizer=tokenizer,
                    tokenizer_kwargs=cfg.model.tokenizer_kwargs if hasattr(cfg.model, "tokenizer_kwargs") else None,
                )
                datasets = datasets.map(
                    preprocessor,
                    **cfg.dataset.map_kwargs if hasattr(cfg.dataset, "map_kwargs") else {"batched": True},
                )
                datasets.save_to_disk(cache_dir)

        train_dataset = datasets["train"]

        # create dataloaders
        assert cfg.batch_size % num_devices(cfg.trainer.devices) == 0, "batch_size must be divisible by the number of devices."
        batch_size = cfg.batch_size // num_devices(cfg.trainer.devices)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
            collate_fn=default_data_collator,
        )

    trainer.fit(
        module,
        train_dataloaders=train_loader,
    )

    if trainer.is_global_zero:
        module.save_trainable_parameters()
    exit(0)


if __name__ == "__main__":
    main()
