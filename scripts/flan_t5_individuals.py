from _common import *

log = logging.getLogger(__name__)

from collections import defaultdict

import lightning as L
import lightning.pytorch as pl
import peft
from flan_t5_checkpoint_path import finetuned_model_path
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator

from datasets import DatasetDict, load_dataset, load_from_disk
from src.tasks.arithmetic import state_dict_sub
from src.utils import num_devices

# disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def remove_special_tokens(tokenizer, token_list: list):
    """
    This function removes special tokens from a list of tokens. It also stops processing
    when it encounters a token with a value of -100.

    Parameters:
        tokenizer (Tokenizer): The tokenizer object used for tokenizing text.
        token_list (list): The list of tokens to be processed.

    Returns:
        list: The list of tokens after removing special tokens.
    """
    ret = []
    for token in token_list:
        if token not in tokenizer.all_special_ids and token > 0:
            ret.append(token)
        if token == -100:
            break
    return ret


def evaluate_accuracy(model, val_loader: DataLoader, tokenizer):
    """
    This function evaluates the accuracy of a language model on a validation set.

    Parameters:
        model (nn.Module): The language model to be evaluated.
        val_loader (DataLoader): The DataLoader object containing the validation data.
        tokenizer (Tokenizer): The tokenizer object used for tokenizing text.

    Returns:
        float: The accuracy of the model on the validation set.
    """
    from tqdm import tqdm

    correct = 0
    total = 0

    model = model.eval()
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="round", leave=False)):
        with torch.no_grad():
            outputs = model.generate(batch["input_ids"])
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            labels = [remove_special_tokens(tokenizer, l) for l in batch["labels"]]
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # compare output_text and labels
            for i, j in zip(output_text, labels):
                if i == j:
                    correct += 1
                total += 1

    # return accuracy
    return correct / total


def evaluate_spearman_rho(model, val_loader: DataLoader, tokenizer):
    """
    This function evaluates the Spearman's rank correlation coefficient (rho) between the model's predictions and the actual labels on a validation set.

    Parameters:
        model (nn.Module): The language model to be evaluated.
        val_loader (DataLoader): The DataLoader object containing the validation data.
        tokenizer (Tokenizer): The tokenizer object used for tokenizing text.

    Returns:
        float: The Spearman's rho between the model's predictions and the actual labels.
    """
    from tqdm import tqdm

    model = model.eval()
    all_preds: List[str] = []
    all_labels: List[str] = []
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="round", leave=False)):
        with torch.no_grad():
            outputs = model.generate(batch["input_ids"])
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            labels = [remove_special_tokens(tokenizer, l) for l in batch["labels"]]
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_preds.extend(output_text)
            all_labels.extend(labels)

    # save `all_preds` and `all_labels`
    # with open("temp/all_preds.txt", "w") as f:
    #     for preds in all_preds:
    #         for pred in preds:
    #             f.write(pred + "\n")
    # with open("temp/all_labels.txt", "w") as f:
    #     for labels in all_labels:
    #         for label in labels:
    #             f.write(label + "\n")

    # calculate spearman's rho
    # 1. convert string list `all_preds` and `all_labels` to numpy array
    # 2. compute spearman's rho
    from scipy.stats import spearmanr

    def parse_flost(s: str):
        import math

        try:
            return float(s)
        except:
            return 0.0

    all_preds = np.array([parse_flost(pred) for pred in all_preds])
    all_labels = np.array([parse_flost(label) for label in all_labels])
    rho = spearmanr(all_preds, all_labels)[0]
    return rho


# metric_func is a dictionary that maps task names to their respective evaluation functions.
# By default, it uses the evaluate_accuracy function for all tasks.
# However, for the "glue-stsb" task, it uses the evaluate_spearman_rho function.
metric_func: Dict[str, Any] = defaultdict(lambda: evaluate_accuracy)
metric_func["glue-stsb"] = evaluate_spearman_rho


class Program:
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
        self.results_path = self.results_dir / "individuals.csv"

        self.fabric = L.Fabric(accelerator="cuda", devices=1)
        self.fabric.launch()

    def run(self):
        self.load_models()
        self.load_datasets()

        self.eval_individuals()

    def eval_individuals(self):
        results = defaultdict(list)

        # pretrained model
        log.info("evaluating pretrained model")
        model = deepcopy(self.pretrained_model)
        model = self.fabric.setup_module(model)
        _results = self.eval_model_on_datasets(model)
        results["model"].append("pretrained")
        for dataset_name, score in _results.items():
            results[dataset_name].append(score)
        print(pd.DataFrame(results))

        # finetuned models
        for dataset_name, task_vector in track(
            zip(self.cfg.test_datasets, self.task_vectors),
            "evaluating fine-tuned models",
        ):
            model = deepcopy(self.pretrained_model)
            for n, p in task_vector.items():
                model.state_dict()[n] += p
            model = self.fabric.setup_module(model)
            _results = self.eval_model_on_datasets(model)
            results["model"].append(dataset_name)
            for dataset_name, score in _results.items():
                results[dataset_name].append(score)
            print(pd.DataFrame(results))

        # save results
        pd.DataFrame(results).to_csv(self.results_path, index=False)

    @torch.no_grad()
    def eval_model_on_datasets(self, model):
        """
        Evaluates the given model on the test datasets specified in the configuration.
        It iterates over each test dataset and its corresponding data loader, computes the metric for the dataset,
        and logs the score. If the `fast_dev_run` configuration option is set, it only evaluates on one batch per dataset.
        The results are returned as a dictionary mapping dataset names to scores.

        Args:
            model (torch.nn.Module): The model to evaluate.

        Returns:
            dict: A dictionary mapping dataset names to evaluation scores.
        """
        results = {}
        model.eval()
        for dataset_name, test_loader in tqdm(
            zip(self.cfg.test_datasets, self.test_loaders),
            desc="evaluating",
            total=len(self.cfg.test_datasets),
            leave=False,
        ):
            if self.cfg.fast_dev_run:
                log.info("fast_dev_run is True, only evaluate on one batch")
                test_loader = [next(iter(test_loader))]
            score = metric_func[dataset_name](model, test_loader, self.tokenizer)
            log.info(f"Eval: dataset: {dataset_name} score: {score}")
            results[dataset_name] = score
        return results

    def load_models(self, task_vector_device=torch.device("cpu")):
        """
        Loads the pretrained model and tokenizer. If a PEFT (parameter-efficient fine-tuning) configuration is provided,
        it also sets up the PEFT model. It then loads the fine-tuned weights for each dataset specified in the configuration.
        If PEFT is used, it computes the task vector as the difference between the PEFT model after merging and unloading,
        and the pretrained model. The task vectors are stored in the instance variable `task_vectors`.

        Args:
            task_vector_device (torch.device, optional): The device to which the task vectors are moved. Defaults to CPU.

        Side Effects:
            Sets the instance variables `tokenizer`, `pretrained_model`, and `task_vectors`.
        """
        cfg = self.cfg

        # load pretrained model and tokenizer
        log.info("loading pretrained model and tokenizer")
        self.tokenizer = instantiate(self.cfg.model.tokenizer)
        self.pretrained_model = instantiate(self.cfg.model.model)
        # if PEFT configuration is provided, set up the PEFT model
        if cfg.peft.peft_config is not None:
            peft_config = peft.PeftConfig = instantiate(cfg.peft.peft_config)
            peft_config.target_modules = list(peft_config.target_modules)
            if hasattr(cfg.peft, "seed") and cfg.peft.seed is not None:
                log.debug(f"set peft seed to {cfg.peft.seed}")
                L.seed_everything(cfg.peft.seed)
            peft_model = peft.get_peft_model(
                deepcopy(self.pretrained_model), peft_config
            )
            peft_model.print_trainable_parameters()
        # fix all parameters of pre-trained model to save memory
        self.pretrained_model.eval()
        for n, p in self.pretrained_model.named_parameters():
            p.requires_grad_(False)

        # load task vectors
        pretrained_sd = self.pretrained_model.state_dict()
        task_vectors = []
        for dataset_name in track(
            cfg.test_datasets, description="loading finetuned weights"
        ):
            if cfg.peft.peft_config is None:
                weight_file = finetuned_model_path(
                    cfg.model.name, template_name="glue_v1", dataset_name=dataset_name
                )
                weights = torch.load(weight_file, map_location="cpu")
                with torch.no_grad():
                    task_vectors.append(
                        state_dict_sub(weights, pretrained_sd, strict=False)
                    )
            else:
                # if peft is used, we need to load the peft model and merge the task vector
                # the task vector is the difference between the peft model after merged and the pretrained model
                weight_file = finetuned_model_path(
                    cfg.model.name, template_name="glue_v1", dataset_name=dataset_name
                )
                weight_file = weight_file.parent / cfg.peft.name / weight_file.name
                weights = torch.load(weight_file, map_location="cpu")
                peft_model.load_state_dict(weights, strict=False)
                with torch.no_grad():
                    # merge and unload the peft model, then compute the task vector
                    _peft_model = deepcopy(peft_model).merge_and_unload()
                    _task_vector = state_dict_sub(
                        _peft_model.state_dict(), pretrained_sd, strict=False
                    )
                    _task_vector = {
                        k: v for k, v in _task_vector.items() if not torch.all(v == 0)
                    }
                    assert len(_task_vector) > 0, "task vector is empty"
                    task_vectors.append(_task_vector)
        # move task vectors to `task_vector_device`
        task_vectors = [
            {
                k: v.detach_().to(task_vector_device, non_blocking=True)
                for k, v in tv.items()
            }
            for tv in task_vectors
        ]
        self.task_vectors = task_vectors

        assert not isinstance(
            self.pretrained_model, peft.PeftModel
        ), "pretrained model should not be a peft model"

    def load_datasets(self):
        """
        Loads the datasets specified in the configuration. If a dataset is found in the cache, it is loaded from there.
        Otherwise, it is instantiated from the configuration, optionally preprocessed, and saved to the cache.
        The method then checks for a validation split in the dataset and raises an error if none is found.
        The validation split is added to the list of test datasets.
        Finally, it sets up the data loaders for the test datasets, both with and without shuffling, and creates an iterator
        for each shuffled test loader.

        Raises:
            ValueError: If a dataset has no validation split.

        Side Effects:
            Sets the instance variables `test_datasets`, `test_loaders`, `shuffled_test_loaders`, and
            `shuffled_test_loader_iters`.
        """
        test_datasets = []
        for dataset_name in track(
            self.cfg.test_datasets, description="loading datasets"
        ):
            if (cache_dir := CACHE_DIR / "datasets" / dataset_name).exists():
                log.info(f"loading dataset {dataset_name} from cache")
                dataset = load_from_disk(cache_dir)
            else:
                config = OmegaConf.load(
                    CONFIG_DIR / "datasets" / f"{dataset_name}.yaml"
                )
                dataset = instantiate(config.datasets)

                if hasattr(config, "preprocessor"):
                    log.info("preprocessing the dataset")
                    preprocessor = instantiate(
                        config.preprocessor,
                        template_file=TEMPLATE_DIR
                        / "glue_v1"
                        / os.path.basename(config.preprocessor.template_file),
                        tokenizer=self.tokenizer,
                        tokenizer_kwargs=self.cfg.model.tokenizer_kwargs
                        if hasattr(self.cfg.model, "tokenizer_kwargs")
                        else None,
                    )
                    dataset = dataset.map(
                        preprocessor,
                        **config.map_kwargs if hasattr(config, "map_kwargs") else {},
                    )
                    dataset.save_to_disk(cache_dir)

            if "validation" in dataset:
                test_dataset = dataset["validation"]
            elif "validation_matched" in dataset:
                test_dataset = dataset["validation_matched"]
            else:
                raise ValueError("dataset has no validation split.")
            test_datasets.append(test_dataset)

        self.test_datasets = test_datasets
        self.test_loaders = self.fabric.setup_dataloaders(
            *[
                DataLoader(
                    test_dataset,
                    batch_size=self.cfg.batch_size * 3,
                    num_workers=self.cfg.num_workers,
                    shuffle=False,
                    collate_fn=default_data_collator,
                )
                for test_dataset in test_datasets
            ]
        )
        self.shuffled_test_loaders = self.fabric.setup_dataloaders(
            *[
                DataLoader(
                    test_dataset,
                    batch_size=self.cfg.batch_size,
                    num_workers=self.cfg.num_workers,
                    shuffle=True,
                    collate_fn=default_data_collator,
                )
                for test_dataset in test_datasets
            ]
        )

        self.shuffled_test_loader_iters = {
            dataset_name: iter(itertools.cycle(test_loader))
            for dataset_name, test_loader in zip(
                self.cfg.test_datasets, self.shuffled_test_loaders
            )
        }


@hydra.main(str(CONFIG_DIR), "flan_t5_default", None)
def main(cfg: DictConfig):
    (program := Program(cfg)).run()


if __name__ == "__main__":
    main()
