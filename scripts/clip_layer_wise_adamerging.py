from _common import *

log = logging.getLogger(__name__)

from src.adamerging import ModelWrapper, load_weights, make_functional, softmax_entropy
from src.clip_eval import eval_single_dataset_preprocess_head
from src.heads import get_classification_head
from src.task_vectors import StateDict, TaskVector
from torch.utils.data import DataLoader

from clip_checkpoint_path import CHECKPOINT_DIR, finetuned_model_path, pretrained_model_path, sam_retraining_model_path


class AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets, cfg):
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.pretrain_lambdas = torch.ones(len(paramslist[0]), 1)
        prior = 0.3
        rlambdas = torch.ones(len(paramslist[0]), len(paramslist) - 1) * prior  # (1 * tasks)
        self.lambdas_raw = torch.nn.Parameter(rlambdas)

        self.classifier = []
        for dataset_name in exam_datasets:
            classification_head = get_classification_head(cfg, dataset_name)
            layer_name = "classifier_{}".format(dataset_name)
            self.add_module(layer_name, classification_head.to(cfg.device))
            self.classifier.append(layer_name)

    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass

    def collect_trainable_params(self):
        return [self.lambdas_raw]

    def get_classification_head(self, dataset_name):
        layer_name = "classifier_{}".format(dataset_name)
        classification_head = getattr(self, layer_name)
        return classification_head

    def get_image_encoder(self):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model

    def forward(self, inp, dataset_name):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)

        layer_name = "classifier_{}".format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)
        return out


@hydra.main(config_path=str(CONFIG_DIR), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg.save = str(CHECKPOINT_DIR / cfg.model)
    cfg.data_location = str(DATA_DIR)
    model = cfg.model

    if cfg.sam_retraining:
        log.info("SAM retrained model is used")
        _finetuned_model_path = sam_retraining_model_path
    else:
        _finetuned_model_path = finetuned_model_path

    exam_datasets = cfg.datasets
    pretrained_checkpoint = pretrained_model_path(model)

    task_vectors = [
        TaskVector(
            pretrained_checkpoint,
            _finetuned_model_path(model, dataset_name),
        )
        for dataset_name in exam_datasets
    ]

    pretrained_model: nn.Module = torch.load(pretrained_checkpoint)
    pretrained_model_dic: StateDict = pretrained_model.state_dict()

    model = ModelWrapper(pretrained_model, exam_datasets)
    model = model.to(cfg.device)
    _, names = make_functional(model)

    paramslist = []
    paramslist += [tuple(v.detach().requires_grad_(False).cpu() for _, v in pretrained_model_dic.items())]  # pretrain
    paramslist += [tuple(v.detach().requires_grad_(False).cpu() for _, v in tv.vector.items()) for i, tv in enumerate(task_vectors)]  # task vectors
    torch.cuda.empty_cache()
    adamerging_mtl_model = AdaMerging(paramslist, model, names, exam_datasets, cfg=cfg)

    print("init lambda:")
    print(adamerging_mtl_model.lambdas())
    print("collect_trainable_params:")
    print(list(adamerging_mtl_model.collect_trainable_params()))

    optimizer = torch.optim.Adam(adamerging_mtl_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)

    from src.datasets.common import maybe_dictionarize
    from src.datasets.registry import get_dataset

    results = {"epoch": [], "dataset": [], "acc": []}
    if cfg.sam_retraining:
        save_dir = RESULTS_DIR / "sam_retraining" / cfg.model
    else:
        save_dir = RESULTS_DIR / cfg.model
    os.makedirs(save_dir, exist_ok=True)
    results_path: Path = save_dir / "layer_wise_adamerging.csv"

    Total_ACC = 0.0
    for dataset_name in exam_datasets:
        image_encoder = adamerging_mtl_model.get_image_encoder()
        classification_head = adamerging_mtl_model.get_classification_head(dataset_name)
        metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, cfg)
        Total_ACC += metrics["top1"]
        log.info("Eval: init: " + " dataset: " + str(dataset_name) + " ACC: " + str(metrics["top1"]))

    pd.DataFrame(results).to_csv(results_path, index=False)
    log.info("Eval: init: " + " Avg ACC:" + str(Total_ACC / len(exam_datasets)) + "\n")

    datasets = {
        dataset_name: get_dataset(
            dataset_name, pretrained_model.val_preprocess, location=cfg.data_location, batch_size=16, num_workers=cfg.num_workers
        )
        for dataset_name in exam_datasets
    }
    shuffled_test_loaders: Dict[str, DataLoader] = {dataset_name: dataset.test_loader_shuffle for dataset_name, dataset in datasets.items()}
    shuffled_test_loader_iters = {dataset_name: iter(itertools.cycle(dataloader)) for dataset_name, dataloader in shuffled_test_loaders.items()}

    for epoch in tqdm(range(epochs := 1000)):
        losses = 0.0
        for dataset_name in exam_datasets:
            # Execute only one batch for each dataset
            batch = next(shuffled_test_loader_iters[dataset_name])
            batch = maybe_dictionarize(batch)
            x = batch["images"].to(cfg.device)
            # y = data["labels"].to(cfg.device)

            outputs = adamerging_mtl_model(x, dataset_name)
            loss = softmax_entropy(outputs).mean(0)
            losses += loss

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        log.debug(list(adamerging_mtl_model.lambdas().data))

        if ((epoch + 1) % 100) == 0:
            os.makedirs(results_path.parent / os.path.basename(results_path).split(".")[0], exist_ok=True)
            torch.save(
                adamerging_mtl_model.lambdas().data,
                results_path.parent / os.path.basename(results_path).split(".")[0] / os.path.basename(results_path).replace(".csv", ".pt"),
            )

            Total_ACC = 0.0
            for dataset_name in exam_datasets:
                image_encoder = adamerging_mtl_model.get_image_encoder()
                classification_head = adamerging_mtl_model.get_classification_head(dataset_name)
                metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, cfg)
                Total_ACC += metrics["top1"]
                log.info("Eval: Epoch: " + str(epoch) + " dataset: " + str(dataset_name) + " ACC: " + str(metrics["top1"]))

                results["epoch"].append(epoch + 1)
                results["dataset"].append(dataset_name)
                results["acc"].append(metrics["top1"])

            log.info("Eval: Epoch: " + str(epoch) + " Avg ACC:" + str(Total_ACC / len(exam_datasets)) + "\n")
            pd.DataFrame(results).to_csv(results_path, index=False)


if __name__ == "__main__":
    main()
