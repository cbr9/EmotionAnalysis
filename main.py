import hydra
import dotenv
dotenv.load_dotenv()
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from src.model import EmotionModel
from src.dataset import EmotionDataModule


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    datamodule: EmotionDataModule = hydra.utils.instantiate(cfg.dataset)
    model: EmotionModel = hydra.utils.instantiate(cfg.model, n_classes=datamodule.n_classes)
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.tune(datamodule=datamodule, model=model)

    cfg.model.lr = model.learning_rate
    cfg.dataset.batch_size = datamodule.batch_size
    OmegaConf.save(config=cfg, f=".hydra/config.yaml")

    trainer.fit(datamodule=datamodule, model=model)
    trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    main()