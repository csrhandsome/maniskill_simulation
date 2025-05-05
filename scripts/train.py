import hydra
import warnings
from omegaconf import OmegaConf
from pathlib import Path

warnings.filterwarnings("ignore")
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    config_path=str(Path(__file__).parent.parent.joinpath("icon/configs")),
    config_name="config.yaml",
    version_base="1.2"
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.train()


if __name__ == "__main__":
    main()