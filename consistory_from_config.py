from omegaconf import OmegaConf
from utils.model_utils import setup_seed
import sys

from consistory_CLI import run_batch, run_cached_anchors


def load_config():
    sys.argv = ['exp.config_path=configs/generate_example.yaml']
    conf_cli = OmegaConf.from_cli(sys.argv)

    config_path = conf_cli.exp.config_path
    conf_file = OmegaConf.load(config_path)

    config = OmegaConf.merge(conf_file, conf_cli)
    return config


if __name__ == "__main__":
    config = load_config()
    setup_seed(config.generation_args.seed)
    gen_args = config.generation_args

    if config.run.run_type == "batch":
        _, image_all = run_batch(config.story_pipeline.gpu, gen_args.seed, gen_args.n_steps,
                                 gen_args.mask_dropout, gen_args.same_latent, gen_args.share_queries,
                                 gen_args.perform_sdsa, gen_args.perform_injection,
                                 config.save_images.downscale_rate, gen_args.n_achors,
                                 config.prompts.style, config.prompts.subject, config.prompts.concept_token,
                                 config.prompts.settings,
                                 config.save_images.out_dir)
        if config.save_images.save_image_all:
            image_all.save(f'{config.save_images.out_dir}/image_all.png')
    elif config.run.run_type == "cached":
        run_cached_anchors(config.story_pipeline.gpu, gen_args.seed, gen_args.n_steps,
                           gen_args.mask_dropout, gen_args.same_latent, gen_args.share_queries,
                           gen_args.perform_sdsa, gen_args.perform_injection,
                           config.save_images.downscale_rate, gen_args.n_achors,
                           config.prompts.style, config.prompts.subject, config.prompts.concept_token,
                           config.prompts.settings,
                           config.run.cache_cpu_offloading, config.save_images.out_dir)

    else:
        raise ValueError(f"Unknown run type {config.run.run_type}")
