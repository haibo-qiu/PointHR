from pcr.engines.defaults import default_argument_parser, default_config_parser, default_setup, Trainer
from pcr.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = Trainer(cfg)
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    if 'ShapeNetPart' in cfg.dataset_type:
        from pcr.engines.partseg import Trainer

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
