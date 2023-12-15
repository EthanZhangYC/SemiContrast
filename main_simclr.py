from experiments.simclr_experiment_ours import SimCLR
import yaml
import argparse
import logging
import os

def parse_option():
    parser = argparse.ArgumentParser("argument for run segmentation pipeline")

    parser.add_argument("--dataset", type=str, default="hippo")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("-e", "--epoch", type=int, default=100)
    parser.add_argument("-f", "--fold", type=int, default=1)
    
    parser.add_argument('--exp_dir', type=str, default='test')
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()
    return args

def get_logger(file_path):
    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

if __name__ == "__main__":
    args = parse_option()
    if args.dataset == "mmwhs":
        with open("config_mmwhs.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    elif args.dataset == "hippo":
        with open("config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    config['batch_size'] = args.batch_size
    config['epochs'] = args.epoch
    config['resume'] = args.resume
    config['exp_dir'] = args.exp_dir
    
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)
    logger = get_logger(args.exp_dir+'/log')
    logger.info(config)
    
    

    simclr = SimCLR(config,logger)
    simclr.train()
