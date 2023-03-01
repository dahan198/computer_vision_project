from model import *
from batch_gen import BatchGenerator
import os
import argparse
import random
import wandb
wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19951012
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--model_dir', default='models')
parser.add_argument('--result_dir', default='results')

nums_epochs = [15, 15, 15, 15, 15]
lrs = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005]
nums_layers = [10, 10, 10, 10, 10]
nums_f_maps = [64, 64, 64, 64, 64]
channel_mask_rates = [0.3, 0.3, 0.3, 0.3, 0.3]
NO_WINDOW = 999_999_999_999
# future_window = 5

# use the full temporal resolution @ 15fps
sample_rate = 2

args = parser.parse_args()

features_dim = 1280
bz = 1
# all_data_files = set(map(lambda x: x.split('.npy')[0], os.listdir("../APAS/" + "features/fold0")))
gt_path = "../APAS/transcriptions_gestures/"
vid_list_file = "../APAS/transcriptions_gestures/"
mapping_file = "../APAS/mapping_gestures.txt"


file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
index2label = dict()
for k, v in actions_dict.items():
    index2label[v] = k
num_classes = len(actions_dict)

# def reset_wandb_env():
#     exclude = {
#         "WANDB_PROJECT",
#         "WANDB_ENTITY",
#         "WANDB_API_KEY",
#     }
#     for k, v in os.environ.items():
#         if k.startswith("WANDB_") and k not in exclude:
#             del os.environ[k]

def main(future_window):
    print(f"future_window = {future_window}")
    print(f"device: {device}")
    num_folds = 5
    for split in range(num_folds):

        num_epochs = nums_epochs[split]
        print(f"Running for {num_epochs} epochs")
        num_layers = nums_layers[split]
        num_f_maps = nums_f_maps[split]
        lr = lrs[split]
        channel_mask_rate = channel_mask_rates[split]
        all_data_files = set(map(lambda x: x.split('.npy')[0], os.listdir("../APAS/" + f"features/fold{split}")))

        valid_files = set(
            map(lambda x: x.split('.csv')[0], (open("../APAS/folds/valid " + str(split) + ".txt").readlines())))
        test_files = set(
            map(lambda x: x.split('.csv')[0], (open("../APAS/folds/test " + str(split) + ".txt").readlines())))
        train_files = all_data_files - valid_files - test_files
        train_data_len = len(train_files) // bz

        future_description = f"future_window_{future_window}" if future_window != NO_WINDOW else "all future"
        features_path = "../APAS/" + "features/fold" + str(split) + "/"
        model_dir = "./{}/".format(args.model_dir) + "/" + str(future_description) + "/split_" + str(split)
        results_dir = "./{}/".format(args.result_dir) + "/" + str(future_description) + "/split_" + str(split)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate, future_window=future_window)
        if args.action == "train":
            config = dict(
                split=split,
                epochs=num_epochs,
                num_layers=num_layers,
                num_f_maps=num_f_maps,
                channel_mask_rate=channel_mask_rate,
                lr=lr,
                future_window=future_description
            )
            # wandb.init(group="experiment_1", job_type="eval")
            # reset_wandb_env()

            wandb.init(
                project="train_report",
                group=future_description,
                job_type=f"fold_{split}",
                name=f"{future_description}_fold_{split}",
                config=config,
                entity='vision_project'
            )

            # wandb.run.name = exp_name + "fold" + str(split)
            batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
            batch_gen.read_data(train_files)

            batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
            batch_gen_tst.read_data(valid_files)

            trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst, train_data_len, wandb)

        if args.action == "predict":
            config = dict(
                split=split,
                future_window=future_window
            )
            wandb.init(
                project="test_report",
                group=f"future_window_{future_window}",
                job_type=f"fold_{split}",
                config=config,
                entity='vision_project'
            )
            batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
            batch_gen_tst.read_data(test_files)
            print(f'using model {num_epochs}')
            trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict, sample_rate,
                            wandb)

        wandb.finish(quiet=True)


if __name__ == '__main__':
    fps = 30 // sample_rate
    windows = [NO_WINDOW]
               # 0,
               # fps,
               # 2 * fps,
               # 3 * fps]
               # 4 * fps,
               # 5 * fps,
               # 7 * fps,
               # 10 * fps,
               # 30 * fps]
    for future_window in windows:
        main(future_window)



