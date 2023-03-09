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
decoders_num = 2

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='predict')
parser.add_argument('--model_dir', default='models')
parser.add_argument('--result_dir', default='results')

nums_epochs = [20, 20, 20, 20, 20]
lrs = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005]
nums_layers = [7, 7, 7, 7, 7]
nums_f_maps = [64, 64, 64, 64, 64]
num_folds = 5
channel_mask_rates = [0.3, 0.3, 0.3, 0.3, 0.3]
NO_WINDOW = 100

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


def init_split(split,future_window):
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

    trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate,
                      future_window=future_window, decoders_num=decoders_num)

    return num_epochs, num_layers, num_f_maps, lr, channel_mask_rate, valid_files, test_files, train_files, \
           train_data_len, future_description, features_path, model_dir, results_dir, trainer


def main(future_window, split=None):
    print(f"future_window = {future_window}")
    print(f"device: {device}")
    if args.action == "train":
        for split in range(num_folds):
            num_epochs, num_layers, num_f_maps, lr, channel_mask_rate, valid_files, test_files, train_files, \
            train_data_len, future_description, features_path, model_dir, results_dir, trainer = init_split(split)

            config = dict(
                split=split,
                epochs=num_epochs,
                num_layers=num_layers,
                num_f_maps=num_f_maps,
                channel_mask_rate=channel_mask_rate,
                lr=lr,
                future_window=future_description,
                decoders_num=decoders_num
            )

            wandb.init(
                project="final_train_report decoders_num=2 layers_num=7",
                group=future_description,
                job_type=f"fold_{split}",
                name=f"{future_description}_fold_{split}",
                config=config,
                entity='vision_project'
            )

            batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
            batch_gen.read_data(train_files)

            batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
            batch_gen_tst.read_data(valid_files)

            trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst, train_data_len, wandb)
            wandb.finish(quiet=True)

    if args.action == "predict":
        config = dict(
            split=split,
        )
        wandb.init(
            project="test_report_final_18",
            config=config,
            entity='vision_project',
            name=f"fold_{split}"
        )
        my_data = []
        plot = {'accuracy': [], 'edit distance': [], 'F1@10': [], 'F1@25': [], 'F1@50': []}
        for future_window in [0, 3, 7, 11, 15, 30, NO_WINDOW]:
            print('--------------------------- split ', str(split), '---------------------------------------')
            num_epochs, num_layers, num_f_maps, lr, channel_mask_rate, valid_files, test_files, train_files, \
            train_data_len, future_description, features_path, model_dir, results_dir, trainer = init_split(split, future_window)
            batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
            batch_gen_tst.read_data(test_files)
            print(f'using model {num_epochs}')
            data, plot = trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                         sample_rate, wandb, future_window=future_window, plot=plot,
                                         future_description=future_description)
            my_data.append(data)
        columns = ["image", "window future", "stage 1", "stage 2", "stage 3"]
        wandb.log({"fold" + str(split) + f"/future window{future_window}": wandb.Table(data=my_data, columns=columns)})
        for p in plot.keys():
            wandb.log({"fold" + str(split) + "/" + p: wandb.plot.line(wandb.Table(data=plot[p],
                                                                                 columns=["window size", p]),
                                          "window size", p, title=p)})
        wandb.finish(quiet=True)


if __name__ == '__main__':
    fps = 30 // sample_rate
    # windows = [NO_WINDOW, 0, 3, 7, 11, 15, 30]
    for split in range(5):
        # for future_window in windows:
        main(0, split)
