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

nums_epochs = [2, 2, 2, 2, 2]
lrs = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005]
nums_layers = [10, 10, 10, 10, 10]
nums_f_maps = [64, 64, 64, 64, 64]
channel_mask_rates = [0.3, 0.3, 0.3, 0.3, 0.3]
# use the full temporal resolution @ 15fps
sample_rate = 1

args = parser.parse_args()

features_dim = 1280
bz = 1
all_data_files = set(map(lambda x: x.split('.npy')[0], os.listdir("../APAS/" + "features/fold0")))
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


def main():
    split = 0

    num_epochs = nums_epochs[split]
    num_layers = nums_layers[split]
    num_f_maps = nums_f_maps[split]
    channel_mask_rate = channel_mask_rates[split]

    config = dict(
        split=split,
        epochs=num_epochs,
        num_layers=num_layers,
        num_f_maps=num_f_maps,
        channel_mask_rate=channel_mask_rate,
    )

    wandb.init(
        project="cv-hyp-test-report",
        config=config,
    )
    wandb.run.name = "fold" + str(split)
    lr = wandb.config.lr

    valid_files = set(
        map(lambda x: x.split('.csv')[0], (open("../APAS/folds/valid " + str(split) + ".txt").readlines())))
    test_files = set(
        map(lambda x: x.split('.csv')[0], (open("../APAS/folds/test " + str(split) + ".txt").readlines())))
    train_files = all_data_files - valid_files - test_files
    train_data_len = len(train_files) // bz

    features_path = "../APAS/" + "features/fold" + str(split) + "/"
    model_dir = "./{}/".format(args.model_dir) + "/split_" + str(split)
    results_dir = "./{}/".format(args.result_dir) + "/split_" + str(split)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate)
    if args.action == "train":
        batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
        batch_gen.read_data(train_files)

        batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
        batch_gen_tst.read_data(valid_files)

        print('----------------------' + str(lr))

        trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst, train_data_len, wandb)

    if args.action == "predict":
        batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
        batch_gen_tst.read_data(test_files)
        trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict, sample_rate,
                        wandb)

    # wandb.finish()


if __name__ == '__main__':
    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'train_loss'},
        'parameters':
            {
                'lr': {'max': 0.0001, 'min': 0.00001}
            }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='fold0')
    wandb.agent(sweep_id, function=main, count=4)




