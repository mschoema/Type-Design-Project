import os
import sys
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt

LISTS_DIR = "../outputFiles/results_from_server/lists/"

def main(experiment_ids):
    ids = os.listdir(LISTS_DIR)
    print(ids)
    n = 0
    available_experiment_ids = []
    for experiment_id in experiment_ids:
        if experiment_id in ids:
            n += 1
            available_experiment_ids.append(experiment_id)
    plt.figure(figsize=(24,24))
    plt.subplot(221)
    plt.title("Training l1 loss")
    plt.subplot(222)
    plt.title("Training iou")
    plt.subplot(223)
    plt.title("Validation l1 loss")
    plt.subplot(224)
    plt.title("Validation iou")
    count = 0
    palette = plt.get_cmap('Set1')
    for experiment_id in available_experiment_ids:
        experiment_path = os.path.join(LISTS_DIR, experiment_id)
        counter_path = os.path.join(experiment_path, "counter_list.obj")
        train_path = os.path.join(experiment_path, "train_lists.obj")
        val_path = os.path.join(experiment_path, "val_lists.obj")
        with open(train_path, 'rb') as t:
            train_l1_loss_list = pickle.load(t)
            train_iou_list = pickle.load(t)
        with open(val_path, 'rb') as v:
            val_l1_loss_list = pickle.load(v)
            val_iou_list = pickle.load(v)
        with open(counter_path, 'rb') as c:
            counter_list = pickle.load(c)
        
        count += 1
        plt.subplot(221)
        plt.plot(counter_list, train_l1_loss_list, marker='', color=palette(count), linewidth=1, alpha=0.9, label=experiment_id)
        plt.subplot(222)
        plt.plot(counter_list, train_iou_list, marker='', color=palette(count), linewidth=1, alpha=0.9, label=experiment_id)
        plt.subplot(223)
        plt.plot(counter_list, val_l1_loss_list, marker='', color=palette(count), linewidth=1, alpha=0.9, label=experiment_id)
        plt.subplot(224)
        plt.plot(counter_list, val_iou_list, marker='', color=palette(count), linewidth=1, alpha=0.9, label=experiment_id)

    plt.figlegend(available_experiment_ids, loc = 'upper center', ncol=n, labelspacing=0. )
    plt.show()

if __name__ == '__main__':
    if (len(sys.argv) == 1):
        print("Please specify experiment ids as arguments")
        exit()
    main(sys.argv[1:])