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
            train_l2_edge_loss_list = pickle.load(t)
            train_iou_list = pickle.load(t)
        with open(val_path, 'rb') as v:
            val_l1_loss_list = pickle.load(v)
            val_l2_edge_loss_list = pickle.load(v)
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

def main2():
    plt.figure(figsize=(12,18))

    model_lists_dirs = ["../outputFiles/results_from_server/lists/8","../outputFiles/results_from_server/lists/9","../outputFiles/results_from_server/lists/10"]
    
    model_lists_dir = model_lists_dirs[0]
    counter_path = os.path.join(model_lists_dir, "counter_list.obj")
    train_path = os.path.join(model_lists_dir, "train_lists.obj")
    val_path = os.path.join(model_lists_dir, "val_lists.obj")

    with open(train_path, 'rb') as t:
        train_l1_loss_list = pickle.load(t)
        train_l2_edge_loss_list = pickle.load(t)
        train_iou_list = pickle.load(t)
    with open(val_path, 'rb') as v:
        val_l1_loss_list = pickle.load(v)
        val_l2_edge_loss_list = pickle.load(v)
        val_iou_list = pickle.load(v)
    with open(counter_path, 'rb') as c:
        counter_list = pickle.load(c)

    plt.subplot(321)
    plt.plot(counter_list, val_l1_loss_list, 'r-', counter_list, train_l1_loss_list, 'b-')
    plt.title('l1_loss')
    # plt.subplot(132)
    # plt.plot(counter_list, val_l2_edge_loss_list, 'r-', counter_list, train_l2_edge_loss_list, 'b-')
    # plt.title('l2_edge_loss')
    plt.subplot(322)
    plt.title('iou')
    plt.plot(counter_list, val_iou_list, 'r-', counter_list, train_iou_list, 'b-')
    
    model_lists_dir = model_lists_dirs[1]
    counter_path = os.path.join(model_lists_dir, "counter_list.obj")
    train_path = os.path.join(model_lists_dir, "train_lists.obj")
    val_path = os.path.join(model_lists_dir, "val_lists.obj")

    with open(train_path, 'rb') as t:
        train_l1_loss_list = pickle.load(t)
        train_l2_edge_loss_list = pickle.load(t)
        train_iou_list = pickle.load(t)
    with open(val_path, 'rb') as v:
        val_l1_loss_list = pickle.load(v)
        val_l2_edge_loss_list = pickle.load(v)
        val_iou_list = pickle.load(v)
    with open(counter_path, 'rb') as c:
        counter_list = pickle.load(c)

    plt.subplot(323)
    plt.plot(counter_list, val_l1_loss_list, 'r-', counter_list, train_l1_loss_list, 'b-')
    plt.title('l1_loss')
    # plt.subplot(132)
    # plt.plot(counter_list, val_l2_edge_loss_list, 'r-', counter_list, train_l2_edge_loss_list, 'b-')
    # plt.title('l2_edge_loss')
    plt.subplot(324)
    plt.title('iou')
    plt.plot(counter_list, val_iou_list, 'r-', counter_list, train_iou_list, 'b-')

    model_lists_dir = model_lists_dirs[2]
    counter_path = os.path.join(model_lists_dir, "counter_list.obj")
    train_path = os.path.join(model_lists_dir, "train_lists.obj")
    val_path = os.path.join(model_lists_dir, "val_lists.obj")

    with open(train_path, 'rb') as t:
        train_l1_loss_list = pickle.load(t)
        train_l2_edge_loss_list = pickle.load(t)
        train_iou_list = pickle.load(t)
    with open(val_path, 'rb') as v:
        val_l1_loss_list = pickle.load(v)
        val_l2_edge_loss_list = pickle.load(v)
        val_iou_list = pickle.load(v)
    with open(counter_path, 'rb') as c:
        counter_list = pickle.load(c)

    plt.subplot(325)
    plt.plot(counter_list, val_l1_loss_list, 'r-', counter_list, train_l1_loss_list, 'b-')
    plt.title('l1_loss')
    # plt.subplot(132)
    # plt.plot(counter_list, val_l2_edge_loss_list, 'r-', counter_list, train_l2_edge_loss_list, 'b-')
    # plt.title('l2_edge_loss')
    plt.subplot(326)
    plt.title('iou')
    plt.plot(counter_list, val_iou_list, 'r-', counter_list, train_iou_list, 'b-')

    plt.show()


if __name__ == '__main__':
    if (len(sys.argv) == 1):
        print("Please specify experiment ids as arguments")
        exit()
    main(sys.argv[1:])