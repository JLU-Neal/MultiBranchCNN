import argparse

import torch
from torch_cluster import fps
from plyfile import PlyData

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    parser.add_argument('--benchmark', '-b', help='Path to benchmark folder')
    parser.add_argument('--outpath', '-o', help='Path to output folder')
    parser.add_argument('--saveply', '-s', action='store_true', help='Save color ply or not')
    args = parser.parse_args()
    print(args)

    #label_tsv = args.benchmark + "/scannet-labels.combined.tsv"
    train_list_file = args.benchmark + "/train.txt"
    test_list_file = args.benchmark + "/test.txt"
    val_list_file = args.benchmark + "/val.txt"
    #label_shapenetcore55 = args.benchmark + "/classes_ObjClassification-ShapeNetCore55.txt"

    ##########################################################Read Source##########################################################

    print("read scene dir:", args.folder)
    scenedir = dir(args.folder, 'd')

    print("read trainval list:", train_list_file)
    train_scene_list = []
    with open(train_list_file, 'r') as train_f:
        for line in train_f.readlines():
            sceneid = line.strip().split("scene")[1]
            spaceid = sceneid.split("_")[0]
            scanid = sceneid.split("_")[1]
            train_scene_list.append(spaceid + scanid)

    print("read test list:", test_list_file)
    test_scene_list = []
    with open(test_list_file, 'r') as train_f:
        for line in train_f.readlines():
            sceneid = line.strip().split("scene")[1]
            spaceid = sceneid.split("_")[0]
            scanid = sceneid.split("_")[1]
            test_scene_list.append(spaceid + scanid)

    print("read val list:", val_list_file)
    val_scene_list = []
    with open(val_list_file, 'r') as train_f:
        for line in train_f.readlines():
            sceneid = line.strip().split("scene")[1]
            spaceid = sceneid.split("_")[0]
            scanid = sceneid.split("_")[1]
            val_scene_list.append(spaceid + scanid)






    # split scene to train and test
    process_train_list = []
    process_test_list = []

    for scene in scenedir:

        sceneid = scene.strip().split("scene")[1]
        spaceid = sceneid.split("_")[0]
        scanid = sceneid.split("_")[1]
        scenename = spaceid + scanid

        if scenename in train_scene_list:

            process_train_list.append(scene)

        elif scenename in test_scene_list:

            process_test_list.append(scene)

    print("Train all:", len(train_scene_list), "Test all:", len(test_scene_list), "Dir all:", len(scenedir))
    print("Process Train:", len(process_train_list), "Process Test:", len(process_test_list))
    ##########################################################Process Data##########################################################
    print("Process Train Scene:")

    for scene in process_train_list:
        ply_file = scene + "/scene" + sceneid + "_vh_clean_2.ply"
        # Read ply file
        print("\nRead ply file:", ply_file)
        plydata = PlyData.read(ply_file).elements[0].data
        pts_num = len(plydata)
        print("points num:", pts_num)
        data = torch.tensor(plydata)
        batch = torch.zeros(pts_num)
        index = fps(data, batch, ratio=0.5, random_start=False)
        data = data[index]
        torch.save(data)

    print("Process Test Scene:")

    for scene in process_test_list:
        scene2instances(scene, args.outpath + "/test/", [label_map, label_info], label_shapenetcore55_map, args.saveply)