# pylint: disable=E1101,R,C,W1202
import ctypes

import torch
import torch.nn.functional as F
import torchvision

import os
import shutil
import time
import logging
import copy
import types
import importlib.machinery
import numpy as np
from dataset import ProjectFromPointsOnSphere
from ModelNetDataLoader import ModelNetDataLoader
from tqdm import tqdm
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/data2/tzf/s2cnn-master', '/data2/tzf/s2cnn-master/examples/shrec17', '/data2/tzf/s2cnn-master/examples', '/data2/tzf/s2cnn-master/draft', '/data2/tzf/s2cnn-master'])

def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred= classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc
def main(log_dir, model_path, augmentation, dataset, batch_size, learning_rate, num_workers,num_point,normal,gpu):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    arguments = copy.deepcopy(locals())

    os.mkdir(log_dir)
    shutil.copy2(__file__, os.path.join(log_dir, "script.py"))
    shutil.copy2(model_path, os.path.join(log_dir, "model.py"))

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    logger.addHandler(fh)

    logger.info("%s", repr(arguments))

    torch.backends.cudnn.benchmark = True

    # Load the model
    loader = importlib.machinery.SourceFileLoader('model', os.path.join(log_dir, "model.py"))
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    model = mod.Model(40)
    model.cuda()

    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    logger.info("{} paramerters in the last layer".format(sum(x.numel() for x in model.out_layer.parameters())))

    bw = model.bandwidths[0]
    _file = 'libmatrix.so'
    _path = '/data2/tzf/s2cnn-master/draft/' + _file
    lib = ctypes.cdll.LoadLibrary(_path)
    # Load the dataset
    # Increasing `repeat` will generate more cached files
    transform=torchvision.transforms.Compose(
        [
            #ToMesh(random_rotations=True, random_translation=0.1),#transform data to mesh
            #ToPoints(random_rotations=True, random_translation=0.1),
            #need to be modified. Originally, the value on the sp  here is based on the ray cast from
            #the points on spherical surface, since that we want to process point cloud data directly,
            #we can try to modified the script, let the ray cast from the point cloud to sphere.
            #ProjectOnSphere(bandwidth=bw)
            ProjectFromPointsOnSphere(bandwidth=bw,lib=lib)
        ]
    )



    #train_set = Shrec17("/data2/tzf/s2cnn-master/examples/shrec17/data", dataset, perturbed=True, download=False, transform=transform, target_transform=target_transform)


    #train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)


    DATA_PATH = 'data/modelnet40_normal_resampled/'

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train',normal_channel=args.normal,transform=transform)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',normal_channel=args.normal,transform=transform)
    train_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,num_workers=4, pin_memory=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0, momentum=0.9)

    def train_step(data, target):
        model.train()
        data, target = data.cuda(), target.cuda()

        prediction = model(data)
        loss = F.nll_loss(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = prediction.data.max(1)[1].eq(target.data).long().cpu().sum()

        return loss.item(), correct.item()

    def get_learning_rate(epoch):
        limits = [100, 200]
        lrs = [1, 0.1, 0.01]
        assert len(lrs) == len(limits) + 1
        for lim, lr in zip(limits, lrs):
            if epoch < lim:
                return lr * learning_rate
        return lrs[-1] * learning_rate

    def train():
        for epoch in range(300):

            lr = get_learning_rate(epoch)
            logger.info("learning rate = {} and batch size = {}".format(lr, train_loader.batch_size))
            for p in optimizer.param_groups:
                p['lr'] = lr

            total_loss = 0
            total_correct = 0
            time_before_load = time.perf_counter()
            for batch_idx, (data, target) in enumerate(train_loader):  # x
                time_after_load = time.perf_counter()
                time_before_step = time.perf_counter()
                loss, correct = train_step(data, target)

                total_loss += loss
                total_correct += correct

                logger.info("[{}:{}/{}] LOSS={:.2} <LOSS>={:.2} ACC={:.2} <ACC>={:.2} time={:.2}+{:.2}".format(
                    epoch, batch_idx, len(train_loader),
                    loss, total_loss / (batch_idx + 1),
                          correct / len(data), total_correct / len(data) / (batch_idx + 1),
                          time_after_load - time_before_load,
                          time.perf_counter() - time_before_step))
                time_before_load = time.perf_counter()

            torch.save(model.state_dict(), os.path.join(log_dir, "state.pkl"))

    best_instance_acc = 0.0
    best_class_acc = 0.0
    for epoch in range(200):

        lr = get_learning_rate(epoch)
        logger.info("learning rate = {} and batch size = {}".format(lr, train_loader.batch_size))
        for p in optimizer.param_groups:
            p['lr'] = lr

        total_loss = 0
        total_correct = 0
        time_before_load = time.perf_counter()
        for batch_idx, (data, target) in enumerate(train_loader):#x
            time_after_load = time.perf_counter()
            time_before_step = time.perf_counter()

            target=target[:,0]
            data,target=data.cuda(),target.cuda()
            loss, correct = train_step(data, target.long())

            total_loss += loss
            total_correct += correct

            logger.info("[{}:{}/{}] LOSS={:.2} <LOSS>={:.2} ACC={:.2} <ACC>={:.2} time={:.2}+{:.2}".format(
                epoch, batch_idx, len(train_loader),
                loss, total_loss / (batch_idx + 1),
                correct / len(data), total_correct / len(data) / (batch_idx + 1),
                time_after_load - time_before_load,
                time.perf_counter() - time_before_step))
            time_before_load = time.perf_counter()


        torch.save(model.state_dict(), os.path.join(log_dir, "state.pkl"))

        with torch.no_grad():
            instance_acc, class_acc = test(model.eval(), testDataLoader)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            print('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = '/data2/tzf/s2cnn-master/examples/shrec17/best_model.pth'
                print('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)







if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--augmentation", type=int, default=1,
                        help="Generate multiple image with random rotations and translations")
    parser.add_argument("--dataset", choices={"test", "val", "train"}, default="train")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.5)
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 1024]')
    parser.add_argument('--normal', action='store_true', default=False,help='Whether to use normal information [default: False]')
    parser.add_argument('--gpu', type=str, default='3', help='specify gpu device')
    args = parser.parse_args()

    main(**args.__dict__)
