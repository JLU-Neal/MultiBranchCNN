# pylint: disable=E1101,R,C
import os
import numpy as np
from dataset import ProjectFromPointsOnSphere
import torch
import torchvision
import types
import importlib.machinery
from ModelNetDataLoader import ModelNetDataLoader
from tqdm import tqdm
def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]

        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        vote_pool = torch.zeros(target.size()[0],num_class).cuda()
        for _ in range(vote_num):
            pred= classifier(points).data
            vote_pool += pred
        pred = vote_pool/vote_num
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

class KeepName:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, file_name):
        return file_name, self.transform(file_name)


def main(log_dir, augmentation, dataset, batch_size, num_workers,num_point,normal,num_votes,gpu):
    #print(check_output(["nodejs", "--version"]).decode("utf-8"))
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    torch.backends.cudnn.benchmark = True

    # Increasing `repeat` will generate more cached files

    transform = torchvision.transforms.Compose(
        [
            # ToMesh(random_rotations=True, random_translation=0.1),#transform data to mesh
            # ToPoints(random_rotations=True, random_translation=0.1),
            # need to be modified. Originally, the value on the sp  here is based on the ray cast from
            # the points on spherical surface, since that we want to process point cloud data directly,
            # we can try to modified the script, let the ray cast from the point cloud to sphere.
            # ProjectOnSphere(bandwidth=bw)
            ProjectFromPointsOnSphere(bandwidth=64)
        ]
    )
    #transform = KeepName(transform)

    #test_set = Shrec17("data", dataset, perturbed=True, download=True, transform=transform)

    DATA_PATH = 'data/modelnet40_normal_resampled/'
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',normal_channel=args.normal,transform=transform)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    loader = importlib.machinery.SourceFileLoader('model', os.path.join(log_dir, "model.py"))
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    model = mod.Model(40)
    model.cuda()

    model.load_state_dict(torch.load(os.path.join(log_dir, "state.pkl")))

    with torch.no_grad():
        instance_acc, class_acc = test(model.eval(), testDataLoader, vote_num=args.num_votes)
        print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--augmentation", type=int, default=1,
                        help="Generate multiple image with random rotations and translations")
    parser.add_argument("--dataset", choices={"test", "val", "train"}, default="val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=1,
                        help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--gpu', type=str, default='3', help='specify gpu device')
    args = parser.parse_args()

    main(**args.__dict__)
