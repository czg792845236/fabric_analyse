```
fabric baseline
使用torchvision（0.3.0），像训练分类模型一样训练检测模型
这是一篇帮助新手入门的baseline模型，熟悉检测的同学请忽略。

torchvision在0.3.0版本中集成了Facebook Research检测框架 maskrcnn_benchmark，不用额外编译库，纯python实现，让检测任务更简单，像处理分类任务一样。

本notebook对torchvision detection 训练入口做了部分删减，使更容易理解。

训练环境
pytorch 1.1.0
torchvison 0.3.0
单卡训练
模型训练
1.torchvision detection数据dataset采用coco格式，先使用Fabric2COCO转换脚本将数据转换为coco格式。

2.训练入口文件为train.py,train.py包含几部分:(1)生成data_loader &dataset; (2)建立model; (3)创建optimizer&&lr_scheduler;(4)training，整个代码结构与分类基本相似

3.前传文件为inference.py

文件内容
train.py训练入口；inference.py前传入口；

coco_utils.py只修改了get_coco函数

coco_eval.py文件未做修改

engine.py文件未做修改

utils.py文件未做修改

transforms.py文件未做修改
```
import os
import time
import datetime
import torch
import utils
import transforms as T
import torchvision
import torchvision.models.detection
from coco_utils import get_coco
from engine import train_one_epoch
print("torch.__version__:{}".format(torch.__version__))
print("torchvision.__version__:{}".format(torchvision.__version__))
'''数据扩增'''
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
'''设置数据集，图片存储路径和标注文件路径'''
def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 21),
    }
    DATA_DIR,_,num_classes=paths[name]
    '''
    DATA_DIR:图片、标注存储根目录
    coco_fabric_dataset：布匹数据、标注具体存储位置
    '''
    coco_fabric_dataset={
        "img_dir": "coco_fabric/images/train",
        "ann_file": "coco_fabric/annotations/instances_train.json"
    }
    datesets =get_coco(DATA_DIR,image_set=image_set,data_set=coco_fabric_dataset, transforms=transform)

    return datesets, num_classes
def main(args):
    '''data_loader &dataset'''
    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)

    train_sampler = torch.utils.data.RandomSampler(dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes,
                                                              pretrained=args.pretrained,
                                                              )
    print(model)

    device = torch.device(args.device)
    model.to(device)

    '''optimizer&&lr_scheduler'''
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    #TO DO:resume &distributed

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser.add_argument('--data-path', default='/workdir/data/coco_dataset/', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=3, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=13, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='outputs/fasterrcnn_fpn_50', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",

        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        # default=True,
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)