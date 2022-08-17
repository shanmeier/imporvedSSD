from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, \
    COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.modules import MultiBoxLoss, MultiBoxLoss_Focal_Giou
from layers.functions import PriorBox, Detect
from utils.nms_wrapper import nms
import time

parser = argparse.ArgumentParser(
    description='SSD Training')
parser.add_argument('-v', '--version', default='SSD_darknet_self_attention_down_and_up_fusion_v3',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument(
    '--basenet', default='./darknet53_backbone.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=32,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=0,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool, help='Use cpu nms')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=4e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument(
    '--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_best_mAP', default=0,
                    type=int, help='resume best mAP')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=100,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='./weights/',
                    help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
log_file = open(args.save_folder + 'print_log.txt', 'w')
if args.dataset == 'VOC':
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    train_sets = [('2014', 'train'), ('2014', 'valminusminival')]
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net
    cfg = COCO_mobile_300
elif args.version == 'SSD_vgg':
    from models.SSD_Net_vgg import build_net
elif args.version == 'SSD_darknet_self_attention_down_and_up_fusion_v3':
    from models.SSD_darknet_self_attention_down_and_up_fusion_v3 import build_net

else:
    print('Unkown version!')

best_mAP = 0

img_dim = (300, 512)[args.size == '512']
rgb_means = ((104, 117, 123), (103.94, 116.78, 123.68))[args.version == 'RFB_mobile']
p = (0.6, 0.2)[args.version == 'RFB_mobile']
num_classes = (21, 81)[args.dataset == 'COCO']
batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9

net = build_net('train', img_dim, num_classes)
print(net, file=log_file)
print(net)
if args.resume_net is None:
    base_weights = torch.load(args.basenet)
    print('Loading base network...', file=log_file)
    print('Loading base network...')
    net.base.load_state_dict(base_weights)

    def xavier(param):
        init.xavier_uniform(param)


    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0


    print('Initializing weights...', file=log_file)
    print('Initializing weights...')
    # initialize newly added layers' weights with kaiming_normal method
    net.extras.apply(weights_init)
    net.dsf.apply(weights_init)
    net.uf.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    # net.Norm.apply(weights_init)
    if args.version == 'RFB_E_vgg':
        net.reduce.apply(weights_init)
        net.up_reduce.apply(weights_init)

else:
    # load resume network
    print('Loading resume network...', file=log_file)
    print('Loading resume network...')
    best_mAP = args.resume_best_mAP
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.cuda:
    net.cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
#                      momentum=args.momentum, weight_decay=args.weight_decay)

criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
# criterion = MultiBoxLoss_Focal_Giou(num_classes, cfg=cfg, overlap_thresh=0.5, prior_for_matching=True, bkg_label=0,
#                                     neg_mining=True, neg_pos=3, neg_overlap=0.5, encode_target=False,
#                                     use_gpu=args.cuda, loss_c="FocalLoss", loss_r='SmoothL1')
                                    # use_gpu=args.cuda, loss_c="FocalLoss", loss_r='Giou')
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


def test(current_best_mAP):
    test_priorbox = PriorBox(cfg)
    with torch.no_grad():
        test_priors = test_priorbox.forward()
        if args.cuda:
            test_priors = test_priors.cuda()
    global num_classes, img_dim
    test_net = build_net('test', img_dim, num_classes)  # initialize detector
    load_state_dict = torch.load(args.save_folder + args.version + '_' + args.dataset + '_current_best_mAP.pth')
    # create new OrderedDict that does not contain `module.`

    from collections import OrderedDict

    load_new_state_dict = OrderedDict()
    for k, v in load_state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        load_new_state_dict[name] = v
    test_net.load_state_dict(load_new_state_dict)
    test_net.cuda()
    test_net.eval()
    print('Finished loading model:', args.version + '_' + args.dataset + '_current_best_mAP.pth', file=log_file)
    print('Finished loading model:', args.version + '_' + args.dataset + '_current_best_mAP.pth')
    # load data

    if args.dataset == 'VOC':
        testset = VOCDetection(
            VOCroot, [('2007', 'test')], None, AnnotationTransform(), project_root=args.save_folder)
    elif args.dataset == 'COCO':
        testset = COCODetection(
            COCOroot, [('2014', 'minival')], None)
        # COCOroot, [('2015', 'test-dev')], None)
    else:
        print('Only VOC and COCO dataset are supported now!')
    detector = Detect(num_classes, 0, cfg)
    save_folder = os.path.join(args.save_folder, args.dataset)
    rgb_means = ((104, 117, 123), (103.94, 116.78, 123.68))[args.version == 'RFB_mobile']
    # top_k = (300, 200)[args.dataset == 'COCO']
    top_k = 200
    max_per_image = top_k
    # test_net

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    num_images = len(testset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    for i in range(num_images):
        # print("testing", i, "th image...")
        img = testset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        with torch.no_grad():
            transform = BaseTransform(test_net.size, rgb_means, (2, 0, 1))
            x = transform(img).unsqueeze(0)
            if args.cuda:
                x = x.cuda()
                scale = scale.cuda()

        out = test_net(x)  # forward pass
        boxes, scores = detector.forward(out, test_priors)
        boxes = boxes[0]
        scores = scores[0]

        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        for j in range(1, num_classes):
            thresh = 0.01
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = nms(c_dets, 0.45, force_cpu=args.cpu)
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

    print('Evaluating detections', file=log_file)
    print('Evaluating detections')
    testset._write_voc_results_file(all_boxes)
    current_mAP = testset._calculate_mean_mAP(save_folder)
    print("current_mAP:", current_mAP, file=log_file)
    print("current_mAP:", current_mAP)
    if current_mAP > current_best_mAP:
        torch.save(test_net.state_dict(), args.save_folder + args.version + '_' + args.dataset + '_best_mAP' + '.pth')
        current_best_mAP = current_mAP
    return current_best_mAP


def train():
    global best_mAP
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...', file=log_file)
    print('Loading Dataset...')

    if args.dataset == 'VOC':
        dataset = VOCDetection(VOCroot, train_sets, preproc(
            img_dim, rgb_means, p), AnnotationTransform(), project_root=args.save_folder)
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, p))
    else:
        print('Only VOC and COCO are supported now!')
        return

    epoch_size = len(dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues = (stepvalues_VOC, stepvalues_COCO)[args.dataset == 'COCO']
    print('Training', args.version, 'on', dataset.name, file=log_file)
    print('Training', args.version, 'on', dataset.name)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    lr = args.lr
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                                  shuffle=True, num_workers=args.num_workers,
                                                  collate_fn=detection_collate))
            loc_loss = 0
            conf_loss = 0
            # if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
            #     torch.save(net.state_dict(), args.save_folder + args.version + '_' + args.dataset + '_epoches_' +
            #                repr(epoch) + '.pth')
            if epoch > 80:
                torch.save(net.state_dict(), args.save_folder + args.version + '_' + args.dataset
                           + '_current_best_mAP.pth')
                torch.save(optimizer.state_dict(), args.save_folder + 'current_optimizer.pth')
                best_mAP = test(best_mAP)
                print("current_best_mAP:", best_mAP, file=log_file)
                print("current_best_mAP:", best_mAP)
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)

        # print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        load_t1 = time.time()
        if iteration % 50 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f||' % (
                      loss_l.item(), loss_c.item()) +
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr), file=log_file)
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f||' % (
                      loss_l.item(), loss_c.item()) +
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))

    torch.save(net.state_dict(), args.save_folder +
               'Final_' + args.version + '_' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * 5)
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    torch.save(net.state_dict(), args.save_folder +
               'Start_' + args.version + '_' + args.dataset + '.pth')
    train()
    log_file.close()
