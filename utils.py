import cv2
import glob
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import paddle

import dataset as dset

from paddle.vision import transforms

import paddle.distributed as dist
from paddle.io import Dataset, DataLoader, DistributedBatchSampler


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def denorm(x):
    out = (x + 1) / 2
    return out.clip(0, 1)


def write_config_to_file(config, save_path):
    with open(os.path.join(save_path, 'config.txt'), 'w') as file:
        for arg in vars(config):
            file.write(str(arg) + ': ' + str(getattr(config, arg)) + '\n')



def make_transform(resize=True, imsize=128, centercrop=False, centercrop_size=128,
                   totensor=True, normalize=True, norm_mean=(0.5, 0.5, 0.5), norm_std=(0.5, 0.5, 0.5)):
        options = []
        if resize:
            options.append(transforms.Resize((imsize,imsize)))
        if centercrop:
            options.append(transforms.CenterCrop(centercrop_size))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize(norm_mean, norm_std))
        transform = transforms.Compose(options)
        return transform


def make_dataloader(batch_size, dataset_type, data_path, shuffle=True, drop_last=True, dataloader_args={},
                    resize=True, imsize=128, centercrop=False, centercrop_size=128, totensor=True,
                    normalize=True, norm_mean=(0.5, 0.5, 0.5), norm_std=(0.5, 0.5, 0.5), num_workers=4):
    # Make transform
    transform = make_transform(resize=resize, imsize=imsize,
                               centercrop=centercrop, centercrop_size=centercrop_size,
                               totensor=totensor, normalize=normalize, norm_mean=norm_mean, norm_std=norm_std)
    # Make dataset
    if dataset_type in ['folder', 'imagenet', 'lfw']:
        # folder dataset
        assert os.path.exists(data_path), "data_path does not exist! Given: " + data_path
        dataset = dset.ImageFolder(root=data_path, transform=transform)
        print('#training images = %d' % len(dataset))
    elif dataset_type == 'lsun':
        assert os.path.exists(data_path), "data_path does not exist! Given: " + data_path
        dataset = dset.LSUN(root=data_path, classes=['bedroom_train'], transform=transform)
    elif dataset_type == 'cifar10':
        if not os.path.exists(data_path):
            print("data_path does not exist! Given: {}\nDownloading CIFAR10 dataset...".format(data_path))
        dataset = dset.CIFAR10(root=data_path, download=True, transform=transform)
    elif dataset_type == 'fake':
        dataset = dset.FakeData(image_size=(3, centercrop_size, centercrop_size), transform=transforms.ToTensor())
    assert dataset
    num_of_classes = len(dataset.classes)
    print("Data found!  # of images =", len(dataset), ", # of classes =", num_of_classes, ", classes:", dataset.classes)
    # Make dataloader from dataset
    sampler = DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)
    dist.init_parallel_env()
    # print('#dataloader = %d' % len(dataloader))
    return dataloader, num_of_classes


def make_gif(image, iteration_number, save_path, model_name, max_frames_per_gif=100):

    # Make gif
    gif_frames = []

    # Read old gif frames
    try:
        gif_frames_reader = imageio.get_reader(os.path.join(save_path, model_name + ".gif"))
        for frame in gif_frames_reader:
            gif_frames.append(frame[:, :, :3])
    except:
        pass

    # Append new frame
    im = cv2.putText(np.concatenate((np.zeros((32, image.shape[1], image.shape[2])), image), axis=0),
                     'iter %s' % str(iteration_number), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA).astype('uint8')
    gif_frames.append(im)

    # If frames exceeds, save as different file
    if len(gif_frames) > max_frames_per_gif:
        print("Splitting the GIF...")
        gif_frames_00 = gif_frames[:max_frames_per_gif]
        num_of_gifs_already_saved = len(glob.glob(os.path.join(save_path, model_name + "_*.gif")))
        print("Saving", os.path.join(save_path, model_name + "_%05d.gif" % (num_of_gifs_already_saved)))
        imageio.mimsave(os.path.join(save_path, model_name + "_%05d.gif" % (num_of_gifs_already_saved)), gif_frames_00)
        gif_frames = gif_frames[max_frames_per_gif:]

    # Save gif
    # print("Saving", os.path.join(save_path, model_name + ".gif"))
    imageio.mimsave(os.path.join(save_path, model_name + ".gif"), gif_frames)


def make_plots(G_losses, D_losses, D_losses_real, D_losses_fake, D_xs, D_Gz_trainDs, D_Gz_trainGs, log_step, save_path, init_epoch=0):
    iters = np.arange(len(D_losses))*log_step + init_epoch
    fig = plt.figure(figsize=(20, 20))
    plt.subplot(311)
    plt.plot(iters, np.zeros(iters.shape), 'k--', alpha=0.5)
    plt.plot(iters, G_losses, color='C0', label='G')
    plt.legend()
    plt.title("Generator loss")
    plt.xlabel("Iterations")
    plt.subplot(312)
    plt.plot(iters, np.zeros(iters.shape), 'k--', alpha=0.5)
    plt.plot(iters, D_losses_real, color='C1', alpha=0.7, label='D_real')
    plt.plot(iters, D_losses_fake, color='C2', alpha=0.7, label='D_fake')
    plt.plot(iters, D_losses, color='C0', alpha=0.7, label='D')
    plt.legend()
    plt.title("Discriminator loss")
    plt.xlabel("Iterations")
    plt.subplot(313)
    plt.plot(iters, np.zeros(iters.shape), 'k--', alpha=0.5)
    plt.plot(iters, np.ones(iters.shape), 'k--', alpha=0.5)
    plt.plot(iters, D_xs, alpha=0.7, label='D(x)')
    plt.plot(iters, D_Gz_trainDs, alpha=0.7, label='D(G(z))_trainD')
    plt.plot(iters, D_Gz_trainGs, alpha=0.7, label='D(G(z))_trainG')
    plt.legend()
    plt.title("D(x), D(G(z))")
    plt.xlabel("Iterations")
    plt.savefig("plots.png")
    plt.clf()
    plt.close()


def save_ckpt(sagan_obj, model=False, final=False):
    print("Saving ckpt...")

    if final:
        # Save final - both model and state_dict
        paddle.save({
                    'step': sagan_obj.step,
                    'G_state_dict': sagan_obj.G.module.state_dict() if hasattr(sagan_obj.G, "module") else sagan_obj.G.state_dict(),    # "module" in case DataParallel is used
                    'G_optimizer_state_dict': sagan_obj.G_optimizer.state_dict(),
                    'D_state_dict': sagan_obj.D.module.state_dict() if hasattr(sagan_obj.D, "module") else sagan_obj.D.state_dict(),    # "module" in case DataParallel is used,
                    'D_optimizer_state_dict': sagan_obj.D_optimizer.state_dict(),
                    }, os.path.join(sagan_obj.config.model_weights_path, '{}_final_state_dict_ckpt_{:07d}.pdparams'.format(sagan_obj.config.name, sagan_obj.step)))
        paddle.save({
                    'step': sagan_obj.step,
                    'G': sagan_obj.G.module if hasattr(sagan_obj.G, "module") else sagan_obj.G,
                    'G_optimizer': sagan_obj.G_optimizer,
                    'D': sagan_obj.D.module if hasattr(sagan_obj.D, "module") else sagan_obj.D,
                    'D_optimizer': sagan_obj.D_optimizer,
                    }, os.path.join(sagan_obj.config.model_weights_path, '{}_final_model_ckpt_{:07d}.pdparams'.format(sagan_obj.config.name, sagan_obj.step)))

    elif model:
        # Save full model (not state_dict)
        paddle.save({
                    'step': sagan_obj.step,
                    'G': sagan_obj.G.module if hasattr(sagan_obj.G, "module") else sagan_obj.G,     # "module" in case DataParallel is used
                    'G_optimizer': sagan_obj.G_optimizer,
                    'D': sagan_obj.D.module if hasattr(sagan_obj.D, "module") else sagan_obj.D,     # "module" in case DataParallel is used
                    'D_optimizer': sagan_obj.D_optimizer,
                    }, os.path.join(sagan_obj.config.model_weights_path, '{}_model_ckpt_{:07d}.pdparams'.format(sagan_obj.config.name, sagan_obj.step)))

    else:
        # Save state_dict
        paddle.save({
                    'step': sagan_obj.step,
                    'G_state_dict': sagan_obj.G.module.state_dict() if hasattr(sagan_obj.G, "module") else sagan_obj.G.state_dict(),
                    'G_optimizer_state_dict': sagan_obj.G_optimizer.state_dict(),
                    'D_state_dict': sagan_obj.D.module.state_dict() if hasattr(sagan_obj.D, "module") else sagan_obj.D.state_dict(),
                    'D_optimizer_state_dict': sagan_obj.D_optimizer.state_dict(),
                    }, os.path.join(sagan_obj.config.model_weights_path, 'ckpt_{:07d}.pdparams'.format(sagan_obj.step)))
        # save_filename = '%s_net.pdparams' % ('latest')
        # paddle.save({
        #     'step': sagan_obj.step,
        #     'G_state_dict': sagan_obj.G.module.state_dict() if hasattr(sagan_obj.G,
        #                                                               "module") else sagan_obj.G.state_dict(),
        #     'G_optimizer_state_dict': sagan_obj.G_optimizer.state_dict(),
        #     'D_state_dict': sagan_obj.D.module.state_dict() if hasattr(sagan_obj.D,
        #                                                               "module") else sagan_obj.D.state_dict(),
        #     'D_optimizer_state_dict': sagan_obj.D_optimizer.state_dict(),
        #     }, os.path.join(sagan_obj.config.model_weights_path, save_filename))


def load_pretrained_model(sagan_obj):
    print("Loading pretrained_model", sagan_obj.config.pretrained_model, "...")
    # Check for path
    assert os.path.exists(sagan_obj.config.pretrained_model), "Path of .pdparams pretrained_model doesn't exist! Given: " + sagan_obj.config.pretrained_model
    checkpoint = paddle.load(sagan_obj.config.pretrained_model)
    # If we know it is a state_dict (instead of complete model)
    if sagan_obj.config.state_dict_or_model == 'state_dict':
        sagan_obj.start = checkpoint['step'] + 1
        sagan_obj.G.load_dict(checkpoint['G_state_dict'])
        sagan_obj.G_optimizer.set_state_dict(checkpoint['G_optimizer_state_dict'])
        sagan_obj.D.load_dict(checkpoint['D_state_dict'])
        sagan_obj.D_optimizer.set_state_dict(checkpoint['D_optimizer_state_dict'])
    # Else, if we know it is a complete model (and not just state_dict)
    elif sagan_obj.config.state_dict_or_model == 'model':
        sagan_obj.start = checkpoint['step'] + 1
        sagan_obj.G = paddle.load(checkpoint['G']).to(sagan_obj.device)
        sagan_obj.G_optimizer = paddle.load(checkpoint['G_optimizer'])
        sagan_obj.D = paddle.load(checkpoint['D']).to(sagan_obj.device)
        sagan_obj.D_optimizer = paddle.load(checkpoint['D_optimizer'])
    # Else try for complete model, then try for state_dict
    else:
        try:
            sagan_obj.start = checkpoint['step'] + 1
            sagan_obj.G.load_state_dict(checkpoint['G_state_dict'])
            sagan_obj.G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
            sagan_obj.D.load_state_dict(checkpoint['D_state_dict'])
            sagan_obj.D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
        except:
            sagan_obj.start = checkpoint['step'] + 1
            sagan_obj.G = paddle.load(checkpoint['G']).to(sagan_obj.device)
            sagan_obj.G_optimizer = paddle.load(checkpoint['G_optimizer'])
            sagan_obj.D = paddle.load(checkpoint['D']).to(sagan_obj.device)
            sagan_obj.D_optimizer = paddle.load(checkpoint['D_optimizer'])

def load_test_pretrained_model(sagan_obj):
    print("Loading pretrained_model", sagan_obj.config.pretrained_model, "...")
    # Check for path
    assert os.path.exists(sagan_obj.config.pretrained_model), "Path of .pdparams pretrained_model doesn't exist! Given: " + sagan_obj.config.pretrained_model
    checkpoint = paddle.load(sagan_obj.config.pretrained_model)
    # If we know it is a state_dict (instead of complete model)
    if sagan_obj.config.state_dict_or_model == 'state_dict':
        sagan_obj.G.load_dict(checkpoint['G_state_dict'])
        sagan_obj.D.load_dict(checkpoint['D_state_dict'])

def make_test_dataloader(batch_size, dataset_type, data_path, shuffle=True, drop_last=True, dataloader_args={},
                    resize=True, imsize=128, centercrop=False, centercrop_size=128, totensor=True,
                    normalize=True, norm_mean=(0.5, 0.5, 0.5), norm_std=(0.5, 0.5, 0.5), num_workers=4):
    # Make transform
    transform = make_transform(resize=resize, imsize=imsize,
                               centercrop=centercrop, centercrop_size=centercrop_size,
                               totensor=totensor, normalize=normalize, norm_mean=norm_mean, norm_std=norm_std)
    # Make dataset
    if dataset_type in ['folder', 'imagenet', 'lfw']:
        # folder dataset
        assert os.path.exists(data_path), "data_path does not exist! Given: " + data_path
        dataset = dset.ImageFolder(root=data_path, transform=transform)
        print('#training images = %d' % len(dataset))
    elif dataset_type == 'lsun':
        assert os.path.exists(data_path), "data_path does not exist! Given: " + data_path
        dataset = dset.LSUN(root=data_path, classes=['bedroom_train'], transform=transform)
    elif dataset_type == 'cifar10':
        if not os.path.exists(data_path):
            print("data_path does not exist! Given: {}\nDownloading CIFAR10 dataset...".format(data_path))
        dataset = dset.CIFAR10(root=data_path, download=True, transform=transform)
    elif dataset_type == 'fake':
        dataset = dset.FakeData(image_size=(3, centercrop_size, centercrop_size), transform=transforms.ToTensor())
    assert dataset
    num_of_classes = len(dataset.classes)
    print("Data found!  # of images =", len(dataset), ", # of classes =", num_of_classes, ", classes:", dataset.classes)
    # Make dataloader from dataset
    dataloader = paddle.io.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **dataloader_args)
    return dataloader, num_of_classes


