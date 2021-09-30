import utils
from sagan_models import Generator, Discriminator
import paddle
import numpy as np
import random
from paddle.vision.transforms import functional as F
import time
import datetime
from PIL import Image
import os

class Tester(object):

    def __init__(self, config):

        # Config
        self.config = config


        # Create directories if not exist
        utils.make_folder(self.config.test_path)


        # Make dataloader
        self.dataloader, self.num_of_classes = utils.make_test_dataloader(batch_size=self.config.batch_size_test,
                                                                          dataset_type=self.config.dataset,
                                                                          data_path=self.config.test_data_path,
                                                                          shuffle=True,
                                                                          drop_last=self.config.drop_last,
                                                                          dataloader_args=self.config.dataloader_args,
                                                                          resize=self.config.resize,
                                                                          imsize=self.config.imsize,
                                                                          centercrop=self.config.centercrop,
                                                                          centercrop_size=self.config.centercrop_size,
                                                                          normalize=self.config.normalize,
                                                                          num_workers=self.config.num_workers,
                                                                          )


        # Data iterator
        self.data_iter = iter(self.dataloader)
        # Build G and D
        self.build_models()

    def build_models(self):
        self.G = Generator(self.config.z_dim, self.config.g_conv_dim, self.num_of_classes)
        self.D = Discriminator(self.config.d_conv_dim, self.num_of_classes)

        # Start with pretrained model (if it exists)
        if self.config.pretrained_model != '':
            utils.load_test_pretrained_model(self)

    def test(self):
        # For BatchNorm
        self.G.eval()

        # Init
        start_time = time.time()


        # Start testing
        for i in range(50000):
            print("id =", i)

            # Get real images & real labels (only need real labels)
            real_images, real_labels = self.get_real_samples()
            print("real_labels:", real_labels)

            # Create random noise
            z = paddle.randn([self.config.batch_size_test, self.config.z_dim])

            # Generate fake images for same real labels
            fake_images = self.G(z, real_labels)

            # Print out log info
            curr_time = time.time()
            curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
            elapsed = str(datetime.timedelta(seconds=(curr_time - start_time)))
            log = ("[{}] : Elapsed [{}]\n".
                   format(curr_time_str, elapsed))
            print('\n' + log)

            # Sample images
            print("Saving image samples..")
            self.G.eval()
            self.save_sample(fake_images, os.path.join(self.config.test_path + '/' + 'fake_{:05d}.png'.format(i)))

    def get_real_samples(self):
        try:
            real_images, real_labels = next(self.data_iter)
        except:
            self.data_iter = iter(self.dataloader)
            real_images, real_labels = next(self.data_iter)

        return real_images, real_labels

    def save_sample(self, images, path):
        # images = paddle.nn.functional.pad(images, pad = [2, 2, 2, 2])

        b, c, h, w = images.shape
        results = np.zeros((1, 3, 1 * h, 1 * w))
        count = 0
        for i in range(1):
            for j in range(1):
                results[:, :, i * h:(i + 1) * h, j * w:(j + 1) * w] = images[count].unsqueeze(0)
                count += 1
        results = 255 * (results + 1) / 2
        result = np.array(results[0].transpose(1, 2, 0), dtype=np.uint8)

        save_result = Image.fromarray(result)
        save_result.save(path)
