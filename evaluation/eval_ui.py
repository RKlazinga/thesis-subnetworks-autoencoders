import math
import os

import torch
import torchvision
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import *
from matplotlib import pyplot as plt
from torch.nn.modules.batchnorm import _BatchNorm

from datasets.dataset_options import DatasetOption
from datasets.synthetic.flat_generators import random_sine_gaussian
from evaluation.analyse_latent_weights import analyse_at_epoch
from models.conv_ae import ConvAE
from settings.global_settings import ds
from settings.train_settings import NETWORK
from utils.file import change_working_dir
from utils.get_run_id import last_run


class InferenceWorker(QThread):
    output = pyqtSignal(Image.Image)

    def __init__(self):
        super().__init__()
        self.to_img = torchvision.transforms.ToPILImage()
        self.tup = None

    def set_tup(self, tup):
        self.tup = tup

    def run(self):
        print(self.tup)
        with torch.no_grad():
            tensor = torch.tensor([self.tup])
        decoded_tensor = model.decoder(tensor)[0]

        if ds == DatasetOption.SYNTHETIC_FLAT:
            plt.clf()
            plt.plot(range(16), decoded_tensor.tolist())
            plt.savefig("tmp.png")
            decoded_im = Image.open("tmp.png")
        else:
            decoded_tensor[decoded_tensor < 0] = 0
            decoded_im = self.to_img(decoded_tensor)
        self.output.emit(decoded_im)


class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.inference_worker = InferenceWorker()
        self.inference_worker.output.connect(self.update_img)

        self.img_label = QLabel()
        self.img_label.setFixedWidth(800)
        self.img_label.setBaseSize(800, 500)
        self.layout.addWidget(self.img_label, 0, 0, 1, len(latent_weight))

        self.sliders = []
        self.labels = []
        self.weights = []
        for i in range(len(latent_weight)):
            slider = QSlider()
            slider.setFixedHeight(100)
            slider.setMaximum(100)
            slider.setMinimum(0)
            slider.setSingleStep(1)
            slider.setValue(0)
            slider.valueChanged.connect(self.new_inference)
            self.layout.addWidget(slider, 1, i)
            weight = round(math.log10(max(latent_weight[i], 1e-6)), 3)
            self.weights.append(weight)
            self.labels.append(QLabel(f"Value: {0}\nWeight: {weight}"))
            self.layout.addWidget(self.labels[-1], 2, i)
            self.sliders.append(slider)

            if weight < -3:
                slider.setDisabled(True)
        self.show()

    def new_inference(self):
        new_value_vec = [float(s.value())/50 for s in self.sliders]
        for s, l, w in zip(self.sliders, self.labels, self.weights):
            l.setText(f"Value: {s.value()/50}\nWeight: {w}")
        self.inference_worker.set_tup(new_value_vec)
        self.inference_worker.start()

    def update_img(self, img: Image.Image):
        self.img_label.setPixmap(img.toqpixmap().scaledToWidth(800))


if __name__ == '__main__':
    change_working_dir()
    _run_id = last_run()
    # _run_id = "threevar-[8, 2, 1]-1b95d97e2"
    epoch = 20

    model = NETWORK.init_from_checkpoint(_run_id, None, None, param_epoch=epoch)
    model.eval()

    # get an example encoding
    # print(model.encoder(random_sine_gaussian(std=0.01, num_variables=3).view((1, 16))))

    # disable running stats since we will be inputting weird values
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.track_running_stats = False

    latent_weight, latent_bias, latent_mean, latent_var = analyse_at_epoch(_run_id, epoch)

    app = QApplication([])
    ui = MainUI()
    app.exec_()
    if os.path.isfile("tmp.png"):
        os.remove("tmp.png")
