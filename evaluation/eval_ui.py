import torch
import torchvision
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import *

from models.conv_ae import ConvAE


class InferenceWorker(QThread):
    output = pyqtSignal(Image.Image)

    def __init__(self):
        super().__init__()
        self.to_img = torchvision.transforms.ToPILImage()
        self.tup = None

    def set_tup(self, tup):
        self.tup = tup

    def run(self):
        tensor = torch.tensor([self.tup])
        decoded_tensor = model.decoder(tensor)[0]
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
        self.img_label.setFixedWidth(200)
        self.img_label.setFixedHeight(200)
        self.layout.addWidget(self.img_label, 0, 0, 1, 4)

        self.sliders = []
        for i in range(6):
            slider = QSlider()
            slider.setMaximum(50)
            slider.setMinimum(0)
            slider.setValue(1)
            slider.valueChanged.connect(self.new_inference)
            self.layout.addWidget(slider, 1, i)
            self.sliders.append(slider)

        self.show()

    def new_inference(self):
        new_value_vec = [float(s.value())/10 for s in self.sliders]
        self.inference_worker.set_tup(new_value_vec)
        self.inference_worker.start()

    def update_img(self, img: Image.Image):
        self.img_label.setPixmap(img.toqpixmap().scaledToWidth(200))


if __name__ == '__main__':
    model = ConvAE(6, 4, 6)
    model.load_state_dict(torch.load("network.pth"))

    app = QApplication([])
    ui = MainUI()
    app.exec_()
