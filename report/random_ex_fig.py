from PIL import ImageFont, Image, ImageDraw
from torchvision.transforms import ToPILImage

from datasets.synthetic.im_generators import sine_2d
from settings import Settings
from utils.file import change_working_dir


if __name__ == '__main__':
    height = 8
    dims = 5
    rim = 8
    spacing = (2, 12)
    size = 28
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size=14)
    textsize = font.getsize("D=1xx")
    bg = Image.new("L", (height*28 + (height-1) * spacing[0] + rim + textsize[0], dims*size + 2 * rim + (dims-1) * spacing[1]), color=0)
    draw = ImageDraw.Draw(bg)
    Settings.NORMAL_STD_DEV = 0.01
    for d in range(dims):
        Settings.NUM_VARIABLES = d
        for i in range(height):
            im = ToPILImage()(sine_2d())
            bg.paste(im, (i*(size + spacing[0]) + textsize[0], rim + d*(size + spacing[1])))
        text = f"D={d}"
        w = font.getsize(text)[0]
        draw.text(((textsize[0] - w) // 2, rim + d*(size + spacing[1]) + textsize[1] // 2), text=text, font=font, fill=255)
    change_working_dir()
    bg.show()
    bg.save("figures/synth_2d_ex.png")
