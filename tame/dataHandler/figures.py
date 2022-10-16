import os

import numpy as np
from PIL import Image
from PyPDF2 import PdfFileWriter, PdfFileReader
import io

from PyPDF2.pdf import PageObject
from reportlab.pdfgen import canvas

from common import determine_grid, open_figure, save_figure

dirs = "../data/images/"


def read_pdf(file: str) -> PageObject:
    return PdfFileReader(open(file, "rb")).getPage(0)


def save_pdf(figure, file):
    output = PdfFileWriter()
    output.addPage(figure)
    outputStream = open(os.path.splitext(file)[0] + "_comb.pdf", "wb")
    output.write(outputStream)
    outputStream.close()


def create_subscript(subs: str, w: float, h: float) -> PageObject:
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(w, h))
    can.setFont("Helvetica", 15)
    can.drawCentredString(w // 2, 4, f"({subs})")
    can.save()
    packet.seek(0)
    new_pdf = PdfFileReader(packet)
    return new_pdf.getPage(0)


def get_size(page: PageObject) -> (float, float):
    w = float(page.mediaBox.getWidth())
    h = float(page.mediaBox.getHeight())
    return w, h


def add_subscript(file: str, subs: str):
    figure = read_pdf(file)
    w, h = get_size(figure)
    subscript = create_subscript(subs, w, h)
    figure.mergePage(subscript)
    return figure


files_list = {
    "high_conf": {
        "figures": {
            "asim/high_conf_class_0_6": "a",
            "asim/high_conf_class_1_6": "b",
            "asim/high_conf_class_2_6": "c",
            "asim/high_conf_class_3_6": "d",
            "asim/high_conf_class_4_6": "e",
            "asim/high_conf_class_5_6": "f"
        },
        "margins": {
            "v": 10,
            "h": 20
        }
    },
    "low_conf": {
        "figures": {
            "asim/low_conf_class_0_6": "a",
            "asim/low_conf_class_1_6": "b",
            "asim/low_conf_class_2_6": "c",
            "asim/low_conf_class_3_6": "d",
            "asim/low_conf_class_4_6": "e",
            "asim/low_conf_class_5_6": "f"
        },
        "margins": {
            "v": 10,
            "h": 20
        }
    },
    "segmented": {
        "figures": {
            "asim/removed/nya4_20101207_233450_5577_cal": "a",
            "asim/segmented/nya4_20101207_233450_5577_cal": "b",
            "asim/removed/nya4_20110105_172023_5577_cal": "c",
            "asim/segmented/nya4_20110105_172023_5577_cal": "d",
            "asim/removed/nya4_20110108_234023_5577_cal": "e",
            "asim/segmented/nya4_20110108_234023_5577_cal": "f"
        },
        "margins": {
            "v": 10,
            "h": 15,
            "subscr": 20
        }
    },
    "cloud": {
        "figures": {
            "asim/acc_cloud_data": "a",
            "asim/roc_cloud_data": "b"
        },
        "margins": {
            "subscr": 20
        }
    },
    "magn": {
        "figures": {
            "magn_train_dist": "a",
            "magn/peaks_all_deg_1_H_preds": "b",
            "magn/all_all_deg_1_H_scatter": "c",
            "magn/peaks_all_deg_1_H_scatter": "d"
        },
        "scales": [1.645, 1, 1, 1]
    },
    "hyperparameters": {
        "figures": {
            "RBFSVM_test2means": "a",
            "RBFSVM_test6means": "b",
            "RBFSVM_train2means": "c",
            "RBFSVM_train6means": "d"
        },
        "margins": {
            "subscr": 30
        }
    }
}

for files in files_list:
    print(files)
    figures = []
    subscripts = []
    pages = []
    widths = []
    heights = []
    dat = files_list.get(files)
    if "margins" in dat.keys():
        vertical_margin = dat.get("margins").get("v", 10)
        horizontal_margin = dat.get("margins").get("h", 20)
        subscr_margin = dat.get("margins").get("subscr", 10)
    else:
        vertical_margin = 20
        horizontal_margin = 20
        subscr_margin = 10
    if "scales" in dat.keys():
        scales = dat.get("scales")
    else:
        scales = [1]
    for f, i in zip(dat.get("figures"), range(len(dat.get("figures")))):
        if len(scales) > 1:
            scale = scales[i]
        else:
            scale = scales[0]
        figure = read_pdf(dirs + f + ".pdf")
        w, h = get_size(figure)
        page = PageObject.createBlankPage(None, width=w*scale, height=h*scale + subscr_margin)
        w, h = get_size(page)
        subscript = create_subscript(dat.get("figures").get(f), w, h)
        page.mergeScaledTranslatedPage(figure, scale, 0, subscr_margin)
        page.mergeScaledTranslatedPage(subscript, 1, 0, 0)
        widths.append(w)
        heights.append(h)
        figures.append(figure)
        subscripts.append(subscript)
        pages.append(page)
    if len(widths) == 2:
        width = 2
        height = 1
    else:
        grid = determine_grid(len(widths))
        width = grid[0]
        height = grid[1]
    w = np.max(widths) * width + horizontal_margin * (width - 1)
    h = np.max(heights) * height + vertical_margin * (height - 1)
    newPage = PageObject.createBlankPage(None, width=w, height=h)
    for i in range(len(figures)):
        page = pages[i]
        y, x = divmod(i, width)
        locx = x * w / width
        locy = h - (y + 1) / height * (h - vertical_margin * (height - 1)) - vertical_margin * y
        newPage.mergeScaledTranslatedPage(page, 1, locx, locy)
    save_pdf(newPage, files + ".pdf")

oath_images = ["00868.png", "00869.png", "02052.png", "02053.png", "03955.png", "03956.png"]
oath_path = "../data/oath_data/images/cropped_scaled/"

fig, main_ax = open_figure(constrained_layout=True)
main_ax.axis("off")
ax_ar = fig.subplots(3, 2)
ax_ar = ax_ar.ravel()
for i in range(len(oath_images)):
    ax = ax_ar[i]
    im_loc = oath_path + oath_images[i]
    im = np.array(Image.open(im_loc))
    ax.imshow(im)
    ax.axis("off")
fig.show()
save_figure(fig, "oath_images", (40, 60))
