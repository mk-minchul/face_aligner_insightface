import numpy as np
UINT8_MAX = 2. ** 8. - 1.
UINT16_MAX = 2. ** 16. - 1.
import cv2
import matplotlib.pyplot as plt
import io


def convert_image_type(image, dtype=np.float32):
    if image.dtype == np.uint8:

        if dtype == np.float32:
            image = image.astype(np.float32)
            image /= UINT8_MAX
            return image
        elif dtype == np.uint8:
            return image
        else:
            raise TypeError('numpy.float32 or numpy.uint8 supported as a target dtype')

    elif image.dtype == np.uint16:

        if dtype == np.float32:
            image = image.astype(np.float32)
            image /= UINT16_MAX
            return image
        elif dtype == np.uint8:
            image = image.astype(np.float32)
            image *= UINT8_MAX / UINT16_MAX
            image = image.astype(np.uint8)
            return image
        elif dtype == np.uint16:
            return image
        else:
            raise TypeError('numpy.float32 or numpy.uint8 or numpy.uint16 supported as a target dtype')

    elif image.dtype == np.float32:
        assert image.max() <= 1
        assert image.min() >= 0
        if dtype == np.float32:
            return image
        elif dtype == np.uint8:
            image *= UINT8_MAX
            image = image.astype(np.uint8)
            return image
        elif dtype == np.uint16:
            image *= UINT16_MAX
            image = image.astype(np.uint16)
            return image

    else:
        raise TypeError('numpy.uint8 or numpy.uint16 or np.float32 supported as an input dtype')


def stack_images(images, num_cols, num_rows, pershape=(112,112)):
    stack = []
    for rownum in range(num_rows):
        row = []
        for colnum in range(num_cols):
            idx = rownum * num_cols + colnum
            if idx > len(images)-1:
                img_resized = np.zeros((pershape[0], pershape[1], 3))
            else:
                img_resized = cv2.resize(images[idx], dsize=pershape)
            row.append(img_resized)
        row = np.concatenate(row, axis=1)
        stack.append(row)
    stack = np.concatenate(stack, axis=0)
    return stack

def prepare_text_img(text, height=300, width=30, fontsize=16, textcolor='C1'):
    text_kwargs = dict(ha='center', va='center', fontsize=fontsize, color=textcolor)
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(width*px, height*px))
    plt.text(0.5, 0.5, text, **text_kwargs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    array = get_img_from_fig(fig)
    plt.clf()
    return array

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img