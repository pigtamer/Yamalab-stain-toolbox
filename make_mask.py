import numpy as np
import shutil as shu
from numpy import *
import cv2 as cv
import os
import glob
import matplotlib.pyplot as plt
from skimage.color import rgb2hed, hed2rgb, separate_stains, combine_stains
from skimage.exposure import rescale_intensity
from matplotlib.colors import LinearSegmentedColormap
from skimage import data
from mpl_toolkits.mplot3d.axes3d import Axes3D  # --- For 3D plot
from skimage.exposure import rescale_intensity
from cv2 import ximgproc
from pathlib import Path

size = 2048
zoom = "1.0"

basepath = "/home/cunyuan/DATA/Kimura"
basepath = "/home/cunyuan/4tb/Kimura/DATA"
dpl = {"chips": basepath+ "/TILES_(%d, %d)/HE/*/*/*/" % (size, size),
       "ihcs": basepath + "/TILES_(%d, %d)/IHC/*/*/*/" % (size, size)}

"""
# ! For testing on macbook
basepath = "/Users/cunyuan/DATA/3768001"

dpl = {
    "chips": basepath + "/TILES_(%d, %d)/HE/*/" % (size, size),
    "ihcs": basepath + "/TILES_(%d, %d)/IHC/*" % (size, size),
}
"""
print(dpl["chips"])


def ig_f(dir, files):
    return [
        f
        for f in files
        if (
            os.path.isfile(os.path.join(dir, f))
            or  ("only" in os.path.join(dir, f)) # 如果之前运行了create_links，在拷贝目录树的时候会把tumor_only作为文件夹拎过来，因此添加这一规则
            and (not ("txt" in os.path.join(dir, f)))
        )
    ]


# shu.copytree(basepath+ "/TILES_(%d, %d)/HE"% (size, size), basepath+ "/TILES_(%d, %d)/DAB"% (size, size), ignore=ig_f)
# shu.copytree(basepath+ "/TILES_(%d, %d)/DAB"% (size, size), basepath+ "/TILES_(%d, %d)/Mask"% (size, size), ignore=ig_f)

# %%
H_DAB = array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])

H_he = H_DAB.copy()
H_he[2, :] = np.cross(H_DAB[0, :], H_DAB[1, :])

H_ki67 = H_DAB.copy()
H_ki67[1, :] = np.cross(H_DAB[0, :], H_DAB[2, :])

cmap_hema = LinearSegmentedColormap.from_list("mycmap", ["white", "navy"])
cmap_eosin = LinearSegmentedColormap.from_list("mycmap", ["white", "darkviolet"])
cmap_dab = LinearSegmentedColormap.from_list("mycmap", ["white", "saddlebrown"])
print("Trans. H.E", H_he)
print("Trans. Ki67", H_ki67)

H = H_ki67


"""
* ! Remember to set H as H_HE when creating labels from HE images!!!
"""


# %%


def reject_outliers(data, m=3):
    """reject_outliers　注意

    该函数会拒绝三倍标准差之外的数值将其替换成平均，非常危险！

    Args:
        data ([type]): [description]
        m (int, optional): [description]. Defaults to 3.

    Returns:
        [type]: [description]
    """
    dt = data.copy()
    dt[(abs(data - np.mean(data)) > m * np.std(data))] = np.mean(data)
    return dt


def in_range(d):
    return (0, np.max(cv.GaussianBlur(d.copy(), (3, 3), 0)))


def flip8(imname):
    imSrc = cv.imread(imname)
    for kr in range(4):
        for pr in range(2):
            if kr + pr == 0:
                continue
            im = imSrc
            for k in range(kr):
                im = cv.rotate(im, cv.ROTATE_90_CLOCKWISE)
            if pr != 0:
                im = cv.flip(im, flipCode=1)
            # plt.imshow(im)
            # plt.title("r%dp%d"%(kr*90, pr))
            # plt.show()
            cv.imwrite(imname[:-4] + "[r%dp%d].tif" % (kr * 90, pr), im)
    return 0


def norm_by_row(M):
    for k in range(M.shape[1]):
        M[k, :] /= np.sqrt(np.sum(np.power(M[k, :], 2)))
    return M


def showbychan(im_ihc):
    for k in range(3):
        plt.figure()
        plt.imshow(im_ihc[:, :, k], cmap="gray")


def rgbdeconv(rgb, conv_matrix, C=0):
    rgb = rgb.copy().astype(float)
    rgb += C
    stains = np.reshape(-np.log10(rgb), (-1, 3)) @ conv_matrix
    return np.reshape(stains, rgb.shape)


def hecconv(stains, conv_matrix, C=0):
    #     from skimage.exposure import rescale_intensity
    stains = stains.astype(float)
    logrgb2 = -np.reshape(stains, (-1, 3)) @ conv_matrix
    rgb2 = np.power(10, logrgb2)
    return np.reshape(rgb2 - C, stains.shape)


def surf(matIn, name="fig", div=(50, 50), SIZE=(8, 6)):
    x = np.arange(0, matIn.shape[0])
    y = np.arange(0, matIn.shape[1])
    x, y = np.meshgrid(y, x)
    fig = plt.figure(figsize=SIZE)
    ax = Axes3D(fig)
    ax.plot_surface(x, y, matIn, rstride=div[0], cstride=div[1], cmap="jet")
    plt.title(name)
    plt.show()


def surf2(mat1, mat2, name=["1", "2"], div=(50, 50), SIZE=(8, 4)):
    x1 = np.arange(0, mat1.shape[0])
    y1 = np.arange(0, mat1.shape[1])
    x1, y1 = np.meshgrid(y1, x1)

    x2 = np.arange(0, mat2.shape[0])
    y2 = np.arange(0, mat2.shape[1])
    x2, y2 = np.meshgrid(y2, x2)

    fig = plt.figure(figsize=SIZE)
    # pax1 = plt.subplot(121)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(x1, y1, mat1, rstride=div[0], cstride=div[1], cmap="jet")
    plt.title(name[0])
    plt.tight_layout()
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot_surface(x2, y2, mat2, rstride=div[0], cstride=div[1], cmap="jet")
    plt.title(name[1])
    plt.tight_layout()
    plt.show()

def surf4(im1, im2, mat1, mat2, name=["1", "2"], div=(50, 50), SIZE=(10,10)):
    x1 = np.arange(0, mat1.shape[0])
    y1 = np.arange(0, mat1.shape[1])
    x1, y1 = np.meshgrid(y1, x1)

    x2 = np.arange(0, mat2.shape[0])
    y2 = np.arange(0, mat2.shape[1])
    x2, y2 = np.meshgrid(y2, x2)

    fig = plt.figure(figsize=SIZE)
    # pax1 = plt.subplot(121)
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(im1, cmap = "gray");plt.axis('off')
    plt.title(name[0])
    plt.tight_layout()
    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(im2, cmap = "gray");plt.axis('off')
    plt.title(name[1])
    plt.tight_layout()
    ax = fig.add_subplot(2, 2, 3, projection="3d")
    ax.plot_surface(x1, y1, mat1, rstride=div[0], cstride=div[1], cmap="jet")
    # plt.title(name[0])
    plt.tight_layout()
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    ax.plot_surface(x2, y2, mat2, rstride=div[0], cstride=div[1], cmap="jet")
    # plt.title(name[1])
    plt.tight_layout()
    plt.show()
# folds = ["%03d"%(f+1) for f in range(10)]
# print(folds)
Hinv = linalg.inv(norm_by_row(H))
def genIHCMask(im_ki67, Hinv, t = 0.2, t0=0, t1 = 255, geps = 1000,grad = 10, MOP_SIZE=5, MCL_SIZE=5):
    img = (im_ki67*255).astype(np.uint8) / 255.
    img[img == 0] = 1e-6
    im_sepa_ki67 = abs(rgbdeconv(img, Hinv))

    h = im_sepa_ki67[:, :, 0]
    e_r = im_sepa_ki67[:, :, 1]
    d = im_sepa_ki67[:, :, 2]

    # fig = plt.figure(figsize=(10,10));
    # plt.subplot(221);plt.imshow(img);plt.title("Input")
    # plt.subplot(222);plt.imshow(rescale_intensity(h, in_range=in_range(h)), cmap=cmap_hema);plt.title("Hema.")
    # plt.subplot(223);plt.imshow(rescale_intensity(e_r, in_range=in_range(e_r)), cmap=cmap_eosin);plt.title("Residual (Eosin)")
    # plt.subplot(224);plt.imshow(rescale_intensity(d, in_range=in_range(d)), cmap=cmap_dab);plt.title("DAB")
    # fig.tight_layout()

    h1= h
    d1=d

    d2 = (d1 * 255).astype(uint8)
    guide = d2

    guidedDAB = ximgproc.guidedFilter(
        guide=guide, src=d2, radius=grad, eps=geps, dDepth=-1
    )
    gd = d1

    guidedDAB = guidedDAB * (guidedDAB > 255 * t)
    _, dmask = cv.threshold(
        guidedDAB, t0, t1, cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    guidedDAB = ((dmask) > 0).astype(uint8)

    guidedDAB = cv.morphologyEx(
        guidedDAB, op=cv.MORPH_OPEN, kernel=np.ones((MOP_SIZE, MOP_SIZE), uint8)
    )
    guidedDAB = cv.morphologyEx(
        guidedDAB,
        op=cv.MORPH_CLOSE,
        kernel=np.ones((MCL_SIZE, MCL_SIZE), uint8),
    )

    # bs = Path(imname).stem
    # mask_level_dir = str(Path(imname)).replace(".tif", ".png")  # two levels up

    if len(np.unique(gd.reshape(-1))) < 5:  # quantitize noise
        gd *= 0
        guidedDAB *= 0

    # plt.imsave(mask_level_dir.replace("IHC", "DAB").replace("_DAB", "_HE"), gd, cmap="gray")
    # plt.imsave(mask_level_dir.replace("IHC", "Mask").replace("_Mask", "_HE"), guidedDAB, cmap="gray")

    # plt.figure(figsize=(10, 6))
    return gd, guidedDAB
