#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import matplotlib.pyplot as plt
from numpy import *
from skimage.color import rgb2hed, hed2rgb,separate_stains, combine_stains
from skimage.exposure import rescale_intensity
from matplotlib.colors import LinearSegmentedColormap
from skimage import data
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D #--- For 3D plot
#https://github.com/scikit-image/scikit-image/blob/0e8cff47a533e240508e2485032dac78220ac6c9/skimage/color/colorconv.py#L1375


# In[2]:


def norm_by_row(M):
    for k in range(M.shape[1]):
        M[k,:] /= np.sqrt(np.sum(np.power(M[k,:],2)))
    return M

def showbychan(im_ihc):
    for k in range(3):
        plt.figure()
        plt.imshow(im_ihc[:, :, k], cmap="gray")

def rgbdeconv(rgb, conv_matrix, C=0):
    rgb = rgb.copy().astype(float)
    rgb += C
    print(rgb.shape)
    print(np.reshape(-np.log10(rgb), (-1, 3)).shape)
    stains = np.reshape(-np.log10(rgb), (-1, 3)) @ conv_matrix
    return np.reshape(stains, rgb.shape)

def hecconv(stains, conv_matrix, C=0):
#     from skimage.exposure import rescale_intensity
    stains = stains.astype(float)
    logrgb2 = -np.reshape(stains, (-1, 3)) @ conv_matrix
    rgb2 = np.power(10, logrgb2)
    return np.reshape(rgb2 - C, stains.shape)

def surf(matIn, name="fig", div = (50, 50), SIZE = (8, 6)):
    x = np.arange(0, matIn.shape[0])
    y = np.arange(0, matIn.shape[1])
    x, y = np.meshgrid(y, x)
    fig = plt.figure(figsize = SIZE)
    ax = Axes3D(fig)
    ax.plot_surface(x, y, matIn, rstride=div[0], cstride=div[1], cmap='jet')
    plt.title(name)
    plt.show()

def in_range(d):
    return (0, np.max(cv.GaussianBlur(d.copy(), (3,3), 0)))


# In[3]:


H_DAB = array([
    [0.65,0.70,0.29],
    [0.07, 0.99, 0.11],
    [0.27,0.57,0.78]
])

H_Mou = H_DAB.copy()
H_Mou[2,:] = np.cross(H_DAB[0,:], H_DAB[1,:])

H_ki67 = H_DAB.copy()
H_ki67[1,:] = np.cross(H_DAB[0,:], H_DAB[2,:])

cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['white','darkviolet'])
cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white','saddlebrown'])
cmap_res = LinearSegmentedColormap.from_list('mycmap', ['white','green'])
print("Trans. H.E", H_Mou)
print("Trans. Ki67", H_ki67)


# In[4]:


im_mask = cv.cvtColor(cv.imread("/Users/cunyuan/DATA/ji1024_orig/4d/val1024/masks/13hepatches_mask.tif",
                                cv.CV_32F), cv.COLOR_BGR2GRAY)

im_hex = cv.imread("/Users/cunyuan/DATA/Kimura/EMca別症例_WSIとLI算出領域/LI算出領域/14-3768/my2048/chips/01_14-3768_Ki67_HE (1, x=35300, y=34306, w=2049, h=2049).tif",
                   cv.CV_32F)
im_ki67 = cv.imread("/Users/cunyuan/DATA/Kimura/EMca別症例_WSIとLI算出領域/LI算出領域/14-3768/my2048/dab/01_14-3768_Ki67_IHC (1, x=31201, y=34255, w=2048, h=2048).tif",
                   cv.CV_32F)
im_res = cv.imread("/Users/cunyuan/DATA/Kimura/EMca別症例_WSIとLI算出領域/LI算出領域/14-3768/my2048/result_1.png",
                   cv.CV_32F)

im_hex = cv.cvtColor(im_hex, cv.COLOR_BGR2RGB)/255.
im_res = cv.cvtColor(im_res, cv.COLOR_BGR2RGB)/255.
im_ki67 = cv.cvtColor(im_ki67, cv.COLOR_BGR2RGB)/255.

plt.imshow(im_ki67)


# # Ki-67 Separation

# In[5]:


H = H_ki67
Hinv = linalg.inv(norm_by_row(H))
plt.figure();plt.imshow(im_ki67)


# In[6]:


img = np.float32(im_ki67)
# img[img==0] = 1E-6
im_sepa_ki67=abs(rgbdeconv(img, Hinv))

h = im_sepa_ki67[:,:,0];e_r = im_sepa_ki67[:,:,1];d = im_sepa_ki67[:,:,2];


fig = plt.figure(figsize=(10,10));
plt.subplot(221);plt.imshow(img);plt.title("Input");plt.axis('off')
plt.subplot(222);plt.imshow(h, cmap=cmap_hema);plt.title("Hema.");plt.axis('off')
plt.subplot(223);plt.imshow(e_r, cmap=cmap_res);plt.title("Residual");plt.axis('off')
plt.subplot(224);plt.imshow(d, cmap=cmap_dab);plt.title("DAB");plt.axis('off')
fig.tight_layout()

surf(h, "Hema.")
surf(e_r, "Residual (Other components)")
surf(d, "DAB")
d.max()
plt.imshow(rescale_intensity(d, in_range=in_range(d)), cmap="gray")
wh0 = h.sum(), e_r.sum(), d.sum()


# In[7]:


wh0


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
img = np.float32(im_res)
im_sepa_res=abs(rgbdeconv(img, Hinv))

h = im_sepa_res[:,:,0];e_r = im_sepa_res[:,:,1];d = im_sepa_res[:,:,2];


fig = plt.figure(figsize=(10,10));
plt.subplot(221);plt.imshow(img);plt.title("Input");plt.axis('off')
plt.subplot(222);plt.imshow(h, cmap=cmap_hema);plt.title("Hema.");plt.axis('off')
plt.subplot(223);plt.imshow(e_r,  cmap=cmap_res);plt.title("Residual");plt.axis('off')
plt.subplot(224);plt.imshow(d,  cmap=cmap_dab);plt.title("DAB");plt.axis('off')
fig.tight_layout()

surf(h, "Hema")
surf(e_r, "Residual (Eosin)")
surf(d, "DAB")


# In[9]:


wh1 = h.sum(), e_r.sum(), d.sum()
c01 = np.array(wh0)/np.array(wh1)
print(c01)


# In[10]:


zdh = np.dstack((h, e_r, d))

zdh = (zdh.reshape(-1,3)*c01.T).reshape(img.shape)
correct_zdh = hecconv(zdh, H)

correct_zdh = rescale_intensity(correct_zdh)

fig = plt.figure(figsize=(20,20))
axis = plt.subplot(1, 2, 1)
axis.imshow(correct_zdh)
axis.set_title("Pseudo Hematoxylin-DAB")
axis.axis('off')

axis = plt.subplot(1,2,2)
axis.imshow(im_ki67)
axis.set_title("True Hematoxylin-DAB")
axis.axis('off')
plt.tight_layout()
plt.show()


# In[ ]:





# In[11]:


surf(correct_zdh[:, :, 2])


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')

downsample_size=(256,256)
downsp_ki67 = cv.resize(im_sepa_ki67, downsample_size)
downsp_res = cv.resize(im_sepa_res, downsample_size)
# downsp_res = cv.resize(zdh, downsample_size)

# downsp_ki67 -= downsp_ki67.mean()
# downsp_res -= downsp_res.mean()

fig=plt.figure(figsize=(10,20))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("Hema")
ax.set_ylabel("Res")
ax.set_zlabel("DAB")

ax.scatter(downsp_ki67[:,:,0].ravel(),
           downsp_ki67[:,:,1].ravel(),
           downsp_ki67[:,:,2].ravel(),
          alpha=0.1)

ax.scatter(downsp_res[:,:,0].ravel(),
           downsp_res[:,:,1].ravel(),
           downsp_res[:,:,2].ravel(),
          alpha=0.1,
          marker='x')

plt.legend(["Std", "Output"])
plt.tight_layout()


# In[13]:


z = np.array([
    downsp_ki67[:,:,0].ravel(),
#     downsp_ki67[:,:,1].ravel(),
    downsp_ki67[:,:,2].ravel(),
    ])
eg, egv = np.linalg.eig(np.cov(z))
egv0 = egv.T[::-1]
eg0 = eg

z = np.array([downsp_res[:,:,0].ravel(),
#            downsp_res[:,:,1].ravel(),
           downsp_res[:,:,2].ravel(),])
eg, egv = np.linalg.eig(np.cov(z))
egv1 = egv.T[::-1]
eg1 = eg
# print(egv1, "\n", egv.T[::-1])
egv0, egv1


# In[14]:



# flag =0
# for k in [0,1]:
#     for j in [0,1]:
#         for i in [0,1]:
#             if k==1:
#                 egv0[0, :] = -egv0[0, :]
#             if j==1:
#                 egv0[1, :] = -egv0[1, :]
#             if i==1:
#                 egv0[2, :] = -egv0[2, :]
#             if np.sum(np.sum(egv0, axis=0)*[1,1,1]) >0.5:
#                 flag=1
#                 break
#         if flag==1: break
#     if flag==1: break

# for k in [0,1]:
#     for j in [0,1]:
#         for i in [0,1]:
#             if k==1:
#                 egv1[0, :] = -egv1[0, :]
#             if j==1:
#                 egv1[1, :] = -egv1[1, :]
#             if i==1:
#                 egv1[2, :] = -egv1[2, :]
#             if np.sum(np.sum(egv1, axis=0)*[1,1,1]) >0.5:
#                 flag=1
#                 break
#         if flag==1: break
#     if flag==1: break
# egv0,egv1


# In[15]:


fig=plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_xlabel("Hema")
ax.set_ylabel("Res")
# ax.set_zlabel("DAB")
ax.set_ylabel("DAB")


ax.scatter(downsp_ki67[:,:,0].ravel(),
#            downsp_ki67[:,:,1].ravel(),
           downsp_ki67[:,:,2].ravel(),
          alpha=0.1)

ax.scatter(downsp_res[:,:,0].ravel(),
#            downsp_res[:,:,1].ravel(),
           downsp_res[:,:,2].ravel(),
          alpha=0.1,
          marker='x')
plt.legend(["Std", "Output"])
egv00 = egv0/5
egv10 = egv1/5
basep = [0.,0.]
ax.arrow(downsp_ki67[:,:,0].mean(),
         downsp_ki67[:,:,2].mean(),
        (-egv00[0,0]),
        (-egv00[0,1]),color='r',head_width=0.01)

ax.arrow(downsp_ki67[:,:,0].mean(),downsp_ki67[:,:,2].mean(),
        (egv00[1,0]),
        (egv00[1,1]),color='g',head_width=0.01)


ax.arrow(downsp_res[:,:,0].mean(),
         downsp_res[:,:,2].mean(),
         (-egv10[0,0]),
         (-egv10[0,1]),color='g',head_width=0.01)

ax.arrow(downsp_res[:,:,0].mean(),
         downsp_res[:,:,2].mean(),
        (-egv10[1,0]),
        (-egv10[1,1]),color='r',head_width=0.01)

# plt.legend(["00", "01", "10", "11"])
# ax.arrow(0,0,egv00[1,0], egv00[1,1],color='b',head_width=0.01)
# ax.arrow(0,0,-egv10[0,0], -egv10[0,1],color='r',head_width=0.01)
# ax.arrow(0,0,-egv10[1,0], -egv10[1,1],color='r',head_width=0.01)

# ax.quiver(basep,basep,-egv10[:,0], -egv10[:,1],color='r')

# ax.quiver(0,0, 0,egv10[0,0], egv10[0,1], egv10[0,2],color='r')
# ax.quiver(0,0, 0,egv10[1,0], egv10[1,1], egv10[1,2],color='r')
# ax.quiver(0,0, 0,egv10[2,0], egv10[2,1], egv10[2,2],color='r')

plt.tight_layout()
plt.grid()


# In[16]:


x1, y1 = downsp_ki67[:,:,0].mean(),downsp_ki67[:,:,2].mean()
triangle1 = np.array([[x1, y1],
            [x1 - egv00[0,0], y1 - egv00[0,1]],
            [x1 + egv00[1,0], y1 +egv00[1,1]]]).astype(np.float32)

x2, y2 = downsp_res[:,:,0].mean(),downsp_res[:,:,2].mean()
triangle2 = np.array([[x2, y2],
            [x2 - egv10[1,0], y2 - egv10[1,1]],
            [x2 - egv10[0,0], y2 - egv10[0,1]],
]).astype(np.float32)

triangle1, triangle2


# In[17]:


np.sum(egv0, axis=0)


# In[18]:


eg0, egv0


# In[19]:


warp_mat = cv.getAffineTransform(triangle2, triangle1)


# In[20]:


tmp = np.hstack([triangle2, [[1],[1],[1]]])
tmp@warp_mat.T


# In[21]:


z = np.array([
    downsp_res[:,:,0].ravel(),
    downsp_res[:,:,2].ravel(),
    ones_like(downsp_ki67[:,:,0].ravel())
    ])
z = warp_mat@z
z.shape


# In[22]:


fig=plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_xlabel("Hema")
ax.set_ylabel("Eosin")
# ax.set_zlabel("DAB")
ax.set_ylabel("DAB")


ax.scatter(downsp_ki67[:,:,0].ravel(),
#            downsp_ki67[:,:,1].ravel(),
           downsp_ki67[:,:,2].ravel(),
          alpha=0.1)

ax.scatter(z[0,:],
#            z[:,:,1].ravel(),
           z[1,:],
          alpha=0.1,
          marker='x')
plt.legend(["Std", "Output"])
egv00 = egv0/5
egv10 = egv1/5
basep = [0.,0.]
ax.arrow(downsp_ki67[:,:,0].mean(),
         downsp_ki67[:,:,2].mean(),
(-egv00[0,0]),
(-egv00[0,1]),color='r',head_width=0.01)
print(downsp_ki67[:,:,0].mean(),
         downsp_ki67[:,:,2].mean(),
(egv00[0,0]),
(egv00[0,1]),
)
ax.arrow(downsp_ki67[:,:,0].mean(),downsp_ki67[:,:,2].mean(),
(egv00[1,0]),
(egv00[1,1]),color='g',head_width=0.01)

# tmp = np.hstack([-egv10 +[[z[0,:].mean(), z[1,:].mean()], [z[0,:].mean(), z[1,:].mean()]], [[1],[1]]])
# print(tmp)
# print(triangle2)
# tmp = tmp@warp_mat.T
# print(tmp)

eg, egv = np.linalg.eig(np.cov(z))
egv10 = egv.T[::-1]/5


ax.arrow(z[0,:].mean(),
         z[1,:].mean(),
         (-egv10[0,0]),
         (-egv10[0,1]),color='r',head_width=0.01)

ax.arrow(z[0,:].mean(),
         z[1,:].mean(),
         (-egv10[1,0]),
         (-egv10[1,1]),color='g',head_width=0.01)
# ax.arrow(0,0,egv00[1,0], egv00[1,1],color='b',head_width=0.01)
# ax.arrow(0,0,-egv10[0,0], -egv10[0,1],color='r',head_width=0.01)
# ax.arrow(0,0,-egv10[1,0], -egv10[1,1],color='r',head_width=0.01)

# ax.quiver(basep,basep,-egv10[:,0], -egv10[:,1],color='r')

# ax.quiver(0,0, 0,egv10[0,0], egv10[0,1], egv10[0,2],color='r')
# ax.quiver(0,0, 0,egv10[1,0], egv10[1,1], egv10[1,2],color='r')
# ax.quiver(0,0, 0,egv10[2,0], egv10[2,1], egv10[2,2],color='r')

plt.tight_layout()
plt.grid()


# In[23]:


egv10


# In[24]:


z.shape


# In[25]:


img = np.float32(im_res)
# img[img==0] = 1E-6
im_sepa_res=abs(rgbdeconv(img, Hinv))

h = z[0, :].reshape(downsample_size);e_r = downsp_ki67[:,:,1];d = z[1, :].reshape(downsample_size)

ress = hecconv(np.dstack([h, e_r, d]), H)

fig = plt.figure(figsize=(10,10));
plt.subplot(221);plt.imshow(ress);plt.title("Adjusted");plt.axis('off')
plt.subplot(222);plt.imshow(h, cmap=cmap_hema);plt.title("Hema.");plt.axis('off')
plt.subplot(223);plt.imshow(e_r,  cmap=cmap_res);plt.title("Residual");plt.axis('off')
plt.subplot(224);plt.imshow(d,  cmap=cmap_dab);plt.title("DAB");plt.axis('off')
fig.tight_layout()

fig = plt.figure(figsize=(10,20), dpi=300);
plt.subplot(131);plt.imshow(im_res);plt.title("Before");plt.axis('off')
plt.subplot(132);plt.imshow(ress);plt.title("After");plt.axis('off')
plt.subplot(133);plt.imshow(im_ki67);plt.title("Physical");plt.axis('off')
fig.tight_layout()


# In[ ]:





# In[26]:


get_ipython().run_line_magic('matplotlib', 'osx')

downsample_size=(64,64)
downsp_ki67 = cv.resize(im_ki67, downsample_size)
downsp_res = cv.resize(im_res, downsample_size)
# downsp_res = cv.resize(zdh, downsample_size)

# downsp_ki67 = np.log10(downsp_ki67+1)
# downsp_res = np.log10(downsp_res+1)

fig=plt.figure(figsize=(10,20))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("R")
ax.set_ylabel("G")
ax.set_zlabel("B")

ax.scatter(downsp_ki67[:,:,0].ravel(),
           downsp_ki67[:,:,1].ravel(),
           downsp_ki67[:,:,2].ravel(),
          alpha=0.1)

ax.scatter(downsp_res[:,:,0].ravel(),
           downsp_res[:,:,1].ravel(),
           downsp_res[:,:,2].ravel(),
          alpha=0.1,
          marker='x')

plt.legend(["Std", "Output"])
plt.tight_layout()


# In[27]:


z = np.array([
    downsp_ki67[:,:,0].ravel(),
    downsp_ki67[:,:,1].ravel(),
    downsp_ki67[:,:,2].ravel(),
    ])
eg, egv = np.linalg.eig(np.cov(z))
egv0 = egv.T[::-1]
eg0 = eg

z = np.array([downsp_res[:,:,0].ravel(),
           downsp_res[:,:,1].ravel(),
           downsp_res[:,:,2].ravel(),])
eg, egv = np.linalg.eig(np.cov(z))
egv1 = egv.T[::-1]
eg1 = eg
# print(egv1, "\n", egv.T[::-1])
egv0, egv1


# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig=plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,
#                      projection="3d"，
                    )
ax.set_xlabel("R")
# ax.set_ylabel("G")
ax.set_ylabel("B")
# ax.set_ylabel("DAB")


ax.scatter(downsp_ki67[:,:,0].ravel(),
#            downsp_ki67[:,:,1].ravel(),
           downsp_ki67[:,:,2].ravel(),
          alpha=0.1)

ax.scatter(downsp_res[:,:,0].ravel(),
#            downsp_res[:,:,1].ravel(),
           downsp_res[:,:,2].ravel(),
          alpha=0.1,
          marker='x')
plt.legend(["Std", "Output"])
egv00 = egv0/5
egv10 = egv1/5
basep = [0.,0.]
# ax.arrow(downsp_ki67[:,:,0].mean(),
#          downsp_ki67[:,:,1].mean(),
#          downsp_ki67[:,:,2].mean(),
#          (-egv00[0,0]),
#          (-egv00[0,1]),
#          (-egv00[0,2]),
#          color='b',head_width=0.01)

# ax.arrow(downsp_ki67[:,:,0].mean(),
#          downsp_ki67[:,:,1].mean(),
#          downsp_ki67[:,:,2].mean(),
#         (egv00[1,0]),
#         (egv00[1,1]),
#          (-egv00[1,2]),
#          color='b',head_width=0.01)


# ax.arrow(downsp_res[:,:,0].mean(),
#          downsp_res[:,:,1].mean(),
#          downsp_res[:,:,2].mean(),
#          (-egv10[0,0]),
#          (-egv10[0,1]),
#          (-egv10[0,2]),
#          color='r',head_width=0.01)

# ax.arrow(downsp_res[:,:,0].mean(),
#          downsp_res[:,:,1].mean(),
#          downsp_res[:,:,2].mean(),
#         (-egv10[1,0]),
#         (-egv10[1,1]),
#          (-egv10[1,2]),
#          color='r',head_width=0.01)

# plt.legend(["00", "01", "10", "11"])
# ax.arrow(0,0,egv00[1,0], egv00[1,1],color='b',head_width=0.01)
# ax.arrow(0,0,-egv10[0,0], -egv10[0,1],color='r',head_width=0.01)
# ax.arrow(0,0,-egv10[1,0], -egv10[1,1],color='r',head_width=0.01)

# ax.quiver(basep,basep,-egv10[:,0], -egv10[:,1],color='r')


# ax.quiver(downsp_ki67[:,:,0].mean(),
#          downsp_ki67[:,:,1].mean(),
#          downsp_ki67[:,:,2].mean(),egv00[0,0], egv00[0,1], egv00[0,2],color='r')
# ax.quiver(downsp_ki67[:,:,0].mean(),
#          downsp_ki67[:,:,1].mean(),
#          downsp_ki67[:,:,2].mean(),egv00[1,0], egv00[1,1], egv00[1,2],color='g')
# ax.quiver(downsp_ki67[:,:,0].mean(),
#          downsp_ki67[:,:,1].mean(),
#          downsp_ki67[:,:,2].mean(),egv00[2,0], egv00[2,1], egv00[2,2],color='b')

# ax.quiver(downsp_res[:,:,0].mean(),
#          downsp_res[:,:,1].mean(),
#          downsp_res[:,:,2].mean(),egv10[0,0], egv10[0,1], egv10[0,2],color='g')
# ax.quiver(downsp_res[:,:,0].mean(),
#          downsp_res[:,:,1].mean(),
#          downsp_res[:,:,2].mean(),-egv10[1,0], -egv10[1,1], -egv10[1,2],color='r')
# ax.quiver(downsp_res[:,:,0].mean(),
#          downsp_res[:,:,1].mean(),
#          downsp_res[:,:,2].mean(),egv10[2,0], egv10[2,1], egv10[2,2],color='b')

plt.tight_layout()
plt.grid()


# In[37]:


egv10[1,:] = -egv10[1,:]
egv10


# In[57]:


plane111 = np.array([[1,0,-1],[0,1,-1]])/sqrt(2)
zp = (im_ki67[:1000, :1000, :].reshape(-1,3)@plane111.T)
fig=plt.figure(figsize=(10,10))
plt.scatter(zp[:,0], zp[:,1], alpha=0.05, s=0.2)


# In[30]:


x1, y1,z1 = downsp_ki67[:,:,0].mean(),downsp_ki67[:,:,1].mean(),downsp_ki67[:,:,2].mean()
triangle1 = np.array([[x1, y1, z1],
                      [x1 + egv00[0,0], y1 + egv00[0,1], z1 + egv00[0,2]],
                      [x1 + egv00[1,0], y1 +egv00[1,1], z1 + egv00[1,2]],
                      [x1 + egv00[2,0], y1 + egv00[2,1], z1 + egv00[2,2]]
                     ]).astype(np.float32)

x2, y2,z2 = downsp_res[:,:,0].mean(),downsp_res[:,:,1].mean(),downsp_res[:,:,2].mean()
triangle2 = np.array([[x2, y2,z2],
            [x2 + egv10[1,0], y2 + egv10[1,1], z2 + egv10[1,2]],
            [x2 + egv10[0,0], y2 + egv10[0,1], z2 + egv10[0,2]],
            [x2 + egv10[2,0], y2 + egv10[2,1], z2 + egv10[2,2]],
]).astype(np.float32)

triangle1, triangle2


# In[31]:


warp_mat = cv.estimateAffine3D(triangle2, triangle1)


# In[32]:


# _,warp_mat,_ = warp_mat
warp_mat = warp_mat[1]
warp_mat


# In[33]:


z = np.array([
    downsp_res[:,:,0].ravel(),
    downsp_res[:,:,1].ravel(),
    downsp_res[:,:,2].ravel(),
    ones_like(downsp_ki67[:,:,0].ravel())
    ])
z = warp_mat@z
z


# In[34]:


get_ipython().run_line_magic('matplotlib', 'osx')

eg, egv = np.linalg.eig(np.cov(z))
egv1 = egv.T[::-1]
eg1 = eg

fig=plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("R")
ax.set_ylabel("G")
ax.set_zlabel("B")
# ax.set_ylabel("DAB")


ax.scatter(downsp_ki67[:,:,0].ravel(),
           downsp_ki67[:,:,1].ravel(),
           downsp_ki67[:,:,2].ravel(),
          alpha=0.1)

ax.scatter(z[0,:].ravel(),
           z[1,:].ravel(),
           z[2,:].ravel(),
          alpha=0.1,
          marker='x')
plt.legend(["Std", "Output"])
egv00 = egv0/5
egv10 = egv1/5
basep = [0.,0.]
# ax.arrow(downsp_ki67[:,:,0].mean(),
#          downsp_ki67[:,:,1].mean(),
#          downsp_ki67[:,:,2].mean(),
#          (-egv00[0,0]),
#          (-egv00[0,1]),
#          (-egv00[0,2]),
#          color='b',head_width=0.01)

# ax.arrow(downsp_ki67[:,:,0].mean(),
#          downsp_ki67[:,:,1].mean(),
#          downsp_ki67[:,:,2].mean(),
#         (egv00[1,0]),
#         (egv00[1,1]),
#          (-egv00[1,2]),
#          color='b',head_width=0.01)


# ax.arrow(z[0,:].mean(),
#          z[1,:].mean(),
#          z[2,:].mean(),
#          (-egv10[0,0]),
#          (-egv10[0,1]),
#          (-egv10[0,2]),
#          color='r',head_width=0.01)

# ax.arrow(z[0,:].mean(),
#          z[1,:].mean(),
#          z[2,:].mean(),
#         (-egv10[1,0]),
#         (-egv10[1,1]),
#          (-egv10[1,2]),
#          color='r',head_width=0.01)

# plt.legend(["00", "01", "10", "11"])
# ax.arrow(0,0,egv00[1,0], egv00[1,1],color='b',head_width=0.01)
# ax.arrow(0,0,-egv10[0,0], -egv10[0,1],color='r',head_width=0.01)
# ax.arrow(0,0,-egv10[1,0], -egv10[1,1],color='r',head_width=0.01)

# ax.quiver(basep,basep,-egv10[:,0], -egv10[:,1],color='r')


ax.quiver(downsp_ki67[:,:,0].mean(),
         downsp_ki67[:,:,1].mean(),
         downsp_ki67[:,:,2].mean(),egv00[0,0], egv00[0,1], egv00[0,2],color='r')
ax.quiver(downsp_ki67[:,:,0].mean(),
         downsp_ki67[:,:,1].mean(),
         downsp_ki67[:,:,2].mean(),egv00[1,0], egv00[1,1], egv00[1,2],color='g')
ax.quiver(downsp_ki67[:,:,0].mean(),
         downsp_ki67[:,:,1].mean(),
         downsp_ki67[:,:,2].mean(),egv00[2,0], egv00[2,1], egv00[2,2],color='b')

ax.quiver(z[0, :].mean(),
         z[1,:].mean(),
         z[2,:].mean(),egv10[0,0], egv10[0,1], egv10[0,2],color='r')
ax.quiver(z[0,:].mean(),
         z[1,:].mean(),
         z[2,:].mean(),egv10[1,0], egv10[1,1], egv10[1,2],color='g')
ax.quiver(z[0,:].mean(),
         z[1,:].mean(),
         z[2,:].mean(),egv10[2,0], egv10[2,1], egv10[2,2],color='b')

plt.tight_layout()
plt.grid()


# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')

z = np.array([
    im_res[:,:,0].ravel(),
    im_res[:,:,1].ravel(),
    im_res[:,:,2].ravel(),
    ones_like(im_res[:,:,0].ravel())
    ])
z = warp_mat@z
z

h = z[0, :].reshape((2048, 2048));
e_r = z[1, :].reshape((2048, 2048));
d = z[2, :].reshape((2048, 2048))

ress = np.dstack([h, e_r, d])

fig = plt.figure(figsize=(10,20), dpi=300);
plt.subplot(131);plt.imshow(im_res);plt.title("Before");plt.axis('off')
plt.subplot(132);plt.imshow(ress);plt.title("After");plt.axis('off')
plt.subplot(133);plt.imshow(im_ki67);plt.title("Physical");plt.axis('off')
plt.tight_layout()


# In[36]:


im_ref = im_ki67 = cv.imread("/Users/cunyuan/DATA/Kimura/Endometrioid Carcinoma/G2/17-6747/my2048/dab/0_12_01_17-6747_Ki67_IHC_(d=1.0000000, x=90846.0, y=2531.0, w=2048.0, h=2048.0, z=12).tif")
im_ref  = cv.cvtColor(im_ki67, cv.COLOR_BGR2RGB)/255.


# In[37]:


plt.figure(figsize=(6,6), dpi=300)
plt.imshow(im_ref)
plt.figure(figsize=(6,6), dpi=300)
plt.imshow(im_res)


# In[38]:


def calib_color(im_ref, im_res, size=(200, 200),):
    hed_ref = abs(rgbdeconv(im_ref, Hinv))
    hed_res = abs(rgbdeconv(im_res, Hinv))
    p_hed_ref = im_ref
    p_hed_res = im_res
    c011 = np.array([p_hed_ref[:,:,0].mean()/p_hed_res[:,:,0].mean(),
                    p_hed_ref[:,:,1].mean()/p_hed_res[:,:,1].mean(),
                    p_hed_ref[:,:,2].mean()/p_hed_res[:,:,2].mean(),])
    return c011


# In[39]:


c01d = calib_color(im_ref, im_res, [1300,300], [0, 300], [250, 250])
print(c01d)


# In[40]:


plt.imshow(im_res)


# In[41]:


patch_res = im_res[:250, 1000:1250, :]


# In[42]:


plt.imshow(patch_res)


# In[43]:


hed_ref = abs(rgbdeconv(im_ref, Hinv))
hed_res = abs(rgbdeconv(im_res, Hinv))


# In[44]:


plt.imshow(hed_ref[:,:,0], cmap=cmap_hema)


# In[45]:


p_hed_res = hed_res[:250, 1000:1250, :]
p_hed_ref = hed_ref[1250:1500, :250, :]


# In[46]:


c011 = calib_color(im_ref, im_res, [1000,0], [0, 1250], [250, 250])
c011[2] = c01d[2]
c011


# In[47]:


zdh = hed_res
zdh = (zdh.reshape(-1,3)*c011.T).reshape(img.shape)
correct_zdh = hecconv(zdh, H)

correct_zdh = rescale_intensity(correct_zdh)

fig = plt.figure(figsize=(20,20))
axis = plt.subplot(1, 2, 1)
axis.imshow(correct_zdh)
axis.set_title("Pseudo Hematoxylin-DAB")
axis.axis('off')

axis = plt.subplot(1,2,2)
axis.imshow(im_ref)
axis.set_title("True Hematoxylin-DAB")
axis.axis('off')
plt.tight_layout()
plt.show()


# In[48]:


n = 3
cvr = 0.2
color_range = np.array([np.linspace(c011[0]*(1-cvr), c011[0]*(1+cvr), n),
                  np.linspace(c011[1]*(1-cvr), c011[1]*(1+cvr), n),
                  np.linspace(c011[2]*(1-cvr), c011[2]*(1+cvr), n)])


# In[49]:


def color_var_plot(hed_res, n, cvr, c011, mode):
    color_range = np.array([np.linspace(c011[0]*(1-cvr), c011[0]*(1+cvr), n),
                  np.linspace(c011[1]*(1-cvr), c011[1]*(1+cvr), n),
                  np.linspace(c011[2]*(1-cvr), c011[2]*(1+cvr), n)])
    fig = plt.figure(figsize=(10,10), dpi=300)
    for k in range(n):
        for j in range(n):
            zdh = hed_res
            if mode=="he":
                v = np.array([color_range[0, k],
                                color_range[1, j],
                                                  c011[2],]).T
                zdh = (zdh.reshape(-1,3)*v).reshape(img.shape)
                char1, char2 = "h, d", "e"
            elif mode=="ed":
                v = np.array([c011[0],
                                  color_range[1, k],
                                  color_range[2, j],]).T
                zdh = (zdh.reshape(-1,3)*v).reshape(img.shape)
                char1, char2 = "h, e", "d"
            else: #hd
                v = np.array([color_range[0, j],
                                  c011[1],
                                  color_range[2, k],]).T
                zdh = (zdh.reshape(-1,3)*v).reshape(img.shape)
                char1, char2 = "e, d", "h"

            correct_zdh = hecconv(zdh, H)

            correct_zdh = rescale_intensity(correct_zdh)
            plt.subplot(n,n,k*n + j +1)
            plt.imshow(correct_zdh)
            plt.title("%2.1f, %2.1f,%2.1f"%(v[0],v[1], v[2]
                                              ))
            plt.axis('off')
            plt.tight_layout()
    plt.show()


# In[50]:


color_var_plot(hed_res, 3, 0.5, c011, "hd")


# In[51]:


color_var_plot(hed_res, 3, 0.5, c011, "he")


# In[52]:


color_var_plot(hed_res, 3, 0.5, c011*[2,1,1], "ed")


# In[53]:


c011


# In[ ]:
