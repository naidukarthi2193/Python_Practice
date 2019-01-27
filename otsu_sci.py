from skimage.filters.rank import otsu
from skimage.filters import threshold_otsu
from skimage import data
import matplotlib as plt
from skimage import morphology as disk

p8 = data.page()

radius = 10
# selem = disk(radius)

# t_loc_otsu is an image
t_loc_otsu = otsu(p8, disk(10))
loc_otsu = p8 >= t_loc_otsu

# t_glob_otsu is a scalar
t_glob_otsu = threshold_otsu(p8)
glob_otsu = p8 >= t_glob_otsu

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
ax1, ax2, ax3, ax4 = ax.ravel()

fig.colorbar(ax1.imshow(p8, cmap=plt.cm.gray), ax=ax1)
ax1.set_title('Original')

fig.colorbar(ax2.imshow(t_loc_otsu, cmap=plt.cm.gray), ax=ax2)
ax2.set_title('Local Otsu ($r=%d$)' % radius)

ax3.imshow(p8 >= t_loc_otsu, cmap=plt.cm.gray)
ax3.set_title('Original >= local Otsu' % t_glob_otsu)

ax4.imshow(glob_otsu, cmap=plt.cm.gray)
ax4.set_title('Global Otsu ($t=%d$)' % t_glob_otsu)

for ax in ax.ravel():
    ax.axis('off')
    ax.set_adjustable('box-forced')