"""
=========
Stem Plot
=========

`~.pyplot.stem` plots vertical lines from a baseline to the y-coordinate and
places a marker at the tip.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
DEBUG=False

def stem2(x,y1,label1,y2,label2,title='Вторая строка заголовка',titsize=12,suptitle='Общий заголовок',supsize=16,pdfname='Report.pdf',SAVEPP=False):
    fig, ax = plt.subplots()
    fig.suptitle(suptitle, fontsize=16,fontweight="bold")
    #fig.set_title()
    #fig.tight_layout()
    #fig.subplots_adjust(bottom=0.0)  #
    #plt.figtext(.5,.8,suptitle, fontsize=supsize,ha='center')  # Add the text/suptitle to figure
    plt.figtext(.5, .9, title, fontsize=titsize, ha='center')
    ax.stem(x, y1, 'b', markerfmt='bo', basefmt=" ", label=label1)
    ax.stem(x, y2, 'g', markerfmt='go', basefmt=" ", label=label2)
    ax.set_ylabel('температура')
    ax.set_xlabel('наблюдения')
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x*0, 'r-')
    #ax.set_title(title,fontsize=titsize)
    ax.legend()

    #plt.show()
    if SAVEPP:
        pp=PdfPages(pdfname)
        pp.savefig(fig)
        #plt.rcParams['text.usetex'] = True
        #pp.attach_note("plot of sin(x)")
        #pp.savefig(fig)
        pp.close()

def stemplot(ax,x, y, label='line',title='Вторая строка заголовка',titsize=12,suptitle='Общий заголовок',supsize=16,pdfname='Report1.pdf',SAVEPP=False):
    if DEBUG:print("lenX = {}, lenY= {}".format(len(x),len(y)))
    #plt.stem(x, y, linefmt='C0-',use_line_collection=True)
    fig, ax = plt.subplots(1,1,1)
    ax=plt.gca()
    fig.suptitle(suptitle, fontsize=supsize, fontweight="bold",ha='center')
    plt.figtext(.5, .9, title, fontsize=titsize, ha='center')
    ax.stem(x, y, 'b', markerfmt='bo', basefmt=" ", label=label, use_line_collection = True)
    ax.set_ylabel('температура')
    ax.set_xlabel('наблюдения')
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x*0, 'r-')
    ax.legend()
    plt.show()

    #plt.savefig('zzz.png')
    if SAVEPP:
        pp=PdfPages(pdfname)
        pp.savefig(fig)
        #plt.rcParams['text.usetex'] = True
        #pp.attach_note("plot of sin(x)")
        #pp.savefig(fig)
        pp.close()

    # plt.plot(x[0], x[0] + 20, '-и)
    # plt.show()
    return True


def rays_plot(ax,start_x, start_y, end_x, end_y,color='-g'):
    ax.plot([start_x, end_x],[start_y,end_y],color)
    plt.savefig('zzz-plus-rays.png')

    return True


# x = np.linspace(0.1, 2 * np.pi, 100)
# y = np.exp(np.sin(x))
# stemplot(x, y)
#############################################################################
#
# The position of the baseline can be adapted using *bottom*.
# The parameters *linefmt*, *markerfmt*, and *basefmt* control basic format
# properties of the plot. However, in contrast to `~.pyplot.plot` not all
# properties are configurable via keyword arguments. For more advanced
# control adapt the line objects returned by `~.pyplot`.
#exit
# markerline, stemlines, baseline = plt.stem(
#     x, y, linefmt='grey', markerfmt='D', bottom=1.1, use_line_collection=True)
# markerline.set_markerfacecolor('green')
# # plt.show()
# plt.savefig(
#     'saved_figure.png'
# )
#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods, classes and modules is shown
# in this example:

import matplotlib

# matplotlib.pyplot.stem
# matplotlib.axes.Axes.stem
