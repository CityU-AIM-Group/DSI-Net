
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def show_seg_results(img, gt, pre, save_path = None, name = None):
    
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.imshow(img)
    ax.axis('off')
    ax = fig.add_subplot(132)
    ax.imshow(gt)
    ax.axis('off')
    ax = fig.add_subplot(133)
    ax.imshow(pre)
    ax.axis('off')
    fig.suptitle('Img, GT, Prediction',fontsize=6)
    if save_path != None and name != None:
        fig.savefig(save_path + name + '.png', dpi=200, bbox_inches='tight')
    ax.cla()
    fig.clf()
    plt.close()

def draw_curves(data_list, label_list, color_list, linestyle_list = None, filename = 'training_curve.png'):
    
    plt.figure()
    for i in range(len(data_list)):
        data = data_list[i]
        label = label_list[i]
        color = color_list[i]
        if linestyle_list == None: 
            linestyle = '-'  
        else: 
            linestyle = linestyle_list[i]
        plt.plot(data, label = label, color = color, linestyle = linestyle)     
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.clf()
    plt.close()
    plt.show()
    plt.close('all')