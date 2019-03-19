#Uncomment and Run this is plot_model does not work
import matplotlib.pyplot as plt
import os

os.environ["PATH"] =  os.environ["PATH"] + os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'

def show_image(imagefile, w, h):
#     %pylab inline
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img=mpimg.imread(imagefile)
    plt.figure(figsize=(w, h))
    imgplot = plt.imshow(img);
    plt.show();

def show_model(model, w, h):
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True);
    show_image('model_plot.png', w, h)