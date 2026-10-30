import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
import matplotlib.animation as animation


def show_image(image, title='image', verbose=True):
    if image is not None:
        if(len(image.shape)==2):
            plt.imshow(image, cmap = 'gray', interpolation = 'bicubic', vmin = 0, vmax = 255)
        else:
            plt.imshow(image, interpolation = 'bicubic', vmin = 0, vmax = 255)

        plt.title(title)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        if verbose:
            print("Image shape:", image.shape)
            print("Image data type:", image.dtype)
    else:
        print("Unable to load the Image, check file name")
        
    
def show_images(images, titles='None', columns = 2, size=20):
    if titles == 'None':
        titles = range(0, len(images))
    plt.figure(figsize=(size,size))
    for i in range(len(images)):
        plt.subplot(len(images) // columns + 1, columns, i + 1)
        if(len(images[i].shape)==2):
            plt.imshow(images[i], cmap = 'gray', interpolation = 'bicubic', vmin = 0, vmax = 255)
        else:
            plt.imshow(images[i], interpolation = 'bicubic', vmin = 0, vmax = 255)

        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    
    plt.show()
    
    
def show_figure(image, title='image', verbose=True):
    if image is not None:
        if(len(image.shape)==2):
            plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
        else:
            plt.imshow(image, interpolation = 'bicubic')

        plt.title(title)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        if verbose:
            print("Image shape", image.shape)
            print("Image data type", image.dtype)
    else:
        print("Unable to load the Image, check file name")
        
    
def show_figures(images, titles='None', columns = 2, size=20):
    if titles == 'None':
        titles = range(0, len(images))
    plt.figure(figsize=(size,10))
    for i in range(len(images)):
        plt.subplot(len(images) // columns + 1, columns, i + 1)
        if(len(images[i].shape)==2):
            plt.imshow(images[i], cmap = 'gray', interpolation = 'bicubic')
        else:
            plt.imshow(images[i], interpolation = 'bicubic')

        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    
    plt.show()
    
        
def show_spectrums3D(images, titles='None', columns = 2, size=20):
    if titles == 'None':
        titles = range(0, len(images))
    fig = plt.figure(figsize=(size,10))
    for i in range(len(images)):
        ax = fig.add_subplot(len(images) // columns + 1, columns, i + 1, projection='3d')
        XX, YY = np.mgrid[0:images[i].shape[0],0:images[i].shape[1]]
        ax.plot_surface(XX, YY, images[i], cmap='coolwarm', linewidth=0, antialiased=False)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    
def create_figure_animation(sequence, interval=50):
    video_sequence = []
    fig = plt.figure()
    for i in range(len(sequence)):
        if len(sequence[i].shape) == 2:
            animated_frame = plt.imshow(sequence[i], cmap = 'gray', interpolation = 'bicubic', animated=True)
        else:
            animated_frame = plt.imshow(sequence[i], interpolation = 'bicubic', animated=True)

            # so it seems that I cannot plot direcly a vidoe, I have to create the pyplot object to print and than create 
            # a new sequence of this pyplot objects, which are not numpy's ndarrays.
        video_sequence.append([animated_frame])
    
    # given the sequence of pylot images, it is possible to create the video 
    video = animation.ArtistAnimation(fig, video_sequence, interval=interval, blit=False, repeat_delay=1000)

    # equivalent to rcParams['animation.html'] = 'jshtml'
    rc('animation', html='jshtml')

    return video


def create_image_animation(sequence, interval=50):
    video_sequence = []
    fig = plt.figure();
    for i in range(len(sequence)):
        if len(sequence[i].shape) == 2:
            animated_frame = plt.imshow(sequence[i], cmap = 'gray', interpolation = 'bicubic', vmin=0, vmax=255, animated=True);
        else:
            animated_frame = plt.imshow(sequence[i], interpolation = 'bicubic', vmin=0, vmax=255, animated=True);

            # so it seems that I cannot plot direcly a vidoe, I have to create the pyplot object to print and than create 
            # a new sequence of this pyplot objects, which are not numpy's ndarrays.
        video_sequence.append([animated_frame]);
    
    # given the sequence of pylot images, it is possible to create the video 
    video = animation.ArtistAnimation(fig, video_sequence, interval=interval, blit=True, repeat_delay=1000);

    # equivalent to rcParams['animation.html'] = 'jshtml'
    rc('animation', html='jshtml')

    return video