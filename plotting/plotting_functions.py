import matplotlib as mpl
import matplotlib.pyplot as plt

# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.xticks()
    plt.axis('off')

    def update_scene(num, frames, patch):
        patch.set_data(frames[num])
        return patch,

    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim