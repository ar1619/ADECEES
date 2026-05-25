import random
import time

# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import animation

import dataset
import evaluation
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from UNet import UNetModel

ROOT_DIR = "./"
DATA_DIR = "../../data/"

def segment_values_above_threshold(image, threshold_plus=0.5, threshold_minus=-0.5, segmentation='positive'):
    """
    Segments the values in the image that are above the threshold.
    """
    # Create a mask of the same shape as the image
    segmented = np.zeros_like(image)

    if segmentation == 'positive':
        # Set the values in the mask to 1 where the image is above the threshold
        segmented[image > threshold_plus] = 1
    if segmentation == 'negative':
        # Set the values in the mask to -1 where the image is below the threshold
        segmented[image < threshold_minus] = 1
    if segmentation == 'both':
        # Set the values in the mask to 1 where the image is above the positive threshold
        segmented[image > threshold_plus] = 1
        # Set the values in the mask to -1 where the image is below the negative threshold
        segmented[image < threshold_minus] = -1
    return segmented

def plot_sequence(diffs, segmentation_type, args_n):
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 3)
    for i in range(6):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        im = ax.imshow(segment_values_above_threshold(diffs[i][0, 0, ...], segmentation=segmentation_type), aspect='equal')
        # cbar = plt.colorbar(im, ax=ax, fraction=0.05)
        # ax.set_title('Diff {}'.format(i))
        ax.axis('off')
        
    filename = 'year/'+args_n+'_' + segmentation_type + '.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
def plot_ensemble(image, mean_output, variance_output, t, filename, n_ensemble=10, segmentation_type='positive'):
    """
    Plot the ensemble mean and variance of the output
    """
    diff = image - mean_output
    diff = diff.cpu().numpy()
    image = image.cpu().numpy()
    mean_output = mean_output.cpu().numpy()
    variance_output = variance_output.cpu().numpy()
    folder = 'ensemble/'

    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.1, 1.1, 1, 1], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(image[0, 0, ...], aspect='equal')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.05)
    cbar1.set_ticks(np.arange(-1, 1.5, 0.5))
    cbar1.ax.tick_params(labelsize=16)  # Increase font size of colorbar ticks
    # ax1.set_title('x$_{weekly}$')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(mean_output[0, 0, ...], aspect='equal', vmax=1, vmin=-1)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.05)
    cbar2.set_ticks(np.arange(-1, 1.5, 0.5))
    cbar2.ax.tick_params(labelsize=16)
    # ax2.set_title('Avg. reconstruction')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(segment_values_above_threshold(diff[0, 0, ...], threshold_plus=t, segmentation=segmentation_type), aspect='equal')
    # cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.05)
    # ax3.set_title('Segmented anomalies')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[3])
    im4 = ax4.imshow(variance_output[0, 0, ...], aspect='equal', cmap='inferno')
    # cbar = plt.colorbar(im4, ax=ax4, orientation='horizontal', fraction=0.05)
    # ax4.set_title('Variance')
    ax4.axis('off')

    # print('max:', max(variance_output[0, 0, ...].flatten()), 'min:', min(variance_output[0, 0, ...].flatten()))
    filename = folder + filename

    plt.savefig(filename)
    plt.clf()

def main():
    """
        Main function for anomaly detection
    """

    _, output = load_parameters(device)

    n_ensemble = 50
    # read file from argument
    if len(sys.argv[1:]) > 0:
        files = sys.argv[1:]
    else:
        raise ValueError("Missing file argument")

    # allow different arg inputs ie 25 or args15 which are converted into argsNUM.json
    file = files[0]
    if file.isnumeric():
        file = f"args{file}.json"
    elif file[:4] == "args" and file[-5:] == ".json":
        pass
    elif file[:4] == "args":
        file = f"args{file[4:]}.json"
    else:
        raise ValueError("File Argument is not a json file")

    # load the json args
    with open(f'{ROOT_DIR}test_args/{file}', 'r') as f:
        args = json.load(f)
    args['arg_num'] = file[4:-5]
    args = defaultdict_from_json(args)

    in_channels = 1
    t = args["threshold"]

    print(f"args{args['arg_num']}")
    unet = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], in_channels=in_channels
            )

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diff = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
            )

    unet.load_state_dict(output["ema"])
    unet.to(device)
    unet.eval()

    ensemble = args["ensemble"]

    args["start_date_test"] = str(files[1])
    args["end_date_test"] = str(files[2])
    if ensemble:
        segmentation_type = str(files[3])
    else:
        sequence = str(files[3])
    testing_dataset = dataset.OCO2Dataset_SR(args, DATA_DIR, False)
    loader = dataset.cycle(testing_dataset)

    # dice_data = []
    # ssim_data = []
    # IOU = []
    # precision = []
    # recall = []
    # FPR = []
    # AUC_scores = []

    start_time = time.time()
    dataset_size = len(testing_dataset)
    var_outputs = []
    images = []

    for i in range(4):
        new = next(loader)
        image = torch.unsqueeze(new["image"],0).to(device)
        filename = new["filenames"]
        print("Image: ", filename, "Index: ", i)

        if ensemble:
            # ensemble test
            outputs = []
            for j in range(n_ensemble):
                output = diff.forward_backward(
                        unet, image,
                        see_whole_sequence=None,
                        t_distance=200
                        )
                outputs.append(output)
                # if j == 9 or j == 49 or j == 99:
            variance_output = torch.var(torch.stack(outputs), dim=0)
            avg_variance = variance_output.mean().item()
            print(f"Average variance for image {filename}: {avg_variance} at n_ensemble ={50}")
            plot_ensemble(image, torch.mean(torch.stack(outputs), dim=0), variance_output, t, "{}_{}_{}_{}.png".format(filename, args['arg_num'], str(n_ensemble), segmentation_type), n_ensemble, segmentation_type)

                # mse = (image - output).square()
                # mse = (mse > 0.5).float()

                # ssim_data.append(
                #         evaluation.SSIM(
                #                 image.reshape(*args["img_size"], image.shape[1]),
                #                 output.reshape(*args["img_size"], image.shape[1])
                #                 )
                #         )



if __name__ == "__main__":
    import sys
    from matplotlib import font_manager
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()
