## Spectrogram Modification of StyleGAN3<br><sub>As of yet, divergent and unfinished</sub>

Please see the main [StyleGAN3](https://github.com/NVlabs/stylegan3) branch for general overview of StyleGAN3, and for the functional standard product. For more in-depth information on the technology, please see these three papers: 

- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)
- [Analyzing and Improving the Image Quality of StyleGAN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.pdf)
- [Alias-Free Generative Adversarial Networks](https://arxiv.org/pdf/2106.12423.pdf)


This fork is an in-progress modification of StyleGAN3 which aims to utilize StyleGAN3 for generation of instances of a lossless audio spectrogram. These spectrograms are then reverse-transformed back into audio, thus allowing StyleGAN3 to be trained on a set of audio files and then generate its own, unique sounds. The current implementation is messy and incomplete. As it stands, the network diverges quite rapidly towards brightness and monochromaticity, though there are many areas still to be worked on. For more in-depth information on these modifications, please see my paper [Generating Sounds that Don't Exist](https://www.dropbox.com/s/51gqnopiid6c7nc/ML_Final_Report.pdf?dl=0).

What follows are instructions on how to reproduce the experiments described in this paper. However, it should be possible to run with any audio dataset.


## Instructions<br><sub>StyleGAN3-Spectrogram, Jon Henshaw, 2/21/2022</sub>


This Readme will walk you through the steps necessary to run the software as intended. 

1) Requirements

Ensure your system complies with all hardware and software requirements for [StyleGAN3](https://github.com/NVlabs/stylegan3). Please note that this "Spectrogram" variant of the StyleGAN3 package has only been tested on a virtual linux machine via Paperspace Gradient, instantiated using a single RTX-5000 GPU, so your results may vary if you attempt to run it with other hardware or software setups. As the StyleGAN3 source files have been modified for this package, no guarantees are made about having maintained the full scope of StyleGAN3's original functionality.

You will need ~100GB of free storage space to complete construction of the dataset and training. A Python installation of 3.8 or later is required, as well as a functioning installation of the Anaconda package manager. You may also wish to update/upgrade command line packages including zip, unzip, and wget.

2) Downloading Data

Navigate to this folder on the command line, and run the following commands:

	chmod +x europadl.sh
	./europadl.sh

This will download the Europa .wav training set. It is ~22GB in size, so ensure you have enough storage space to hold it. You may want to empty your trash after, as there are excess files that will be discarded during this process.

3) Environment Setup

To automatically set up the required Python environment with Anaconda, run:

	conda update -n base conda
	conda env create -f environment.yml

Once the environment has been successfully created, type:

	conda activate stylegan3-spect

If you have not installed conda previously, you may need to run the following first:

	pip install conda
	conda init bash
	exec bash

If you are not using a bash terminal, replace "bash" with the name of the terminal you are using.

4) Construct Data Set

Once conda's stylegan3-spect environment has been set up and has been activated, it is time to use the dataset tool to construct the dataset for use with training. You can do so with the following command:

	python dataset_tool.py --source wav_files/europa_wav --dest datasets/europa.zip

The dataset will take up another ~66GB. Once this has completed, if you need to free up some space, you are free to delete the .wav files. However, you will have to re-download them if you eventually intend to create another dataset utilizing these files, or to use them for comparison with generated files.

5) Training

To begin training for 10kimgs, run:

	python train.py --outdir=trained_networks/europa_net --gpus=1 --batch=8 --gamma=8 --data=datasets/europa.zip --cfg=stylegan3-t --kimg=10 --tick=1 --snap=1 --metrics=none

This will generate another ~11GB of files in the form of network snapshots, etc.

6) Continue Training

Once training has been completed, you can chose to resume training for more kimgs if you wish, utilizing:

	python train.py --outdir=trained_networks/europa_net --gpus=1 --batch=8 --gamma=8 --data=datasets/europa.zip --cfg=stylegan3-t --kimg=10 --tick=1 --snap=1 --metrics=none --resume trained_networks/europa_net/00000-stylegan3-t-europa-gpus1-batch8-gamma8/network-snapshot-000010.pkl

This will create a new folder, 00001-stylegan3-t-europa-gpus1-batch8-gamma8, which will contain new snapshots, etc, numbered 0-10. You may resume training from any snapshot you've previously generated by replacing the --resume filepath with the one you wish to use.

7) Generation

If you wish to export the first 10 seed images from a particular snapshot, you can run:

	python gen_images.py --network trained_networks/europa_net/00000-stylegan3-t-europa-gpus1-batch8-gamma8/network-snapshot-000010.pkl --outdir /notebooks/generated/10 --seeds 0-9

You are also free to use any --seeds numbers or ranges you would like, as well as choosing any of your existing --network snapshots from which to generate the images.

8) Warning

Please be careful with the volume control on your computer if you decide to listen to the .wav files generated. Headphones are recommended for maximum thrill.
