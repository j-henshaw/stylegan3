#——————————————————————————————————————————————————————————————————————————————#
# Defines a transform and a reverse transform for a 4-channel                  #
# image file using doubles instead of 8-bit ints, stored as .npy               #
#——————————————————————————————————————————————————————————————————————————————#

#——————————————————————————————————————————————————————————————————————————————#
# Imports                                                                      #
#——————————————————————————————————————————————————————————————————————————————#

import numpy as np
import scipy.fft as fft
from PIL import Image as img
import soundfile as sf

#——————————————————————————————————————————————————————————————————————————————#
# Function Definitions                                                         #
#——————————————————————————————————————————————————————————————————————————————#

# This fn uses np.interp() to resample any buffer at a new sample rate
def splRteInterpolate(buff, old_rate, new_rate):
    #Initialize
    ratio = new_rate/old_rate
    if ratio == 1: #early exit dude
        return buff
    
    #Calculate old n new sample times; may drop a sample
    old_samp_times = np.arange(len(buff))/old_rate
    new_samp_times = np.arange(int(np.floor(len(buff) * ratio)))/new_rate

    #Stereo vs mono
    channels = getLenChan(buff)[1]
    
    if channels == 1:
        return np.interp(new_samp_times, old_samp_times, buff)
    else:
        nu_buff = np.empty( (int(np.floor( len(buff) * ratio)) , 2) )
        #Left
        nu_buff[:,0] = np.interp(new_samp_times, old_samp_times, buff[:,0])
        #Right
        nu_buff[:,1] = np.interp(new_samp_times, old_samp_times, buff[:,1])
        return nu_buff


#——————————————————————————————————————————————————————————————————————————————#
# This function takes in a mono or stereo buffer and returns its shape:
# num_samples, num_channels
def getLenChan(buff):
    #Get dimensions
    wavlen = buff.shape[0]
    chan = 1
    try:
        chan = buff.shape[1]
    except IndexError:
        chan = 1

    if (chan != 1) and (chan != 2):
        raise ValueError("This file is neither mono nor stereo")

    return wavlen,chan

#——————————————————————————————————————————————————————————————————————————————#
# This function takes in a 3D array representing a lossless spectrogram, and
# outputs a .png to the current folder
def ARRtoPNG(img_array, outname, SPL_RTE):
    #Get shape
    h,w,d = img_array.shape
    img_arr = np.empty((h,w,d),dtype=np.float32)

    #Put on log2 scale so octaves are evenly spaced
    old_freqs = fft.rfftfreq(h*2) * SPL_RTE
    new_freqs = np.logspace(np.log2(old_freqs[1]), np.log2(old_freqs[-1]), \
                                num=h, base=2, dtype=np.float32)
    for n in range(w):
        img_arr[:,n,0] = np.interp(old_freqs[1:], new_freqs, img_array[:,n,0])
        img_arr[:,n,1] = np.interp(old_freqs[1:], new_freqs, img_array[:,n,1])
        img_arr[:,n,2] = np.interp(old_freqs[1:], new_freqs, img_array[:,n,2])
        img_arr[:,n,3] = np.interp(old_freqs[1:], new_freqs, img_array[:,n,3])

    #Normalize magnitudes to [0,1]
    img_arr = np.abs(img_arr)
    img_arr = img_arr / (np.max(img_arr) - np.min(img_arr))
    img_arr = img_arr - np.min(img_arr)

    #Create int8 array and convert from 'CMYK' to 'RGB'
    png_arr = np.empty((h,w,3),dtype=np.uint8)
    png_arr[:,:,0] = np.round(255 * (1 - img_arr[:,:,0]) * (1 - img_arr[:,:,3]))
    png_arr[:,:,1] = np.round(255 * (1 - img_arr[:,:,1]) * (1 - img_arr[:,:,3]))
    png_arr[:,:,2] = np.round(255 * (1 - img_arr[:,:,2]) * (1 - img_arr[:,:,3]))

    #Invert colors so it's easier to look at
    png_arr = 255 - png_arr
    
    #Export
    png_array = img.fromarray(png_arr,'RGB')
    png_array.save(outname + '.png','png')

    return

#——————————————————————————————————————————————————————————————————————————————#
# This function outputs a .npy
def ARRtoNPY(img_array, outname):
    #outname = "npy/" + outname + '_t'
    #Save it
    np.save(outname,img_array,allow_pickle=False)
    return

#——————————————————————————————————————————————————————————————————————————————#
# This function returns a string describing a time interval of the number
# of samples provided at the sample rate provided. Assumes both are positive
def timeInterval(samples,rate):
    secs = samples/rate
    mins = int(secs//60)
    secs = secs%60
    hrs = int(mins//60)
    mins = mins%60

    return format(hrs,'02.0f') + ":" + \
           format(mins,'02.0f') + ":" + \
           format(secs,'.3f')

#——————————————————————————————————————————————————————————————————————————————#
# This function takes in the name of a .wav file as a string, and
# saves a .npy to a subfolder. Optional parameter of WINDOW_SIZE is
# default set to 2048, and must be even, ideally a power of 2. If the named
# .wav file is neither mono nor stereo, will abort early and fail. Optional
# boolean parameter make_png tells whether or not to create a .png as well as
# a .npy. Returns a 3d nparray with 4-channel information on a 2d grid
def WAVtoARR(wav_name,WINDOW_SIZE = 1024,SPL_RTE = 44100,\
            make_png=False,make_npy=False,make_square=True):

    num_freqs = WINDOW_SIZE//2 #since we discard neg freqs (rfft) & DC offset
    #print("Processing " + wav_name + "...")

    #Read in the file
    try:
        buff,rate = sf.read(wav_name)
    except RuntimeError:
        return None
    wavlen,chan = getLenChan(buff)

    if rate != SPL_RTE:
        #print("Resampling from " + str(rate) + " to " + str(SPL_RTE))
        buff = splRteInterpolate(buff,rate,SPL_RTE)
        rate = SPL_RTE

    #Get mid/side arrays on [-1,1], padded with zeros to even num_windows
    mid,sid = makeMidSide(buff,WINDOW_SIZE)

    #Pad or truncate channels to a length that would make a square image
    if make_square:
        target_len = WINDOW_SIZE * num_freqs 
        diff = target_len - len(mid)
        if diff > 0:
            #print("\tPadding signal with " + str(diff) + " zeroes (" + \
                  #timeInterval(diff,rate) + "), to make square")
            mid = np.concatenate((mid,np.zeros(diff)))
            sid = np.concatenate((sid,np.zeros(diff)))
        elif diff < 0:
            #print("\tTruncating signal by " + str(-diff) + " samples (" + \
                  #timeInterval(-diff,rate) + "), to make square")
            mid = mid[0:target_len]
            sid = sid[0:target_len]

    #Construct empty img array
    num_windows = len(mid)//WINDOW_SIZE
    img_array = np.empty((num_freqs,num_windows,4), dtype=np.float32)

    #Fill it
    for n in range(num_windows):
        #Since our audio signal is entirely real, compute rfft for each window
        mid_freqs = fft.rfft(mid[ n * WINDOW_SIZE : (n+1) * WINDOW_SIZE])
        sid_freqs = fft.rfft(sid[ n * WINDOW_SIZE : (n+1) * WINDOW_SIZE])
        
        #Deconstruct real and complex parts, and insert into img array
        #Assume signal's sum = 0 for transform and reverse
        for i in range(num_freqs):
            #Each pixel gets: [mid_r, mid_i, sid_r, sid_i]
            #Also, we are ignoring the first index because we assume it = 0
            #Low freqs go at the bottom of the image
            img_array[num_freqs-1-i][n] = np.array(\
                [mid_freqs[i + 1].real, mid_freqs[i + 1].imag, \
                 sid_freqs[i + 1].real, sid_freqs[i + 1].imag],\
                dtype=np.float32)

    #Output .npy file
    if make_npy:
        ARRtoNPY(img_array,wav_name[0:-4])

    #Output .png file
    if make_png:
        ARRtoPNG(img_array, wav_name[0:-4],rate)

    return img_array


#——————————————————————————————————————————————————————————————————————————————#
# This function takes an audio buffer from a wav file and returns two
# 1D numpy arrays containing the mid (sum/2) and side (diff/2) data.
# Optional parameter of WINDOW_SIZE is default set to 1024. Mono/Stereo
# compatibility only. Will pad each array with zeros to fit evenly into
# a discrete number of windows. Values returned will be float32s on [-1,1]
def makeMidSide(buff,WINDOW_SIZE=1024):
    #Get dimensions
    wavlength,channels = getLenChan(buff)

    #Mono and stereo compatibility only
    if channels != 1 and channels != 2:
        raise ValueError("This audio file has " + str(channels) + \
                         " channels, which is not supported.")

    #Normalize to [-1,1]
    buff = buff.astype(np.float32)
    min_adj = np.min(buff)
    max_adj = np.max(buff)
    mag_adj = (max_adj - min_adj) / 2
    mag_adj = max(mag_adj, np.finfo(np.float32).eps) #div by 0 safety
    min_adj += mag_adj
    buff = (buff - min_adj) / mag_adj

    #We round up the length to the nearest window size, padding zeros
    bonus = WINDOW_SIZE - wavlength % WINDOW_SIZE
    mid = 0
    sid = 0

    #Calculate mid and side arrays
    #For mono signals
    if channels == 1:
        if bonus > 0:
            mid = np.concatenate((buff,np.zeros(bonus)))
        else:
            mid = buff
        sid = np.zeros(wavlength + bonus)
    #For stereo signals
    else:
        if bonus > 0:
            # [-1,1] + [-1,1] could be on [-2,2]
            mid = np.concatenate((((buff[:,0] + buff[:,1])/2), \
                                  np.zeros(bonus)))
            # [-1,1] - [-1,1] could be on [-2,2]
            sid = np.concatenate((((buff[:,0] - buff[:,1])/2), \
                                  np.zeros(bonus)))
        else:
            mid = (buff[:,0] + buff[:,1]) / 2
            sid = (buff[:,0] - buff[:,1]) / 2

    return mid,sid


#——————————————————————————————————————————————————————————————————————————————#
# This function takes in the name of an .npy file and makes a wav
def NPYtoWAV(npy_name,SPL_RTE=44100):
    if not ('.npy' in npy_name):
        npy_name += '.npy'
    
    img_arr = np.load(npy_name,allow_pickle=False)
    ARRtoWAV(img_arr, npy_name[0:-4] + '_t', SPL_RTE=SPL_RTE)


#——————————————————————————————————————————————————————————————————————————————#
# This function takes in the name of an .npy file and makes a wav
def NPYtoARR(npy_name,SPL_RTE=44100):
    if not ('.npy' in npy_name):
        npy_name += '.npy'
    img_arr = np.load(npy_name,allow_pickle=False)

    if len(img_arr.shape) < 3: #Something went wrong with .npy
        #Remove metadata:
        #Find the actual dimensions
        size = 1
        falsity = len(img_arr)
        while (falsity > size*2):
            size *= 2
        #Reshape
        img_arr = (img_arr[falsity - size:]).reshape(size,size,4)
    
    ARRtoWAV(img_arr, npy_name[0:-4], SPL_RTE=SPL_RTE)


#——————————————————————————————————————————————————————————————————————————————#
# This function takes in the name of an .npy file and makes a wav
def NPYtoPNG(npy_name,SPL_RTE=44100):
    if not ('.npy' in npy_name):
        npy_name += '.npy'

    #Convert npy to array
    img_arr = NPYtoARR(npy_name,SPL_RTE=SPL_RTE)
    #Convert Array to png
    ARRtoPNG(img_arr, npy_name[0:-4], SPL_RTE=SPL_RTE)
    

#——————————————————————————————————————————————————————————————————————————————#
# This function takes a 3D array representing a lossless spectrogram, and
# outputs a .wav to the current directory
def ARRtoWAV(img_array,filename,SPL_RTE=44100):
    window_size,num_windows,signal = img_array.shape
    num_freqs = window_size
    window_size *= 2

    #Build 1D mid/side frequency arrays
    mid = np.empty((num_freqs) * num_windows,dtype=np.cdouble)
    sid = np.empty((num_freqs) * num_windows,dtype=np.cdouble)
    for window in range(num_windows):
        for freq in range(num_freqs):
            mid[num_freqs - 1 - freq + window * num_freqs] = \
                img_array[freq,window,0] + img_array[freq,window,1]*1j
            sid[num_freqs - 1 - freq + window * num_freqs] = \
                img_array[freq,window,2] + img_array[freq,window,3]*1j

    #Calculate irfft of each window and load into stereo buffer
    buff = np.empty((window_size*num_windows, 2), dtype=np.float32)
    for window in range(num_windows):
        mid_temp = fft.irfft(np.insert( \
                mid[window*num_freqs:(window+1)*num_freqs],0,0))
        sid_temp = fft.irfft(np.insert( \
                sid[window*num_freqs:(window+1)*num_freqs],0,0))
        
        buff[window*window_size:(window+1)*window_size,0] = \
                mid_temp + sid_temp
        buff[window*window_size:(window+1)*window_size,1] = \
                mid_temp - sid_temp
                
    #Normalize
    buff = buff / ((np.max(buff) - np.min(buff))/2)
    buff -= (np.min(buff) + 1)

    #Save file
    if ".tfrecords" in filename:
        filename = filename[0:-10] + '_fromSpect.wav'
    elif "." in filename:
        filename = filename[0:-4] + '_fromSpect.wav'
    if not ('.wav' in filename):
        filename += '.wav'
        
    sf.write(filename, buff, SPL_RTE)


#——————————————————————————————————————————————————————————————————————————————#
# This function takes the names of two wav files, each as a string,
# and returns their (cosine-like) similarity. If they are of unequal
# lengths, the shorter will be considered as being padded with zeros
def wavSim(w1,w2):
    w1, rate1 = sf.read(w1)
    w2, rate2 = sf.read(w2)

    #Ensure equal sample rate
    if rate1 > rate2:
        w2 = splRteInterpolate(w2,rate2,rate1)
    elif rate1 < rate2:
        w1 = splRteInterpolate(w1,rate1,rate2)
    
    #Get mid/side arrays
    mid1,sid1 = makeMidSide(w1)
    mid2,sid2 = makeMidSide(w2)

    #Match their lengths
    diff = len(mid1) - len(mid2)
    if diff > 0:
        mid2 = np.concatenate(( mid2, np.zeros(diff) ))
        sid2 = np.concatenate(( sid2, np.zeros(diff) ))
    elif diff < 0:
        mid1 = np.concatenate(( mid1, np.zeros(-diff) ))
        sid1 = np.concatenate(( sid1, np.zeros(-diff) ))

    #Prevent divide-by-zero and divergence issues w/r/t blank or mono files
    mach_eps = np.finfo(np.float32).eps
    eps_vect = np.full(len(mid2),mach_eps)
    if np.all(sid1 == 0):
        sid1 = eps_vect
    if np.all(sid2 == 0):
        sid2 = eps_vect
    if np.all(mid1 == 0):
        mid1 = eps_vect
    if np.all(mid2 == 0):
        mid2 = eps_vect

    #Calculate cosine similarities of the mid channels and the side channels
    mid1_norm = np.linalg.norm(mid1)
    mid2_norm = np.linalg.norm(mid2)
    sid1_norm = np.linalg.norm(sid1)
    sid2_norm = np.linalg.norm(sid2)

    mid_sim = np.dot(mid1,mid2) / (mid1_norm * mid2_norm)
    sid_sim = np.dot(sid1,sid2) / (sid1_norm * sid2_norm)

    #Return their euclidean distance scaled to [0,1]
    return np.sqrt((mid_sim**2 + sid_sim**2)/2)

    
