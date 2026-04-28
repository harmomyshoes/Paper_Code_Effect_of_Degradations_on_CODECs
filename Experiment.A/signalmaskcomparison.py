import sys
import os.path
import pandas as pd
import numpy as np
import psychoacoustic as psycho
from common import *
from parameters import *
	
import prototype_filter
import subband_filtering
import scaled_fft
import matplotlib.pyplot as plt


def generate_mask(degradedwavfile,bitrate=64):
    '''Only according to single signal to calculate the MNR and bitallocation'''
    """Compare the MNR (Mask-to-Noise Ratio) values for different bit allocations."""

    input_buffer_degr = WavRead(degradedwavfile)
    params_degr = EncoderParameters(input_buffer_degr.fs, input_buffer_degr.nch, bitrate)
    baseband_filter_degr = prototype_filter.prototype_filter().astype('float32')
    subband_samples_degr = np.zeros((params_degr.nch, N_SUBBANDS, FRAMES_PER_BLOCK), dtype='float32')
    accum_spl_formatted_degr, accum_mask_formatted_degr, accum_gl_mask_formatted_degr, accum_noise_mask_formatted_degr, accum_tone_mask_formatted_degr, accum_smr_formatted_degr= [], [], [], [],[],[]
    block_index = 0

    # Main loop, executing until all samples have been processed.
    while input_buffer_degr.nprocessed_samples < input_buffer_degr.nsamples:
        #print(f"Block {block_index} processed") 
        # In each block 12 frames are processed, which equals 12x32=384 new samples per block.


        ####The subband filtering for the degraded wav file###
        for frm in range(FRAMES_PER_BLOCK):
            samples_read = input_buffer_degr.read_samples(SHIFT_SIZE)
            # If all samples have been read, perform zero padding.
            if samples_read < SHIFT_SIZE:
                for ch in range(params_degr.nch):
                    input_buffer_degr.audio[ch].insert(np.zeros(SHIFT_SIZE - samples_read))

            # Filtering = dot product with reversed buffer.
            """
            Subband filtering
            """
            for ch in range(params_degr.nch):
                subband_samples_degr[ch,:,frm] = subband_filtering.subband_filtering(input_buffer_degr.audio[ch].reversed(), baseband_filter_degr)   
            
        # Declaring arrays for keeping table indices of calculated scalefactors and bits allocated in subbands.
        # Number of bits allocated in subband is either 0 or in range [2,15].
        scfindices_degr = np.zeros((params_degr.nch, N_SUBBANDS), dtype='uint8')


        # Finding scale factors, psychoacoustic model and bit allocation calculation for subbands. Although 
        # scaling is done later, its result is necessary for the psychoacoustic model and calculation of 
        # sound pressure levels.
        for ch in range(params_degr.nch):
            scfindices_degr[ch,:] = get_scalefactors(subband_samples_degr[ch,:,:], params_degr.table.scalefactor)
            #subband_bit_allocation_degr[ch,:],mnr_formatted_degr[ch,:] = psycho.model1(input_buffer_degr.audio[ch].ordered(), params_degr,scfindices_degr)
            table = params_degr.table
            X = scaled_fft.scaled_fft_db(input_buffer_degr.audio[ch].ordered())
            #This X is return the energy dB in (512/2)+1 = 257 bin, in each bin it is has 86Hz range, we asumme its the 16bit, so the 96dB mean the highest energy in thiks bin.    
            #print(f"X is: {X}")  

            scf = table.scalefactor[scfindices_degr]  
            subband_spl = np.zeros(N_SUBBANDS)
            for sb in range(N_SUBBANDS):
                subband_spl[sb] = np.max(X[int(1 + sb * SUB_SIZE): int(1 + sb * SUB_SIZE + SUB_SIZE)])
                subband_spl[sb] = np.maximum(subband_spl[sb], 20 * np.log10(scf[0,sb] * 32768) - 10)
                
            peaks = []
            for i in range(3, FFT_SIZE // 2 - 6):
                if X[i]>=X[i+1] and X[i]>X[i-1]:
                    peaks.append(i)


            #determining tonal and non-tonal components
            tonal = TonalComponents(X)
            tonal.flag[0:3] = IGNORE
            
            for k in peaks:
                # if k== 13:
                #     print("STOP")
                is_tonal = True
                if k > 2 and k < 63:
                    testj = [-2,2]
                elif k >= 63 and k < 127:
                    testj = [-3,-2,2,3]
                else:
                    testj = [-6,-5,-4,-3,-2,2,3,4,5,6]
                for j in testj:
                    if tonal.spl[k] - tonal.spl[k+j] < 7:
                        is_tonal = False
                        break
                if is_tonal:
                    tonal.spl[k] = add_db(tonal.spl[k-1:k+2])
                    tonal.flag[k+np.arange(testj[0], testj[-1] + 1)] = IGNORE
                    tonal.flag[k] = TONE
                    tonal.tonecomps.append(k)
                


            #non-tonal components for each critical band
            ## i is the critical band num, the j is the corresponding FFT bin in the critical band
            for i in range(table.cbnum - 1):
                weight = 0.0
                msum = DBMIN
                for j in range(table.cbound[i], table.cbound[i+1]):
                    if tonal.flag[i] == UNSET:
                        msum = add_db((tonal.spl[j], msum))
                        weight += np.power(10, tonal.spl[j] / 10) * (table.bark[table.map[j]] - i)
                if msum > DBMIN:
                    index  = weight/np.power(10, msum / 10.0)
                    center = table.cbound[i] + int(index * (table.cbound[i+1] - table.cbound[i])) 
                    if tonal.flag[center] == TONE:
                        center += 1
                    tonal.flag[center] = NOISE
                    tonal.spl[center] = msum
                    tonal.noisecomps.append(center)
                
            
            #decimation of tonal and non-tonal components
            #under the threshold in quiet
            for i in range(len(tonal.tonecomps)):
                if i >= len(tonal.tonecomps):
                    break
                k = tonal.tonecomps[i]
                if tonal.spl[k] < table.hear[table.map[k]]:
                    tonal.tonecomps.pop(i)
                    tonal.flag[k] = IGNORE
                    i -= 1

            for i in range(len(tonal.noisecomps)):
                if i >= len(tonal.noisecomps):
                    break
                k = tonal.noisecomps[i]
                if tonal.spl[k] < table.hear[table.map[k]]:
                    tonal.noisecomps.pop(i)
                    tonal.flag[k] = IGNORE
                    i -= 1


            #decimation of tonal components closer than 0.5 Bark
            for i in range(len(tonal.tonecomps) -1 ):
                if i >= len(tonal.tonecomps) -1:
                    break
                this = tonal.tonecomps[i]
                next = tonal.tonecomps[i+1]
                if table.bark[table.map[this]] - table.bark[table.map[next]] < 0.5:
                    if tonal.spl[this]>tonal.spl[next]:
                        tonal.flag[next] = IGNORE
                        tonal.tonecomps.remove(next)
                    else:
                        tonal.flag[this] = IGNORE
                        tonal.tonecomps.remove(this)

            

            #individual masking thresholds
            masking_tonal = []
            masking_noise = []

            for i in range(table.subsize):
                masking_tonal.append(())
                zi = table.bark[i]
                for j in tonal.tonecomps:
                    zj = table.bark[table.map[j]]
                    dz = zi - zj
                    if dz >= -3 and dz <= 8:
                        avtm = -1.525 - 0.275 * zj - 4.5
                        if dz >= -3 and dz < -1:
                            vf = 17 * (dz + 1) - (0.4 * X[j] + 6)
                        elif dz >= -1 and dz < 0:
                            vf = dz * (0.4 * X[j] + 6)
                        elif dz >= 0 and dz < 1:
                            vf = -17 * dz
                        else:
                            vf = -(dz - 1) * (17 - 0.15 * X[j]) - 17
                        masking_tonal[i] += (X[j] + vf + avtm,)

            for i in range(table.subsize):
                masking_noise.append(())
                zi = table.bark[i]
                for j in tonal.noisecomps:
                    zj = table.bark[table.map[j]]
                    dz = zi - zj
                    if dz >= -3 and dz <= 8:
                        avnm = -1.525 - 0.175 * zj - 0.5
                        if dz >= -3 and dz < -1:
                            vf = 17 * (dz + 1) - (0.4 * X[j] + 6)
                        elif dz >= -1 and dz < 0:
                            vf = dz * (0.4 * X[j] + 6)
                        elif dz >= 0 and dz < 1:
                            vf = -17 * dz
                        else:
                            vf = -(dz - 1) * (17 - 0.15 * X[j]) - 17
                        masking_noise[i] += (X[j] + vf + avnm,)
                        # if i == 47:
                        #     print(f"The masking from nontonal is index {j}, contribution with direct SPL is: {X[j]}, spread is: {vf}, and the offset bias is: {avnm} dB")


            #global masking thresholds
            masking_global,masking_noise_db,masking_tonal_db = [], [], []

            for i in range(table.subsize):
                maskers = (table.hear[i],) + masking_tonal[i] + masking_noise[i]
                masking_global.append(add_db(maskers))
                masking_noise_db.append(add_db(masking_noise[i]))
                masking_tonal_db.append(add_db(masking_tonal[i]))



            #minimum masking thresholds
            mask = np.zeros(N_SUBBANDS)
            for sb in range(N_SUBBANDS):
                first = table.map[int(sb * SUB_SIZE)]
                after_last  = table.map[int((sb + 1) * SUB_SIZE - 1)] + 1
                mask[sb] = np.min(masking_global[first:after_last])


            #signal-to-mask ratio for each subband
        smr = subband_spl - mask
        np.set_printoptions(suppress=True, precision=2)
        #print(f"The subband_spl is: {subband_spl}")
        #print(f"The mask is: {mask}")
        accum_spl_formatted_degr.append(subband_spl)
        accum_mask_formatted_degr.append(mask)
        accum_gl_mask_formatted_degr.append(masking_global)
        accum_noise_mask_formatted_degr.append(masking_noise_db)
        accum_tone_mask_formatted_degr.append(masking_tonal_db)
        accum_smr_formatted_degr.append(smr)

        block_index = block_index + 1
        
    #print (f"The bit allocation of original wav file is: \n{subband_bit_allocation_orig}\n The bit allocation of degraded wav file is: \n{subband_bit_allocation_degr}")

    acuum_spl_orig = np.array(accum_spl_formatted_degr)
    acuum_mask_degr = np.array(accum_mask_formatted_degr)
    accum_global_mask = np.array(accum_gl_mask_formatted_degr)
    accum_noise_mask = np.array(accum_noise_mask_formatted_degr)
    accum_tone_mask = np.array(accum_tone_mask_formatted_degr)
    acuum_smr_orig = np.array(accum_smr_formatted_degr)


    return acuum_spl_orig, accum_global_mask, accum_noise_mask, accum_tone_mask, acuum_mask_degr, acuum_smr_orig, block_index


def plot_spl_mask(referencewavfile, degradedwavfile, suptitle="SPL and Mask Comparison"):
    acuum_spl_orig_ref, accum_global_mask_ref, accum_noise_mask_ref, accum_tone_mask_ref, acuum_mask_degr_ref, acuum_smr_orig_ref, _ = generate_mask(referencewavfile)
    acuum_spl_orig_deg, accum_global_mask_deg, accum_noise_mask_deg, accum_tone_mask_deg, acuum_mask_degr_deg, acuum_smr_orig_deg, _ = generate_mask(degradedwavfile)
    # Dummy data shapes for demonstration
    data_dict = {
        "SPL": (acuum_spl_orig_ref, acuum_spl_orig_deg),
        "Mask": (acuum_mask_degr_ref, acuum_mask_degr_deg),
#        "SMR": (acuum_smr_orig_ref, acuum_smr_orig_deg),
    }

    subbands = np.arange(32)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex = True)
    axes = axes.flatten()

    for i, (title, (data1, data2)) in enumerate(data_dict.items()):
        mean1 = np.mean(data1, axis=0)
        std1 = np.std(data1, axis=0)
        mean2 = np.mean(data2, axis=0)
        std2 = np.std(data2, axis=0)

        axes[i].errorbar(subbands, mean1, yerr=std1, label='Ref', fmt='-o', capsize=4)
        axes[i].errorbar(subbands, mean2, yerr=std2, label='Deg', fmt='--x', capsize=1)
        axes[i].set_title(f"{title} - Mean ± Std Dev", fontsize=12)
        axes[i].set_xlabel("Subband Index")
        axes[i].set_ylabel("Value (dB)")
        axes[i].grid(True)
        axes[i].legend()

    # Hide unused subplot
    # fig.delaxes(axes[-1])
    plt.suptitle(suptitle, fontsize=16, y=1.01)
    #plt.subplots_adjust(top=0.9)  # Adjust the top to make room for the title
    plt.tight_layout()
    plt.show()

def plot_spl_mask_single(referencewavfile, suptitle="SPL and Mask Comparison"):
    acuum_spl_orig_ref, accum_global_mask_ref, accum_noise_mask_ref, accum_tone_mask_ref, acuum_mask_degr_ref, acuum_smr_orig_ref, _ = generate_mask(referencewavfile)
    # Dummy data shapes for demonstration
    data_dict = {
        "SPL": (acuum_spl_orig_ref),
        "Mask": (acuum_mask_degr_ref),
#        "SMR": (acuum_smr_orig_ref, acuum_smr_orig_deg),
    }

    subbands = np.arange(32)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex = True)
    axes = axes.flatten()

    for i, (title, (data1)) in enumerate(data_dict.items()):
        mean1 = np.mean(data1, axis=0)
        std1 = np.std(data1, axis=0)
     

        axes[i].errorbar(subbands, mean1, yerr=std1, label='Ref', fmt='-o', capsize=4)
        axes[i].set_title(f"{title} - Mean ± Std Dev", fontsize=12)
        axes[i].set_xlabel("Subband Index")
        axes[i].set_ylabel("Value (dB)")
        axes[i].grid(True)
        axes[i].legend()

    # Hide unused subplot
    # fig.delaxes(axes[-1])
    plt.suptitle(suptitle, fontsize=16, y=1.01)
    #plt.subplots_adjust(top=0.9)  # Adjust the top to make room for the title
    plt.tight_layout()
    plt.show()

def plot_detail_mask(referencewavfile, degradedwavfile, suptitle="Mask Comparison"):
    # Dummy data shapes for demonstration
    acuum_spl_orig_ref, accum_global_mask_ref, accum_noise_mask_ref, accum_tone_mask_ref, acuum_mask_degr_ref, acuum_smr_orig_ref, _ = generate_mask(referencewavfile)
    acuum_spl_orig_deg, accum_global_mask_deg, accum_noise_mask_deg, accum_tone_mask_deg, acuum_mask_degr_deg, acuum_smr_orig_deg, _ = generate_mask(degradedwavfile)
    data_dict = {
        "Noise_M": (accum_noise_mask_ref, accum_noise_mask_deg),
        "Tone_M": (accum_tone_mask_ref, accum_tone_mask_deg),
        "Global_M": (accum_global_mask_ref, accum_global_mask_deg),
    }

    subbands = np.arange(102)
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex = True)
    axes = axes.flatten()

    for i, (title, (data1, data2)) in enumerate(data_dict.items()):
        mean1 = np.mean(data1, axis=0)
        std1 = np.std(data1, axis=0)
        mean2 = np.mean(data2, axis=0)
        std2 = np.std(data2, axis=0)

        axes[i].errorbar(subbands, mean1, yerr=std1, label='Ref', fmt='-o', capsize=4)
        axes[i].errorbar(subbands, mean2, yerr=std2, label='Deg', fmt='--x', capsize=1)
        axes[i].set_title(f"{title} - Mean ± Std Dev", fontsize=12)
        axes[i].set_xlabel("Subband Index")
        axes[i].set_ylabel("Value (dB)")
        axes[i].grid(True)
        axes[i].legend()

    # Hide unused subplot
    # fig.delaxes(axes[-1])
    plt.suptitle(suptitle, fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()

def deviation_mask(referencewavfile, degradedwavfile):
    _, _, _, _, acuum_mask_degr_ref, _, block_index_r = generate_mask(referencewavfile)
    _, _, _, _, acuum_mask_degr_deg, _, block_index_d = generate_mask(degradedwavfile)
    if block_index_r != block_index_d:
        raise ValueError("The number of blocks in the reference and degraded files must be the same.")
        return 0
    else:
        # Calculate the total mask for each block then retrun the delta per frame
        # total_mask_orig = np.array(accum_global_mask_ref, dtype='int16')
        total_mask_orig = np.array(acuum_mask_degr_ref)
        total_mask_degr = np.array(acuum_mask_degr_deg)
        total_mask_delta_per_frame =  np.sum(np.abs(total_mask_degr-total_mask_orig))/block_index_r

        return total_mask_delta_per_frame

    


class TonalComponents:
    """Marking of tonal and non-tonal components in the psychoacoustic model."""
    def __init__(self, X):
        self.spl = np.copy(X)
        self.flag = np.zeros(X.size, dtype='uint8')
        self.tonecomps  = []
        self.noisecomps = []