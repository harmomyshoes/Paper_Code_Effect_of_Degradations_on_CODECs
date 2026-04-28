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


def compare_mnr_bitallocation(originalwavfile,degradedwavfile,bitrate):
    """Compare the MNR (Mask-to-Noise Ratio) values for different bit allocations."""
    """Because the purpose of this code is not to encode audio files, but to compare the bitallocation strategy,and
    finally the MNR values, so we don't need to generate the the mp3 audio files."""
    input_buffer_orig = WavRead(originalwavfile)
    params_orig = EncoderParameters(input_buffer_orig.fs, input_buffer_orig.nch, bitrate)
    baseband_filter_orig = prototype_filter.prototype_filter().astype('float32')
    subband_samples_orig = np.zeros((params_orig.nch, N_SUBBANDS, FRAMES_PER_BLOCK), dtype='float32')

    input_buffer_degr = WavRead(degradedwavfile)
    params_degr = EncoderParameters(input_buffer_degr.fs, input_buffer_degr.nch, bitrate)
    baseband_filter_degr = prototype_filter.prototype_filter().astype('float32')
    subband_samples_degr = np.zeros((params_degr.nch, N_SUBBANDS, FRAMES_PER_BLOCK), dtype='float32')

    if input_buffer_orig.nsamples != input_buffer_degr.nsamples:
        sys.exit("The two wav files have different number of samples.")

    block_index = 0
    accum_subband_bit_allocation_orig, accum_subband_bit_allocation_degr, accum_mnr_formatted_orig, accum_mnr_formatted_degr = [], [], [], []

    # Main loop, executing until all samples have been processed.
    while input_buffer_orig.nprocessed_samples < input_buffer_orig.nsamples:
        #print(f"Block {block_index} processed") 
        # In each block 12 frames are processed, which equals 12x32=384 new samples per block.

        ####The subband filtering for the original wav file###
        for frm in range(FRAMES_PER_BLOCK):
            samples_read = input_buffer_orig.read_samples(SHIFT_SIZE)
            # If all samples have been read, perform zero padding.
            if samples_read < SHIFT_SIZE:
                for ch in range(params_orig.nch):
                    input_buffer_orig.audio[ch].insert(np.zeros(SHIFT_SIZE - samples_read))

            # Filtering = dot product with reversed buffer.
            """
            Subband filtering
            """
            for ch in range(params_orig.nch):
                subband_samples_orig[ch,:,frm] = subband_filtering.subband_filtering(input_buffer_orig.audio[ch].reversed(), baseband_filter_orig)

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
        scfindices_orig = np.zeros((params_orig.nch, N_SUBBANDS), dtype='uint8')
        scfindices_degr = np.zeros((params_degr.nch, N_SUBBANDS), dtype='uint8')
        subband_bit_allocation_orig = np.zeros((params_orig.nch, N_SUBBANDS), dtype='uint8')
        subband_bit_allocation_degr = np.zeros((params_degr.nch, N_SUBBANDS), dtype='uint8')
        mnr_formatted_orig = np.zeros((params_orig.nch, N_SUBBANDS), dtype='float32')
        mnr_formatted_degr = np.zeros((params_degr.nch, N_SUBBANDS), dtype='float32')



        # Finding scale factors, psychoacoustic model and bit allocation calculation for subbands. Although 
        # scaling is done later, its result is necessary for the psychoacoustic model and calculation of 
        # sound pressure levels.
        for ch in range(params_orig.nch):
            scfindices_orig[ch,:] = get_scalefactors(subband_samples_orig[ch,:,:], params_orig.table.scalefactor)
            subband_bit_allocation_orig[ch,:],mnr_formatted_orig[ch,:] = psycho.model1(input_buffer_orig.audio[ch].ordered(), params_orig,scfindices_orig)

        for ch in range(params_degr.nch):
            scfindices_degr[ch,:] = get_scalefactors(subband_samples_degr[ch,:,:], params_degr.table.scalefactor)
            subband_bit_allocation_degr[ch,:],mnr_formatted_degr[ch,:] = psycho.model1(input_buffer_degr.audio[ch].ordered(), params_degr,scfindices_degr)

        #print (f"The MNRatio Array of original wav file is: \n{mnr_formatted_orig}\nThe MNRatio Array of degraded wav file is: \n{mnr_formatted_degr}")
        accum_mnr_formatted_orig.append(mnr_formatted_orig)
        accum_mnr_formatted_degr.append(mnr_formatted_degr)
        #print (f"The bit allocation of original wav file is: \n{subband_bit_allocation_orig}\n The bit allocation of degraded wav file is: \n{subband_bit_allocation_degr}")
        accum_subband_bit_allocation_orig.append(subband_bit_allocation_orig)
        accum_subband_bit_allocation_degr.append(subband_bit_allocation_degr)
        block_index = block_index + 1 


    acuum_mnr_orig = np.array(accum_mnr_formatted_orig)
    acuum_mnr_degr = np.array(accum_mnr_formatted_degr)
    total_mnr_orig = np.sum(np.minimum(0.0, acuum_mnr_orig))
    total_mnr_degr = np.sum(np.minimum(0.0, acuum_mnr_degr))
    total_mnr_delta_per_frame = (total_mnr_degr-total_mnr_orig)/block_index
    
    #print(f"The Result of Comparison on strggling metrics on average {block_index} frames are: \n") 
    #print(f"The total MNR of original signal is: {total_mnr_orig} dB\n")
    #print(f"The total MNR of degraded signal is: {total_mnr_degr} dB\n")

    bit_allocation_orig = np.array(accum_subband_bit_allocation_orig, dtype='int16')
    bit_allocation_degr = np.array(accum_subband_bit_allocation_degr, dtype='int16')
    bit_allocation_delta_per_frame =  np.sum(np.abs(bit_allocation_degr-bit_allocation_orig))/block_index

    #print(f"The Delta of the bitallocation is : {bit_allocation_delta_per_frame}\n")
    return acuum_mnr_orig, acuum_mnr_degr, total_mnr_orig, total_mnr_degr,total_mnr_delta_per_frame, bit_allocation_orig, bit_allocation_degr, bit_allocation_delta_per_frame


def single_mnr_bitallocation(degradedwavfile,bitrate):
    '''Only according to single signal to calculate the MNR and bitallocation'''
    """Compare the MNR (Mask-to-Noise Ratio) values for different bit allocations.
    To be notification return the MNR and bitallocation of the degraded wav file,the TOTAL MNR NOT DIVIDE FRAME SIZE"""

    input_buffer_degr = WavRead(degradedwavfile)
    params_degr = EncoderParameters(input_buffer_degr.fs, input_buffer_degr.nch, bitrate)
    baseband_filter_degr = prototype_filter.prototype_filter().astype('float32')
    subband_samples_degr = np.zeros((params_degr.nch, N_SUBBANDS, FRAMES_PER_BLOCK), dtype='float32')


    block_index = 0
    accum_subband_bit_allocation_degr, accum_mnr_formatted_degr = [], []

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
        subband_bit_allocation_degr = np.zeros((params_degr.nch, N_SUBBANDS), dtype='uint8')
        mnr_formatted_degr = np.zeros((params_degr.nch, N_SUBBANDS), dtype='float32')



        # Finding scale factors, psychoacoustic model and bit allocation calculation for subbands. Although 
        # scaling is done later, its result is necessary for the psychoacoustic model and calculation of 
        # sound pressure levels.
        for ch in range(params_degr.nch):
            scfindices_degr[ch,:] = get_scalefactors(subband_samples_degr[ch,:,:], params_degr.table.scalefactor)
            subband_bit_allocation_degr[ch,:],mnr_formatted_degr[ch,:] = psycho.model1(input_buffer_degr.audio[ch].ordered(), params_degr,scfindices_degr)

        #print (f"The MNRatio Array of original wav file is: \n{mnr_formatted_orig}\nThe MNRatio Array of degraded wav file is: \n{mnr_formatted_degr}")
        accum_mnr_formatted_degr.append(mnr_formatted_degr)
        #print (f"The bit allocation of original wav file is: \n{subband_bit_allocation_orig}\n The bit allocation of degraded wav file is: \n{subband_bit_allocation_degr}")
        accum_subband_bit_allocation_degr.append(subband_bit_allocation_degr)
        block_index = block_index + 1 



    acuum_mnr_degr = np.array(accum_mnr_formatted_degr)
    total_mnr_degr = np.sum(np.minimum(0.0, acuum_mnr_degr))
    
 #   print(f"The Result of Comparison on strggling metrics on average {block_index} frames are: \n") 
 #   print(f"The total MNR of degraded signal is: {total_mnr_degr} dB\n")

    bit_allocation_degr = np.array(accum_subband_bit_allocation_degr, dtype='int16')

#    print(f"The bitallocation is : {bit_allocation_degr}\n")
    return  acuum_mnr_degr, total_mnr_degr, bit_allocation_degr


def single_sample_scale_FFT(degradedwavfile,bitrate):
    input_buffer_degr = WavRead(degradedwavfile)
    params_degr = EncoderParameters(input_buffer_degr.fs, input_buffer_degr.nch, bitrate)
    baseband_filter_degr = prototype_filter.prototype_filter().astype('float32')
    subband_samples_degr = np.zeros((params_degr.nch, N_SUBBANDS, FRAMES_PER_BLOCK), dtype='float32')


    block_index = 0
    accum_X = []

    # Main loop, executing until all samples have been processed.
    while input_buffer_degr.nprocessed_samples < input_buffer_degr.nsamples:
        print(f"Block {block_index} processed") 
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
        table = params_degr.table



        # Finding scale factors, psychoacoustic model and bit allocation calculation for subbands. Although 
        # scaling is done later, its result is necessary for the psychoacoustic model and calculation of 
        # sound pressure levels.
        for ch in range(params_degr.nch):
            scfindices_degr[ch,:] = get_scalefactors(subband_samples_degr[ch,:,:], params_degr.table.scalefactor)
            X = scaled_fft.scaled_fft_db(input_buffer_degr.audio[ch].ordered())
            print(f"The X is: {X}")
#            plotFFT(X)
        block_index = block_index + 1
        accum_X.append(X)
    return accum_X 
            
def plotFFT(spectrum_db):
    plt.figure(figsize=(12, 2))
    plt.plot(np.arange(257), spectrum_db)
    plt.title("Magnitude Spectrum (Normalized to -96 dB)")
    plt.xlabel("FFT Bin Index")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.show()

def plotbitallocation(bit_allocation, title):
    bit_allocation = bit_allocation[:, 0, :]
    plt.figure(figsize=(12, 2))
    plt.imshow(bit_allocation, aspect='auto', cmap='gray',
                     vmin=bit_allocation.min(), vmax=bit_allocation.max())
    plt.title(title)
    plt.xlabel("Subband")   
    plt.ylabel("Frame")
    plt.colorbar(label="Bit Allocation")
    plt.show()

def genertate_spl_mask_smr_mnr_bit_xlsx(acuum_spl, acuum_mask, acuum_smr,acuum_mnr, bit_allocation,folder, filename):
    # ── REPLACE THESE with your actual arrays ─────────────────────────────────
# data_orig:  NumPy array shape (n_frames, n_subbands)
# data_degr:  NumPy array shape (n_frames, n_subbands)
    data_spl = acuum_spl
    data_mask = acuum_mask
    data_smr = acuum_smr
    data_mnr = acuum_mnr[:, 0, :]
    data_bit_all = bit_allocation[:, 0, :]
# ────────────────────────────────────────────────────────────────────────────

    n_frames, n_subbands = data_spl.shape

    # Column labels
    cols = [f"SB{sub:02d}" for sub in range(n_subbands)]

    # Row MultiIndex: (Frame 1, Orig), (Frame 1, Degr), (Frame 2, Orig), …
    frames = [f"Frame {i+1}" for i in range(n_frames)]
    row_tuples = [(f, tag) for f in frames for tag in ("SPL","MASK", "SMR","MNR","Bit")]
    row_index = pd.MultiIndex.from_tuples(row_tuples, names=["Frame", "Type"])

    # Stack the arrays so that each frame contributes two rows (Orig then Degr)
    combined = np.vstack([
        np.vstack((data_spl[i], data_mask[i], data_smr[i],data_mnr[i], data_bit_all[i]))
        for i in range(n_frames)
    ])

    df = pd.DataFrame(combined, index=row_index, columns=cols)


    # Display in Jupyter / IPython
    # with pd.option_context(
    #     'display.max_rows', None,
    #     'display.max_columns', None,
    #     'display.width', 2000,
    #     'display.max_colwidth', None
    # ):
    #     display(df)

    df.to_excel(f'{folder}/{filename}.xlsx', index=True, header=True)
    

def single_spl_mask_smr(degradedwavfile,bitrate):
    '''Only according to single signal to calculate the MNR and bitallocation'''
    """Compare the MNR (Mask-to-Noise Ratio) values for different bit allocations."""

    input_buffer_degr = WavRead(degradedwavfile)
    params_degr = EncoderParameters(input_buffer_degr.fs, input_buffer_degr.nch, bitrate)
    baseband_filter_degr = prototype_filter.prototype_filter().astype('float32')
    subband_samples_degr = np.zeros((params_degr.nch, N_SUBBANDS, FRAMES_PER_BLOCK), dtype='float32')
    accum_spl_formatted_degr, accum_mask_formatted_degr,accum_smr_formatted_degr = [], [], []

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
            for i in range(table.cbnum - 1):
                weight = 0.0
                msum = DBMIN
                for j in range(table.cbound[i], table.cbound[i+1]):
                    if tonal.flag[i] == UNSET:
                        msum = add_db((tonal.spl[j], msum))
                        weight += np.power(10, tonal.spl[j] / 10) * (table.bark[table.map[j]] - i)
                if msum > DBMIN:
                    index  = weight/np.power(10, msum / 10.0)
                    center = table.cbound[i] + np.int(index * (table.cbound[i+1] - table.cbound[i])) 
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


            #global masking thresholds
            masking_global = []
            for i in range(table.subsize):
                maskers = (table.hear[i],) + masking_tonal[i] + masking_noise[i]
                masking_global.append(add_db(maskers))


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
        accum_smr_formatted_degr.append(smr)
        block_index = block_index + 1 
    #print (f"The bit allocation of original wav file is: \n{subband_bit_allocation_orig}\n The bit allocation of degraded wav file is: \n{subband_bit_allocation_degr}")

    acuum_spl_orig = np.array(accum_spl_formatted_degr)
    acuum_mask_degr = np.array(accum_mask_formatted_degr)
    acuum_smr_degr = np.array(accum_smr_formatted_degr)
    return acuum_spl_orig, acuum_mask_degr, acuum_smr_degr

def get_peak_tonal_nontonal(degradedwavfile,bitrate):
    input_buffer_degr = WavRead(degradedwavfile)
    params_degr = EncoderParameters(input_buffer_degr.fs, input_buffer_degr.nch, bitrate)
    baseband_filter_degr = prototype_filter.prototype_filter().astype('float32')
    subband_samples_degr = np.zeros((params_degr.nch, N_SUBBANDS, FRAMES_PER_BLOCK), dtype='float32')
    accum_peak, accum_masking_tone, accum_masking_noise = [],[],[]
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
            for i in range(table.cbnum - 1):
                weight = 0.0
                msum = DBMIN
                for j in range(table.cbound[i], table.cbound[i+1]):
                    if tonal.flag[i] == UNSET:
                        msum = add_db((tonal.spl[j], msum))
                        weight += np.power(10, tonal.spl[j] / 10) * (table.bark[table.map[j]] - i)
                if msum > DBMIN:
                    index  = weight/np.power(10, msum / 10.0)
                    center = table.cbound[i] + np.int(index * (table.cbound[i+1] - table.cbound[i])) 
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


            accum_masking_tone.append(masking_tonal)
            accum_masking_noise.append(masking_noise)
            #global masking thresholds
            masking_global = []
            maskers_46, maskers_47 = [], []
            for i in range(table.subsize):
                maskers = (table.hear[i],) + masking_tonal[i] + masking_noise[i]
                if i == 46:
                    maskers_46.append(maskers)
                elif i == 47:
                    maskers_47.append(maskers)
                masking_global.append(add_db(maskers))



            #minimum masking thresholds
            mask = np.zeros(N_SUBBANDS)
            for sb in range(N_SUBBANDS):
                first = table.map[int(sb * SUB_SIZE)]
                after_last  = table.map[int((sb + 1) * SUB_SIZE - 1)] + 1
                mask[sb] = np.min(masking_global[first:after_last])


            #signal-to-mask ratio for each subband
        #smr = subband_spl - mask
        np.set_printoptions(suppress=True, precision=2)
        #print(f"The subband_spl is: {subband_spl}")
        #print(f"The mask is: {mask}")
        block_index = block_index + 1
        # accum_peak.append(peaks)
    #print (f"The bit allocation of original wav file is: \n{subband_bit_allocation_orig}\n The bit allocation of degraded wav file is: \n{subband_bit_allocation_degr}")

    #acuum_smr_degr = np.array(accum_smr_formatted_degr)
    return accum_peak

def test_mask(degradedwavfile,bitrate):
    '''Only according to single signal to calculate the MNR and bitallocation'''
    """Compare the MNR (Mask-to-Noise Ratio) values for different bit allocations."""

    input_buffer_degr = WavRead(degradedwavfile)
    params_degr = EncoderParameters(input_buffer_degr.fs, input_buffer_degr.nch, bitrate)
    baseband_filter_degr = prototype_filter.prototype_filter().astype('float32')
    subband_samples_degr = np.zeros((params_degr.nch, N_SUBBANDS, FRAMES_PER_BLOCK), dtype='float32')
    accum_spl_formatted_degr, accum_mask_formatted_degr,accum_smr_formatted_degr,accum_gl_mask_formatted_degr, accum_46_mask_formatted_degr,accum_47_mask_formatted_degr = [], [], [], [], [], []
    accum_tonal_noise_comp, accum_tonal_tone_comp = [],[]
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
                if k== 13:
                    print("STOP")
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
                    if j == 14:
                        print("STOP What is the center bin?")
                    if tonal.flag[i] == UNSET:
                        msum = add_db((tonal.spl[j], msum))
                        weight += np.power(10, tonal.spl[j] / 10) * (table.bark[table.map[j]] - i)
                if msum > DBMIN:
                    index  = weight/np.power(10, msum / 10.0)
                    center = table.cbound[i] + np.int(index * (table.cbound[i+1] - table.cbound[i])) 
                    if (center == 14):
                        print("STOP WHY this is noise?")
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
                        if i == 47:
                            print(f"The masking from nontonal is index {j}, contribution with direct SPL is: {X[j]}, spread is: {vf}, and the offset bias is: {avnm} dB")


            #global masking thresholds
            masking_global = []
            maskers_46, maskers_47 = [], []
            for i in range(table.subsize):
                maskers = (table.hear[i],) + masking_tonal[i] + masking_noise[i]
                if i == 46:
                    maskers_46.append(maskers)
                elif i == 47:
                    maskers_47.append(maskers)
                masking_global.append(add_db(maskers))



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
        accum_smr_formatted_degr.append(smr)
        block_index = block_index + 1
        accum_46_mask_formatted_degr.append(maskers_46)
        accum_47_mask_formatted_degr.append(maskers_47) 
        accum_tonal_noise_comp.append(tonal.noisecomps)
        accum_tonal_tone_comp.append(tonal.tonecomps)
        
    #print (f"The bit allocation of original wav file is: \n{subband_bit_allocation_orig}\n The bit allocation of degraded wav file is: \n{subband_bit_allocation_degr}")

    acuum_spl_orig = np.array(accum_spl_formatted_degr)
    acuum_mask_degr = np.array(accum_mask_formatted_degr)
    accum_global_mask = np.array(accum_gl_mask_formatted_degr)
    acuum_smr_degr = np.array(accum_smr_formatted_degr)
    return acuum_spl_orig, accum_global_mask, acuum_mask_degr, acuum_smr_degr,accum_46_mask_formatted_degr, accum_47_mask_formatted_degr,accum_tonal_noise_comp,accum_tonal_tone_comp

def return_X_onWav(degradedwavfile,bitrate):
    '''Only according to single signal to calculate the MNR and bitallocation'''
    """Compare the MNR (Mask-to-Noise Ratio) values for different bit allocations."""

    input_buffer_degr = WavRead(degradedwavfile)
    params_degr = EncoderParameters(input_buffer_degr.fs, input_buffer_degr.nch, bitrate)
    baseband_filter_degr = prototype_filter.prototype_filter().astype('float32')
    subband_samples_degr = np.zeros((params_degr.nch, N_SUBBANDS, FRAMES_PER_BLOCK), dtype='float32')
    X_list = []

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
            X_list.append(X) 
    return X_list



class TonalComponents:
  """Marking of tonal and non-tonal components in the psychoacoustic model."""
  
  def __init__(self, X):
    self.spl = np.copy(X)
    self.flag = np.zeros(X.size, dtype='uint8')
    self.tonecomps  = []
    self.noisecomps = []