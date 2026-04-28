# debug_runner.py
import signalcomparison as sc  # Import your module
import signalmaskcomparison as smc

import sys
sys.path.append('/home/codecrack/Jnotebook/')
from CODECbreakCode import Evaluator

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # Now relative paths will work from here
if __name__ == "__main__":
    # Optionally set a breakpoint before running main
    # You can also insert a breakpoint inside encoder.main as described above.
#    encoder.main("/home/codecrack/Mp3_Encoder/samples/sinewave_500ms.wav", "/home/codecrack/Mp3_Encoder/samples/sinewave_500ms.mp3", 64)
#    sc.compare_mnr_bitallocation("/home/codecrack/Mp3_Encoder/samples/TestTone_15Hz_0.2s.wav","/home/codecrack/Mp3_Encoder/samples/TestTone_24000Hz_0.2s.wav",64)
#    sc.single_mnr_bitallocation("/home/codecrack/Mp3_Encoder/samples/TestTone_1200Hz_0.2s.wav",64)
#    sc.single_mnr_bitallocation("/home/codecrack/Mp3_Encoder/samples/TestTone_24000Hz_0.2s.wav",64)
#    Evaluator.GeneratingMP3RefFile('/home/codecrack/Mp3_Encoder/samples/1.1testtone_sweep/', 'TestTone_100Hz_0.2s.wav', 64)
#    a, b, c = Evaluator.MeasureNMRDelta('/home/codecrack/Mp3_Encoder/samples/1.1testtone_sweep/', 'TestTone_100Hz_0.2s.wav', 'TestTone_100Hz_0.2s.wav', 64) 
#  

    # sc.single_spl_mask_smr('/home/codecrack/Mp3_Encoder/samples/Extra.Roughness/TestTone_300_1100_1200Hz_0.2s.wav',64)
    # _, accum_global_mask_rough, _, _,_,_,_,_ =      sc.test_mask('/home/codecrack/Mp3_Encoder/samples/Extra.Roughness/TestTone_300_1100_1200Hz_0.2s.wav',64)
    # _, accum_global_mask_unrough, _, _,_,_,_,_ =      sc.test_mask('/home/codecrack/Mp3_Encoder/samples/Extra.Roughness/TestTone_300_800_1200Hz_0.2s.wav',64)
    #accum_peak = sc.get_peak_tonal_nontonal('/home/codecrack/Mp3_Encoder/samples/Extra.Roughness/TestTone_300_1100_1200Hz_0.2s.wav',64)
    # output_fold = '/home/codecrack/Mp3_Encoder/samples/1.1testtone_sweep/'
    # refrfile = 'TestTone_100Hz_0.2s.wav'
    # refrfullpath = f"{output_fold}/{refrfile}"
    # degrfile = 'TestTone_13100Hz_0.2s.wav'
    # degrfilepath = f"{output_fold}/{degrfile}"
    # acuum_mnr_orig, acuum_mnr_degr, total_mnr_orig, total_mnr_degr,total_mnr_delta_per_frame, bit_allocation_orig, bit_allocation_degr, bit_allocation_delta_per_frame = sc.compare_mnr_bitallocation(refrfullpath,degrfilepath,64)
    # print(f"Accumulated MNR Original: {acuum_mnr_orig}")
    plot_spl_mask = smc.plot_spl_mask('/home/codecrack/Mp3_Encoder/samples/Extra.Roughness/TestTone_300_1100_1200Hz_0.2s.wav','/home/codecrack/Mp3_Encoder/samples/Extra.Roughness/TestTone_300_1100_1200Hz_0.2s.wav',64)   