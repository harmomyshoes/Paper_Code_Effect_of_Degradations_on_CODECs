import os
import shutil
from CODECbreakCode.AudioMixer import FullTrackAudioMixer
import CODECbreakCode.Evaluator as Evaluator
from CODECbreakCode.Evaluator import MeasureHAAQIOutput

class ProjectManager:
    """Manages multiple audio projects for evaluation"""
    
    def __init__(self, base_dir, project_names, solution=[0]*28):
        self.base_dir = base_dir
        self.projects = {}
        self.reference_files = {}
        self.reference_mp3_files = {}
        self.haaqi_measurers = {}  # Store HAAQI measurers per project
        self.solution = solution
        
        # Initialize each project
        for project_name in project_names:
            self._init_project(project_name)
    
    def _init_project(self, project_name):
        """Initialize a single project"""
        mixing_path = os.path.join(self.base_dir, project_name)
        print(f"Initialized {project_name}:")        
        if not os.path.exists(mixing_path):
            print(f"Warning: Path {mixing_path} does not exist. Skipping {project_name}.")
            return
        
        # Create noise generator for this project
        noise_generator = FullTrackAudioMixer(mixing_path)
        self.projects[project_name] = noise_generator
        
        # Generate reference files
        ref_wav = noise_generator.TestDynNoisedFullTrack(
            self.solution, 
            f"Reference_{project_name}_FULL.wav",
            isNormalised=False,
            isCompensated=True
        )
        self.reference_files[project_name] = ref_wav
        
        # Compress to MP3
        ref_mp3 = Evaluator.Mp3LameLossyCompress(ref_wav, 64)
        self.reference_mp3_files[project_name] = ref_mp3
        print(f"  Reference WAV: {ref_wav}")
        print(f"  Reference MP3: {ref_mp3}")        
        # Initialize HAAQI measurer with the reference MP3
        haaqi_measurer = MeasureHAAQIOutput(ref_mp3)
        self.haaqi_measurers[project_name] = haaqi_measurer
        
        # Test HAAQI on itself (should be close to 1.0 or perfect score)
        self_score = haaqi_measurer.MeasureHAQQIOutput(ref_mp3)
        print(f"  HAAQI self-test score: {self_score}")

    def EraseTheMixing(self):
        """
        Remove all files and subdirectories in the given mixing-result directories.
        """
        # List of directories to clear
        dirs_to_empty = [
            os.path.join(self.base_dir, "tmp"),
            os.path.join(self.base_dir, "Mixing_Result"),
            os.path.join(self.base_dir, "Mixing_Result_Mp3"),
            os.path.join(self.base_dir, "Mixing_Result_Mp3_Wav"),
            os.path.join(self.base_dir, "Mixing_Result_AAC"),
            os.path.join(self.base_dir, "Mixing_Result_AAC_Wav"),
            os.path.join(self.base_dir, "Mixing_Result_NeuralCodec_Wav"),
        ]

        for directory_to_empty in dirs_to_empty:
            if os.path.exists(directory_to_empty) and os.path.isdir(directory_to_empty):
                # Iterate over all items (files and subdirectories)
                for item in os.listdir(directory_to_empty):
                    item_path = os.path.join(directory_to_empty, item)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        print(f"Error removing {item_path}: {e}")
                print(f"Cleared all contents in '{directory_to_empty}'.")
            else:
                print(f"Directory '{directory_to_empty}' does not exist or is not a directory.")