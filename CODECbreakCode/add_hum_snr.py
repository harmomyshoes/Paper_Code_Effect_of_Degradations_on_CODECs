import random
import numpy as np
from numpy.typing import NDArray
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import calculate_desired_noise_rms, calculate_rms


class AddHumSNR(BaseWaveformTransform):
    """
    Add humming noise to the input. A random Signal to Noise Ratio (SNR) will be picked
    uniformly in the Decibel scale. This aligns with human hearing, which is more
    logarithmic than linear.
    """
    supports_multichannel = True

    def __init__(
        self,
        min_snr_db: float = 5.0,
        max_snr_db: float = 80.0,
        p: float = 0.5,
        frequencies: list = [50, 150]
    ):
        """
        :param min_snr_db: Minimum signal-to-noise ratio in dB. A lower number means more noise.
        :param max_snr_db: Maximum signal-to-noise ratio in dB. A greater number means less noise.
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        if min_snr_db > max_snr_db:
            raise ValueError("min_snr_db must not be greater than max_snr_db")
        if min_snr_db <= 0:
            raise ValueError("min_snr_db must be greater than 0")
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.frequencies = frequencies

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            # Pick SNR in decibel scale
            snr = random.uniform(self.min_snr_db, self.max_snr_db)

            clean_rms = calculate_rms(samples)
            noise_rms = calculate_desired_noise_rms(clean_rms=clean_rms, snr=snr)

            # In humming noise, the RMS gets roughly equal to the std
            self.parameters["noise_std"] = float(noise_rms)

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        new_audio_signal = np.copy(samples)
        t = np.arange(samples.size) / sample_rate
        for freq in self.frequencies:
            sine_wave = self.parameters["noise_std"] * np.sin(2 * np.pi * freq * t)
            new_audio_signal += sine_wave
        return new_audio_signal