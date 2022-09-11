import torchaudio
from panns_inference import SoundEventDetection

from sed_demo.utils import to_df


sed_model = SoundEventDetection(checkpoint_path=None, device="cuda")


def sed(filepath: str):
    ts_output = None

    if filepath is None:
        text = "Please record a sound to proceed."
        return ts_output, text

    audio, sr = torchaudio.load(filepath)

    if audio.abs().sum() < 1:
        text = "No audio data, did you allow microphone permissions?"
        return ts_output, text

    audio = torchaudio.functional.resample(audio, sr, 32000)
    framewise_output = sed_model.inference(audio)
    ts_output = to_df(framewise_output[0])
    text = "Successfully processed."

    if ts_output is None:
        text = "No detected sound events."

    return ts_output, text
