#!/usr/bin/env python3
"""
Audio Feedback (Howling) Simulation System

Simulates acoustic feedback between microphone and speaker in a room environment.
Generates microphone and speaker signals containing howling effects.
"""

import argparse
import numpy as np
import soundfile as sf
import copy
import rir_generator as rir


def model(input_spec):
    """
    Audio processing model (placeholder for custom processing)
    
    Args:
        input_spec: Input audio spectrum
        
    Returns:
        Processed audio spectrum (currently passthrough)
    """
    return input_spec


def configure_arguments():
    """Configure command line arguments using argparse"""
    parser = argparse.ArgumentParser(description="Audio Feedback Simulation")
    
    # Audio file arguments
    parser.add_argument("-i", "--input", default="./data/LDC93S6A.wav",
                       help="Input clean speech file (default: ./data/LDC93S6A.wav)")
    parser.add_argument("-m", "--mic_output", default="./data/howling_mic.wav",
                       help="Microphone output with howling (default: ./data/howling_mic.wav)")
    parser.add_argument("-s", "--speaker_output", default="./data/howling_speaker.wav",
                       help="Speaker output with howling (default: ./data/howling_speaker.wav)")
    
    # Room configuration arguments
    parser.add_argument("--room_size", nargs=3, type=float, default=[5, 5, 3],
                       help="Room dimensions [L W H] in meters (default: 5 5 3)")
    parser.add_argument("--mic_pos", nargs=3, type=float, default=[3, 2, 1],
                       help="Microphone position [x y z] in meters (default: 3 2 1)")
    parser.add_argument("--speaker_pos", nargs=3, type=float, default=[3, 2.05, 1],
                       help="Speaker position [x y z] in meters (default: 3 2.05 1)")
    
    # Acoustic parameters
    parser.add_argument("--rt60", type=float, default=0.5,
                       help="Reverberation time in seconds (default: 0.5)")
    parser.add_argument("--gain", type=float, default=1,
                       help="System gain from mic to speaker (default: 0.4)")
    
    return parser.parse_args()


def howlingSimulate():
    # Parse command line arguments
    args = configure_arguments()
    
    # Load clean speech file
    try:
        x, fs = sf.read(args.input)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # Generate Room Impulse Response (RIR)
    try:
        rir_data = rir.generate(
            c=340,                  # Speed of sound (m/s)
            fs=fs,                  # Sampling rate
            r=args.mic_pos,         # Receiver (microphone) position
            s=args.speaker_pos,     # Source (speaker) position
            L=args.room_size,       # Room dimensions
            reverberation_time=args.rt60,  # RT60 reverberation time
            nsample=4096            # RIR length in samples
        )
    except Exception as e:
        print(f"Error generating RIR: {e}")
        return

    # System parameters
    G = args.gain                  # System gain from mic to speaker
    N = min(2000, len(rir_data))   # Length of RIR to use
    hop_size = 64                  # Frame shift size
    fft_size = hop_size * 2        # FFT window size
    x_len = len(x)                 # Length of input signal
    frame_num = x_len // hop_size  # Number of frames to process

    # Initialize buffers
    rir_buff = np.zeros(N + hop_size)      # Buffer for speaker output
    data_ana_buff = np.zeros(fft_size)     # Analysis buffer
    data_sys_buff = np.zeros(fft_size)     # Synthesis buffer
    windows = np.sqrt(np.hanning(fft_size)) # Window function (sqrt-Hann)

    # Output buffers
    y = np.zeros((frame_num, hop_size))    # Speaker output
    d = np.zeros((frame_num, hop_size))    # Microphone output
    y1 = np.zeros(hop_size)               # Feedback signal buffer

    # Main processing loop
    for i in range(frame_num):
        # Microphone signal = clean speech + feedback
        x1 = x[i*hop_size:(i+1)*hop_size] + y1
        d[i,:] = x1

        # Analysis filter bank (STFT)
        data_ana_buff[:-hop_size] = data_ana_buff[hop_size:]  # Shift buffer
        data_ana_buff[hop_size:] = x1                         # Add new samples
        data_ana_win = data_ana_buff * windows                # Apply window
        data_spec = np.fft.fft(data_ana_win)                 # FFT

        # Process spectrum (placeholder for custom processing)
        data_spec_enh = model(data_spec)

        # Synthesis filter bank (ISTFT)
        data_sys = np.real(np.fft.ifft(data_spec_enh))  # IFFT
        data_sys_win = data_sys * windows               # Apply window

        # Overlap-add synthesis
        data_sys_buff += data_sys_win
        data_out = copy.deepcopy(data_sys_buff[:hop_size])
        data_sys_buff[:hop_size] = data_sys_buff[hop_size:]
        data_sys_buff[hop_size:] = 0

        # Forward path (speaker output)
        x1 = data_out
        y_tmp = G * x1                     # Apply system gain
        np.clip(y_tmp, -2, 2, out=y_tmp)   # Clip to prevent overflow
        y[i,:] = y_tmp

        # Feedback path (RIR convolution)
        rir_buff[N:] = y[i,:]  # Add new speaker output to buffer
        for j in range(hop_size):
            # Convolve with RIR
            data = rir_buff[1+j:1+j+N]
            y1[j] = np.dot(data, rir_data[:N])
        rir_buff[:N] = rir_buff[hop_size:]  # Shift buffer

    # Save output files
    try:
        sf.write(args.mic_output, d.reshape(-1, 1), fs)
        sf.write(args.speaker_output, y.reshape(-1, 1), fs)
        print(f"Successfully generated output files:\n"
              f"- Microphone signal: {args.mic_output}\n"
              f"- Speaker signal: {args.speaker_output}")
    except Exception as e:
        print(f"Error saving output files: {e}")


if __name__ == "__main__":
    howlingSimulate()