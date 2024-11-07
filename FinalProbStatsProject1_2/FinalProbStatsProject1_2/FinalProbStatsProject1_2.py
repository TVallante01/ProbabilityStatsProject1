
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.interpolate import make_interp_spline

# Constants
DATA_BITS = 8
ACK_BIT_INDEX = 8
CHECKSUM_BIT_INDEX = 9
MAX_ATTEMPTS = 100  # Maximum messages to be sent, not retransmissions
NOISE_LEVELS = np.linspace(0, 1, 20)  # Noise scale from 0 to 1 in 20 steps

# Function to generate a 10-bit signal with a checksum
def generateSignal():
    dataBits = np.random.randint(0, 2, DATA_BITS)
    checksum = sum(dataBits) % 2  # checksum = sum(dataBits) % 2
    signal = np.append(dataBits, [0, checksum])  # Ack bit is 0 initially
    return signal

# Function to verify signal and set acknowledgment
def verifySignal(signal):
    dataBits = signal[:DATA_BITS]
    checksum = signal[CHECKSUM_BIT_INDEX]
    calculatedChecksum = sum(dataBits) % 2
    isValid = (checksum == calculatedChecksum)
    # If valid, set acknowledgment bit to 1
    if isValid:
        signal[ACK_BIT_INDEX] = 1
    return isValid, signal

# Function to add noise to signal
def addNoise(signal, noiseLevel):
    noisySignal = signal.copy()
    for i in range(len(noisySignal)):
        if np.random.rand() < noiseLevel:
            noisySignal[i] = 1 - noisySignal[i]  # Flip bit
    return noisySignal

# Simulation over noise levels for both scenarios
truePositivesWithCorrection = []
truePositivesWithoutCorrection = []

for noiseLevel in NOISE_LEVELS:
    # Counters for each scenario
    truePositiveCountWithCorrection = 0
    truePositiveCountWithoutCorrection = 0
    
    # Scenario 1: With Error Correction
    for _ in range(MAX_ATTEMPTS):
        messageSent = generateSignal()
        ackReceived = False
        
        # Retransmit until an ack is received or max retransmissions reached
        for attemptCount in range(MAX_ATTEMPTS):
            noisySignal = addNoise(messageSent, noiseLevel)
            isValid, receivedSignal = verifySignal(noisySignal)
            
            # If ack received, count as a true positive for this message and break loop
            if isValid and receivedSignal[ACK_BIT_INDEX] == 1:
                truePositiveCountWithCorrection += 1
                ackReceived = True
                break

    # Scenario 2: Without Error Correction
    for _ in range(MAX_ATTEMPTS):
        messageSent = generateSignal()
        noisySignal = addNoise(messageSent, noiseLevel)
        isValid, receivedSignal = verifySignal(noisySignal)
        
        if isValid and receivedSignal[ACK_BIT_INDEX] == 1:
            truePositiveCountWithoutCorrection += 1

    # Calculate true positive rates
    truePositivesWithCorrection.append(truePositiveCountWithCorrection / MAX_ATTEMPTS)
    truePositivesWithoutCorrection.append(truePositiveCountWithoutCorrection / MAX_ATTEMPTS)

# Smoothing the curves
noiseLevelsSmooth = np.linspace(NOISE_LEVELS.min(), NOISE_LEVELS.max(), 300)
truePositivesWithCorrectionSmooth = make_interp_spline(NOISE_LEVELS, truePositivesWithCorrection)(noiseLevelsSmooth)
truePositivesWithoutCorrectionSmooth = make_interp_spline(NOISE_LEVELS, truePositivesWithoutCorrection)(noiseLevelsSmooth)

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot True Positive Rates for both scenarios
plt.plot(noiseLevelsSmooth, truePositivesWithCorrectionSmooth, label="With Error Correction", color="blue")
plt.plot(noiseLevelsSmooth, truePositivesWithoutCorrectionSmooth, label="Without Error Correction", color="orange")
plt.xlabel("Noise Level")
plt.ylabel("True Positive Rate")
plt.title("True Positive Rate Over Noise Levels")
plt.legend()
plt.show()

# Print summary statistics
print("Summary Statistics for the Simulation:")
print(f"True Positives With Error Correction: {np.mean(truePositivesWithCorrection) * 100:.2f}%")
print(f"True Positives Without Error Correction: {np.mean(truePositivesWithoutCorrection) * 100:.2f}%")
