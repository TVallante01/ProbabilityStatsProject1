import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Constants
DATA_BITS = 8
ACK_BIT_INDEX = 8
CHECKSUM_BIT_INDEX = 9
MAX_ATTEMPTS = 100
NOISE_LEVELS = np.linspace(0, 1, 20)  # Noise scale from 0 to 1 in 20 steps

# Function to generate a 10-bit signal with a checksum
def generateSignal():
    dataBits = np.random.randint(0, 1, DATA_BITS)
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

# Simulation over noise levels
truePositives = []
falsePositives = []
garbledMessages = []
correctedMessages = []
uncorrectedMessages = []
totalMessages = []


for noiseLevel in NOISE_LEVELS:
    # Counters for each scenario
    garbledCount = 0
    correctedCount = 0
    uncorrectedCount = 0
    attempts = 0
    truePositiveCount = 0
    falsePositiveCount = 0
    
    for _ in range(MAX_ATTEMPTS):
        messageSent = generateSignal()
        ackReceived = False
        attemptCount = 0
        
        while not ackReceived and attemptCount < MAX_ATTEMPTS:
            attemptCount += 1
            noisySignal = addNoise(messageSent, noiseLevel)
            isValid, receivedSignal = verifySignal(noisySignal)
            
            if isValid and receivedSignal[ACK_BIT_INDEX] == 1:
                ackReceived = True
                truePositiveCount += 1
                correctedCount += (attemptCount > 1)
            else:
                garbledCount += 1
                if isValid:
                    falsePositiveCount += 1
            
        # Track uncorrected messages if max attempts reached without acknowledgment
        if not ackReceived:
            uncorrectedCount += 1
        
        attempts += attemptCount
    
    # Collect statistics for each noise level
    garbledMessages.append(garbledCount)
    correctedMessages.append(correctedCount)
    uncorrectedMessages.append(uncorrectedCount)
    totalMessages.append(attempts)
    truePositives.append(truePositiveCount / attempts)

# Smoothing the curves
noiseLevelsSmooth = np.linspace(NOISE_LEVELS.min(), NOISE_LEVELS.max(), 300)
truePositivesSmooth = make_interp_spline(NOISE_LEVELS, truePositives)(noiseLevelsSmooth)

# Plotting the results
plt.figure(figsize=(14, 7))

# Plot True Positive Rate vs Noise Level
plt.subplot(1, 2, 1)
plt.plot(noiseLevelsSmooth, truePositivesSmooth, label="True Positive Rate", color="green")
plt.xlabel("Noise Level")
plt.ylabel("True Positive Rate")
plt.title("True Positive Rate Over Noise Levels")
plt.legend()

plt.tight_layout()
plt.show()

# Print summary statistics
print("Summary Statistics for the Simulation:")
print(f"Total Messages Sent: {sum(totalMessages)}")
print(f"Garbled Messages: {sum(garbledMessages)}")
print(f"Corrected Messages: {sum(correctedMessages)}")
print(f"Uncorrected Messages: {sum(uncorrectedMessages)}")
