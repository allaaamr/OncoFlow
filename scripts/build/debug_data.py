import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.patches as mpatches

import numpy as np
import matplotlib.pyplot as plt

img =np.load('/home/alaa.mohamed/MIU2/PatientID_0037_image.npy')
days = np.load('/home/alaa.mohamed/MIU2/PatientID_0037_days.npy')
treat = np.load('/home/alaa.mohamed/MIU2/PatientID_0037_treatment.npy')

print("img ", img.shape)
print("treat ", treat)
print("days ", days)


