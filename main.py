import numpy as np
import tensorflow as tf
#import time
import pyaudio
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
'''
import datetime
import serial
'''

plt.ion()

figure, ax = plt.subplots(figsize=(8,6))
x = np.linspace(0, 1000, 1000)
y = np.linspace(0, 100000, 1000)
line1, = ax.plot(x, y, label='4')
legend = plt.legend()


slice_list = [0, 100, 100, 150, 450, 550]
num_list = [100, 50, 50]
model = tf.keras.models.load_model('beekeeping_0_100_100_100_150_50_450_550_50_200_100x2_3.h5')
CHUNK = 9600
RATE = 9600


audio = pyaudio.PyAudio()

for index in range(audio.get_device_count()):
    desc = audio.get_device_info_by_index(index)
    print("DEVICE: {device}, INDEX: {index}, RATE: {rate} ".format(
        device=desc["name"], index=index, rate=int(desc["defaultSampleRate"])))


p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)#, input_device_index=1)
'''
py_serial = serial.Serial(
    port='COM5',
    baudrate=9600,
)
'''


while True:
    data = np.fromstring(stream.read(CHUNK), dtype=np.int16)
    '''
    if py_serial.readable():
        response = py_serial.readline()
        print(response[:len(response) - 1].decode())
        print((datetime.datetime.now()).strftime('%H:%M:%S'))

    str_res = str(ser.readline(), 'UTF-8')
    '''
    mag = abs(np.fft.fft(data))
    x = np.linspace(0, RATE, len(mag))

    line1.set_xdata(x[:1000])
    line1.set_ydata(mag[:1000])


    size = len(mag) // RATE
    sampling_1 = mag[size * slice_list[0]:size * slice_list[1]:size*((slice_list[1]-slice_list[0])//num_list[0])]
    sampling_2 = mag[size * slice_list[2]:size * slice_list[3]:size*((slice_list[3]-slice_list[2])//num_list[1])]
    sampling_3 = mag[size * slice_list[4]:size * slice_list[5]:size*((slice_list[5]-slice_list[4])//num_list[2])]
    sampling = (np.hstack([sampling_1, sampling_2, sampling_3])).tolist()
    yhat = model.predict([sampling])
    print(np.argmax(yhat, axis=1))#, "  at  ", (datetime.datetime.now()).strftime('%H:%M:%S'))
    line1.set_label("%s" % str(np.argmax(yhat, axis=1)))

    legend.remove()
    legend = plt.legend()

    figure.canvas.draw()
    figure.canvas.flush_events()



stream.stop_stream()
stream.close()
p.terminate()


