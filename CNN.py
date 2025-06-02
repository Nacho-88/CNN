import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from astropy.io import fits
import glob
import os
import cv2

# Epochs of training
EPOCHS = 50

#####################################################################################

def load_fits(directorio, shape_objetive=(512, 512, 16), max_imgs=1000, filtro_optico="optic_0"):
    """
    Carga archivos .fits agrupados de 16 en 16 como volúmenes 3D.
    
    Args:
        directorio (str): Ruta al directorio con archivos .fits.
        shape_objetive (tuple): (alto, ancho, profundidad) objetivo de salida.
        max_imgs (int): Número máximo de volúmenes a cargar.
        filtro_optico (str): Substring que deben contener los archivos .fits.
    
    Returns:
        np.ndarray: Array de forma (N, alto, ancho, profundidad, 1)
    """
    archivos = sorted(glob.glob(os.path.join(directorio, "*.fits")))
    archivos = [f for f in archivos if filtro_optico in os.path.basename(f)]
    
    # Asegurar que hay suficientes archivos para hacer volúmenes completos
    num_grupos = min(len(archivos) // 16, max_imgs)
    
    lista_volumenes = []

    for i in range(num_grupos):
        grupo = archivos[i * 16 : (i + 1) * 16]
        slices = []

        for archivo in grupo:
            with fits.open(archivo) as hdul:
                data = hdul[0].data.astype(np.float32)
                
                # Redimensionar cada frame 2D al tamaño objetivo (alto, ancho)
                data = cv2.resize(data, (shape_objetive[1], shape_objetive[0]), interpolation=cv2.INTER_AREA)
                
                # Normalización por frame
                data -= np.min(data)
                max_val = np.max(data)
                if max_val != 0:
                    data /= max_val

                slices.append(data)

        # Apilar slices en eje profundidad
        volumen = np.stack(slices, axis=-1)  # (alto, ancho, 16)

        # Agregar canal
        volumen = np.expand_dims(volumen, axis=-1)  # (alto, ancho, 16, 1)

        lista_volumenes.append(volumen)
        
        print(f"[OK] Grupo {i+1}/{num_grupos} cargado correctamente.")

        print(f"\n[FIN] Total volúmenes cargados: {len(lista_volumenes)}")

    return np.array(lista_volumenes)

######################################################################################################################

#Creación y carga de la CNN
size = (3, 3, 3)

inputs1= tf.keras.layers.Input(shape=(2048, 2048, 16, 1), name="input_volume")

x1 = tf.keras.layers.Conv3D(32, size, padding='same', activation='relu')(inputs1)
x1 = tf.keras.layers.Conv3D(32, size, padding='same', activation='relu')(x1)

x2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x1)
x2 = tf.keras.layers.Conv3D(64, size, padding='same', activation='relu')(x2)
x2 = tf.keras.layers.Conv3D(64, size, padding='same', activation='relu')(x2)

x3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x2)
x3 = tf.keras.layers.Conv3D(128, size, padding='same', activation='relu')(x3)
x3 = tf.keras.layers.Conv3D(128, size, padding='same', activation='relu')(x3)

x4 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x3)
x4 = tf.keras.layers.Conv3D(256, size, padding='same', activation='relu')(x4)
x4 = tf.keras.layers.Conv3D(256, size, padding='same', activation='relu')(x4)

x5 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x4)
x5 = tf.keras.layers.Conv3D(512, size, padding='same', activation='relu')(x5)
x5 = tf.keras.layers.Conv3D(512, size, padding='same', activation='relu')(x5)

y4 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x5)
y4 = tf.keras.layers.Concatenate()([y4, x4])
y4 = tf.keras.layers.Conv3D(256, size, padding='same', activation='relu')(y4)
y4 = tf.keras.layers.Conv3D(256, size, padding='same', activation='relu')(y4)

y3 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(y4)
y3 = tf.keras.layers.Dropout(0.2)(y3)
y3 = tf.keras.layers.Concatenate()([y3, x3])
y3 = tf.keras.layers.Conv3D(128, size, padding='same', activation='relu')(y3)
y3 = tf.keras.layers.Conv3D(128, size, padding='same', activation='relu')(y3)

y2 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(y3)
y2 = tf.keras.layers.Dropout(0.2)(y2)
y2 = tf.keras.layers.Concatenate()([y2, x2])
y2 = tf.keras.layers.Conv3D(64, size, padding='same', activation='relu')(y2)
y2 = tf.keras.layers.Conv3D(64, size, padding='same', activation='relu')(y2)

y1 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(y2)
y1 = tf.keras.layers.Dropout(0.2)(y1)
y1 = tf.keras.layers.Concatenate()([y1, x1])
y1 = tf.keras.layers.Conv3D(32, size, padding='same', activation='relu')(y1)
y1 = tf.keras.layers.Conv3D(32, size, padding='same', activation='relu')(y1)

x= tf.keras.layers.Conv3D(1, size, activation='sigmoid', padding='same')(y1)

model1 = tf.keras.models.Model(inputs=inputs1, outputs=x)
model1.summary()


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model1.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
###################################################################################
#Carga de las imagenes
x_cosmicos = load_fits("D:/FIE/Photsat/photsat_frames_cosmic_rays", max_imgs=1000, filtro_optico="optic_0")

x_sc = load_fits("D:/FIE/Photsat/no_noisy_frames", max_imgs=1000, filtro_optico="optic_0")

###################################################################################
#Entrenamiento de la CNN
hist1 = model1.fit(x_cosmicos, x_sc, #Falta por poner las imágenes que le pasamos para que entrene
    epochs=EPOCHS,
    batch_size=5, #Creo que con 5 irá bien
    shuffle=True, 
    validation_split=0.2)

####################################################################################



###############################################################################################
#HASTA EL hist1 YO CREO QUE ESTÁ BIEN HECHO. LO DE DEBAJO ESTABA EN EL CÓDIGO DEL GITHUB Y 
#TODAVÍA NO SE SI VA A HABER QUE USARLO
###############################################################################################


# ## Plot results

# # Learning

# plt.figure()
# plt.plot(hist1.epoch, hist1.history['loss'],'r',hist1.epoch, hist1.history['val_loss'],'r:')
# plt.xlabel('Epoch')
# plt.ylabel('Loss function')
# plt.legend(['Train loss', 'Validation loss'])
# plt.show()

# # Graph

# n = 0
# l = np.arange(0, x_test.shape[1])

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(l, x_test[n,:,0], color='black', linewidth=0.5, label='signal')
# plt.grid(color='gray', linestyle=':', linewidth=1)
# plt.ylabel('Normalized pulse amplitude')
# plt.xlim(0, x_test.shape[1])
# plt.title('Input signal')

# plt.subplot(2,1,2)

# THR = 0.03

# l1 = l[x_test_o2[n,:,0]>THR]
# l2 = l[y_pred1[n,:,0]>THR]

# x_test_o2b = x_test_o2[n,x_test_o2[n,:,0]>THR,0]
# y_pred1b = y_pred1[n,y_pred1[n,:,0]>THR,0]

# markerline, stemlines, baseline = plt.stem(l1, x_test_o2b, linefmt='red', markerfmt='rx', label='Ideal')
# plt.setp(stemlines, 'linewidth', 0.5)
# plt.setp(markerline, 'linewidth', 0.5)
# plt.setp(baseline, 'linewidth', 0.5)
# plt.xlim(0, x_test.shape[1])
# markerline, stemlines, baseline = plt.stem(l2, y_pred1b, linefmt='blue', markerfmt='b+', label='U-net')
# plt.setp(stemlines, 'linewidth', 0.5)
# plt.setp(markerline, 'linewidth', 0.5)
# plt.setp(baseline, 'linewidth', 0.5)
# plt.grid(color='gray', linestyle=':', linewidth=1)
# plt.xlabel(r'time [$\mu$s]')
# plt.ylabel('Normalized pulse amplitude')
# plt.xlim(0, x_test.shape[1])
# plt.title('Output signal')
# plt.legend()
# plt.show()
    
# y_pred = model1.predict(x_test_hist)
