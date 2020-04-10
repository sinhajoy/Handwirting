import os
import sys
from PIL import Image
import pandas as pd
import numpy as np


def scale_invert(raw_path, proc_path,height,width):

    """
    Function that scales and inverts each image to store them in a common directory.
    The proportions of the original image are preserved and a fill is added until reaching
    the target width.

    Arguments:

      - raw_path: Path of the original image. (String)
      - proc_path: Path where to store the processed image. (String)
      - height: Height of the images. (Int)
      - width: Width of the images. (Int)
    """
    # Upload the image
    im = Image.open(raw_path)

   
    raw_width, raw_height = im.size
    new_width = int(round(raw_width * (height / raw_height)))
    im = im.resize((new_width, height), Image.NEAREST)
    im_map = list(im.getdata())
    im_map = np.array(im_map)
    im_map = im_map.reshape(height, new_width).astype(np.uint8)

    #We fill and invert the values.
    data = np.full((height, width - new_width + 1), 255)
    im_map = np.concatenate((im_map, data), axis=1)
    im_map = im_map[:, 0:width]
    im_map = (255 - im_map)
    im_map = im_map.astype(np.uint8)
    im = Image.fromarray(im_map)

    #We store all images in common directory
    im.save(str(proc_path), "png")
    print("Processed image saved: " + str(proc_path))


def extract_training_batch(ctc_input_len,batch_size,im_path,csv_path):

    """
    Function that extracts a batch of images and their transcripts randomly to train the ANN.

        Arguments:

          - ctc_input_len: Length of the input sequence to the CTC layer. (Int)
          - batch_size: Size of the batch. (Int)
          - im_path: Path to the directory where the images are stored. (String)
          - csv_path: Path to the training dataset. (Int)

        Departure:

          - batchx: Tensor that contains the images as input matrices to the ANN.
            (Floats array: [batch_size, height, width, 1])
          - sparse: SparseTensor that contains the labels as positive integer values. (SparseTensor: index, values, shape)
          - transcriptions: Array with the transcripts corresponding to the "batchx" images. (String array: [batch_size])
          - seq_len: Array with the length of the input sequence to the CTC layer, "ctc_input_len". (Ints array: [batch_size])
    """

    #We randomly extract a DataFrame of "batch_size" size from the Training Dataset.
    df = pd.read_csv(csv_path, sep=",",index_col="index")
    df_sample=df.sample(batch_size).reset_index()

    # Declaramos las variables para la salida.
    batchx = []
    transcriptions = []
    index = []
    values=[]
    seq_len=[]

    # Creamos el lote a partir del Dataframe de muestras aleatorias.
    for i in range(batch_size):
        im_apt = df_sample.loc[i, ['image']].as_matrix()
        df_y =df_sample.loc[i, ['transcription']].as_matrix()
        for fich in im_apt:

            # Extraemos la imagen y la mapeamos en una matriz normalizada.
            fich = str(fich)
            fich = fich.replace("['", "").replace("']", "")
            im = Image.open(im_path + fich + ".png")
            width, height = im.size
            im_map = list(im.getdata())
            im_map = np.array(im_map)
            im_map = im_map / 255
            result=im_map.reshape(height, width,1)
            batchx.append(result)

            # Extraemos las etiquetas parseando la transcripción.
            original=""
            for n in list(str(df_y)):
                if n == n.lower() and n == n.upper():
                    if n in "0123456789":
                        values.append(int(n))
                        original = original + n
                elif n == n.lower():
                    values.append(int(ord(n) - 61))
                    original = original + n
                elif n == n.upper():
                    values.append(int((ord(n) - 55)))
                    original = original + n

            # Añadimos el indice del SparseTensor.
            for j in range(len(str(df_y))-4):
                index.append([i,j])

            # Añadimos las transcripciones y la longitud de la secuencia de entrada a la CTC.
            transcriptions.append(original)
            seq_len.append(ctc_input_len)

    # Creamos el Array que contiene todas las imagenes normalizadas el lote, entrada de la ANN.
    batchx = np.stack(batchx, axis=0)

    # Creamos el SparseTensor con el indice, las etiquetas que representan cada caracter y la longitud máxima de palabra
    shape=[batch_size,18]
    sparse=index,values,shape

    return batchx, sparse, transcriptions, seq_len


def extract_ordered_batch(ctc_input_len,batch_size,im_path,csv_path,cont):

    """
        Function that extracts a batch of images and their transcripts in an orderly manner to validate or test the ANN.

        Arguments:

          - ctc_input_len: Length of the input sequence to the CTC layer. (Int)
          - batch_size: Size of the batch. (Int)
          - im_path: Path to the directory where the images are stored. (String)
          - csv_path: Path to the validation dataset. (Int)
          - cont: Auxiliary index that allows the extraction of batches in an orderly manner. (Int)

        Departure:

          - batchx: Tensor that contains the images as input matrices to the ANN.
            (Floats array: [batch_size, height, width, 1])
          - sparse: SparseTensor that contains the labels as positive integer values. (SparseTensor: index, values, shape)
          - transcriptions: Array with the transcripts corresponding to the "batchx" images. (String array: [batch_size])
          - seq_len: Array with the length of the input sequence to the CTC layer, "ctc_input_len". (Ints array: [batch_size])
          - num_samples: Number of samples extracted. (Int)
    """

    # Extraemos secuencialmente un DataFrame de tamaño "batch_size" del Dataset.
    df = pd.read_csv(csv_path, sep=",",index_col="index")
    df_sample=df.loc[int(cont*batch_size):int((cont+1)*batch_size)-1,:].reset_index()
    num_samples=int(len(df_sample.axes[0]))



    # Declaramos las variables para la salida.
    batchx = []
    transcriptions = []
    index = []
    values=[]
    seq_len=[]

    # Creamos el lote a partir del Dataframe de muestras aleatorias.
    if len(df_sample.axes[0]) is not 0:
        for i in range(len(df_sample.axes[0])):
            im_apt = df_sample.loc[i, ['image']].as_matrix()
            df_y =df_sample.loc[i, ['transcription']].as_matrix()
            for fich in im_apt:

                # Extraemos la imagen y la mapeamos en una matriz normalizada.
                fich = str(fich)
                fich = fich.replace("['", "").replace("']", "")
                im = Image.open(im_path + fich + ".png")
                width, height = im.size
                im_map = list(im.getdata())
                im_map = np.array(im_map)
                im_map = im_map / 255
                result=im_map.reshape(height, width,1)
                batchx.append(result)

                # We extract the tags parsing the transcript.
                original=""
                for n in list(str(df_y)):
                    if n == n.lower() and n == n.upper():
                        if n in "0123456789":
                            values.append(int(n))
                            original=original+n
                    elif n==n.lower():
                        values.append(int(ord(n)-61))
                        original = original + n
                    elif n==n.upper():
                        values.append(int((ord(n)-55)))
                        original = original + n

                # We add the index of the SparseTensor.
                for j in range(len(str(df_y))-4):
                    index.append([i,j])

                # We add the transcripts and the length of the input sequence to the CTC.
                transcriptions.append(original)
                seq_len.append(ctc_input_len)

        
# We create the Array that contains all the normalized images in the batch, ANN input.
        batchx=np.stack(batchx, axis=0)

   
# We create the SparseTensor with the index, the labels that represent each character and the maximum word length
    shape=[batch_size,18]
    sparse=index,values,shape

    return batchx, sparse, transcriptions, seq_len, num_samples




def validation(curr_epoch,ctc_input_len, batch_size, im_path, csv_path, inputs, targets, keep_prob, seq_len, session, cost, ler):

    """
Function that performs the validation of the ANN on a specific dataset.

        Arguments:

          - curr_epoch: Current time. (Int)
          - ctc_input_len: Length of the input sequence to the CTC layer. (Int)
          - batch_size: Size of the batch. (Int)
          - im_path: Path to the directory where the images are stored. (String)
          - csv_path: Path to the validation dataset. (Int)
          - inputs: Placeholder of the model's input. (placeholder)
          - targets: Placeholder of the target outputs. (placeholder)
          - keep_prob: Placeholder for the dropout probability. (placeholder)
          - seq_len: Placeholder for the length of the input sequence to the CTC layer. (placeholder)
          - session: Current TensorFlow session. (Session)
          - cost: Tensor for the output of the CTC error. (Tensor: [1])
          - ler: Tensor for the exit of the LER. (Tensor: [1])

        Departure:

          - val_tuple: Result of the validation of the model in a complete epoch. (Tuple: {'epoch', 'cost', 'LER'})
    """

    # Variables auxiliaries.
    cont = 0
    total_val_cost = 0
    total_val_ler = 0
    
# Loop to perform validation on the entire Dataset
    while cont >= 0:
       
# We extract the batches sequentially using "cont"
        val_inputs, val_targets, val_original, val_seq_len, num_samples = extract_ordered_batch(
            ctc_input_len, batch_size, im_path, csv_path, cont)

        
# If the number of samples extracted equals "batch_size", a complete batch has been extracted.
        if num_samples == batch_size:
            val_feed = {inputs: val_inputs,
                    targets: val_targets,
                    keep_prob: 1,
                    seq_len: val_seq_len}
            val_cost, val_ler = session.run([cost, ler], val_feed)
            total_val_cost += val_cost
            total_val_ler += val_ler
            cont += 1

        
        # It was not possible to extract any more lots and therefore the average of "cost" and "ler" is calculated
        elif num_samples == 0:
            val_tuple = {'epoch': [curr_epoch], 'val_cost': [total_val_cost / (cont + 1)],
                    'val_ler': [total_val_ler / (cont + 1)]}
            cont = -1


        # The complete batch could not be extracted, there are not enough samples in the Dataset and therefore,
        # the average of "cost" and "ler" are calculated.
        else:
            val_feed = {inputs: val_inputs,
                    targets: val_targets,
                    keep_prob: 1,
                    seq_len: val_seq_len}
            val_cost, val_ler = session.run([cost, ler], val_feed)
            total_val_cost += val_cost
            total_val_ler += val_ler
            val_tuple = {'epoch': [curr_epoch], 'val_cost': [total_val_cost / (cont + 1)],
                    'val_ler': [total_val_ler / (cont + 1)]}
            cont = -1

    return val_tuple

