import os
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import librosa, librosa.display, IPython.display as ipd
import json
from mutagen.mp3 import MP3
import noisereduce as no
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import boto3

def saveMel(y, directory):
    N_FFT = 1024  # Number of frequency bins for Fast Fourier Transform
    HOP_SIZE = 1024  # Number of audio frames between STFT columns
    SR = 44100  # Sampling frequency
    N_MELS = 30  # Mel band parameters
    WIN_SIZE = 1024  # number of samples in each STFT window
    WINDOW_TYPE = 'hann'  # the windowin function
    FEATURE = 'mel'  # feature representation

    fig = plt.figure(1, frameon=False)
    fig.set_size_inches(6, 6)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ## describe the 3 parameters below

    HOP_SIZE = 1024
    N_MELS = 128
    FMIN = 1400
    S = librosa.feature.melspectrogram(y=y, sr=SR,
                                       n_fft=N_FFT,
                                       hop_length=HOP_SIZE,
                                       n_mels=N_MELS,
                                       htk=True,
                                       fmin=FMIN,
                                       fmax=SR/2)
    librosa.display.specshow(librosa.power_to_db(S ** 2, ref=np.max), fmin=FMIN, y_axis='linear')
    print(directory)
    fig.savefig(directory)
    fig.clear()
    ax.cla()
    plt.clf()
    plt.close('all')


def downloadfilefroms3(bucket, filepath, localpath="/tmp"):
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket, filepath, localpath+"/"+filepath.split("/")[-1])
        return localpath+"/"+filepath.split("/")[-1]

    except Exception as e:
        print(e)

def uploadfiletos3(localpath,bucket,objname):

    s3 = boto3.client("s3")
    s3.upload_file(localpath,bucket,objname)
    print("upload complete "+str(localpath) + ", to s3://" + bucket + "/" + objname)



def createmelspecs(bucket, objpath, desired=1, minimum=0.5, stride=0, name=5):

    # these are hard coded for now, once working, turn it into functional args
    size = {'desired': desired,  # [seconds]
            'minimum': minimum,  # [seconds]
            'stride': stride,  # [seconds]
            'name': name}
    step = 1

    path=downloadfilefroms3(bucket, objpath)

    outputpath="output/melspecs/"+"/".join(objpath.split("/")[1:3])
    print(outputpath)
    if step > 0:
        try:

            directory = "/tmp/mels-2class/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            if not os.path.exists(directory + path.rsplit('/', 1)[1].replace(' ', '')[:-4] + "1_1.png"):
                y, sr = librosa.load(path, mono=True)
                y = no.reduce_noise(audio_clip=y, noise_clip=y, verbose=False)
                step = (size['desired'] - size['stride']) * sr
                nr = 0;
                for start, end in zip(range(0, len(y), step), range(size['desired'] * sr, len(y), step)):
                    nr = nr + 1
                    print(nr)
                    if end - start > size['minimum'] * sr:
                        melpath = path.rsplit('/', 1)[1]
                        melpath = directory + melpath.replace(' ', '')[:-4] + str(nr) + "_" + str(nr) + ".png"
                        print(melpath)
                        saveMel(y[start:end], melpath)
                        uploadfiletos3(melpath, bucket, outputpath+"/"+melpath.split("/")[-1])
            pass
        except ZeroDivisionError as e:
            print("excepting zero division error")
    else:
        print("Error: Stride should be lower than desired length.")




s3_client = boto3.client('s3')
def download_dir(prefix, local, bucket, client=s3_client):
    """
    params:
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    - bucket: s3 bucket with target contents
    - client: initialized s3 client object
    """
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket':bucket,
        'Prefix':prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(bucket, k, dest_pathname)

def useCSV(csvPath='/opt/ml/processing/input_data/labels.csv'):
    '''
    :param
    csvFile contains filenames and class.
    '''
    BUCKET = os.getenv('BUCKET', 'my-classification-audio-files')


    label_df = pd.read_csv(csvPath)
    for index, row in label_df.iterrows():
        try:
            createmelspecs(BUCKET, row['filename'])
        except Exception as e:
            print("file not found may be")
    return

def useSQS():
    # Get the service resource
    AWS_REGION = os.getenv('AWS_REGION', "us-west-2")
    sqs = boto3.resource('sqs', region_name=AWS_REGION)
    SQS_QUEUE = os.getenv('SQS_QUEUE', 'xeno-canto-melspecs')

    # # Get the queue
    queue = sqs.get_queue_by_name(QueueName=SQS_QUEUE)
    messages_missed = 0 # will store info to break the loop when there are no more messages in the queue
    while messages_missed < 10: # If we don't get messages from Queue 10 time continuosly then bye bye
        mp3file = queue.receive_messages(WaitTimeSeconds=20, MaxNumberOfMessages=1)

        for message in mp3file:
            messages_missed -= 1
            messageRecords = json.loads(message.body)["Records"]

            for record in messageRecords:
                bucket = record["s3"]["bucket"]["name"]
                key = record["s3"]["object"]["key"]
                key = key.replace("+", " ")
                print("s3://" + bucket + "/" + key)

                try:
                    createmelspecs(bucket, key)
                except:
                    print("error occured for file: " + key)
            
            message.delete()


if __name__=="__main__":
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--bucket', metavar='b', dest="bucket", type=str,
    #                     help='name of the source bucket')
    # parser.add_argument('--objpath', dest='objpath', metavar="o", type=str,
    #                     help='objectpath including folder')


    # args = parser.parse_args()

    # print(args)

    # createmelspecs(args.bucket, args.objpath)

    # useSQS()
    useCSV()

    