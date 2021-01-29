def processAllFiles(csvPath='/opt/ml/processing/input_data/labels.csv'):
    '''
    :param
    csvFile contains filenames and class.
    '''
    label_df = pd.read_csv(csvPath)
    basePath = '/opt/ml/processing/input_data/audio/audio/44100/'
    for index, row in label_df.iterrows():
        try:
            createmelspecs(basePath + row['filename'], subfolder=row['broadclass'])
        except Exception as e:
            print("file not found may be")
    return