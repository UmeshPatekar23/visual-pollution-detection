# data handling
import pandas as pd
import torch 
import argparse 
import os
import cv2

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# to log the results using current date and time
from datetime import datetime

# progress bar
from tqdm import tqdm

PATTRN_WEIGHTS = '"D:\hackaton\project\EDI\models\pattrn\weights\best.pt"'

def prepare_results(model_name, n_segments):
    print("Resizing images to 960x540 and saving them to resized_dataset/images if they do not exist...")
    df_test = pd.read_csv('dataset/test.csv')
    df_test['image_path'] = 'dataset/images/' + df_test['image_path']
    os.makedirs('resized_dataset/images', exist_ok=True)
    # read each image and resize it to 960x540
    for i in tqdm(range(len(df_test))):
        if os.path.exists('resized_'+df_test['image_path'][i]):
            continue
        img = cv2.imread(df_test['image_path'][i])
        img = cv2.resize(img, (960, 540))
        cv2.imwrite('resized_'+df_test['image_path'][i], img)

    # load test data
    df_test = pd.read_csv('dataset/test.csv')
    # correct paths for test images
    df_test['image_path'] = 'resized_dataset/images/' + df_test['image_path']    

    # realase the GPU memory
    torch.cuda.empty_cache()

    # if the pattrn model weights do not exist, or if they are corrupted, download them
    if not os.path.exists(f'models/pattrn/weights/best.pt') or os.stat('models/pattrn/weights/best.pt').st_size < 1E8:
        print("Downloading model weights...")
        torch.hub.download_url_to_file(PATTRN_WEIGHTS, 'models/pattrn/weights/best.pt', progress=True)
            
    # Load Model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=f'models/{model_name}/weights/best.pt')  # local model

    # create a dataframe to store the results
    total_results_df = pd.DataFrame()

    # segment the test data to avoid memory issues
    n_segments = n_segments
    no_predictions = 0
    
    print("total images:", len(df_test))

    # start inference
    for n in tqdm(range(n_segments), desc='Inference on {} segments'.format(n_segments)):
        divide = len(df_test) // n_segments

        df_test_t = df_test.copy()

        # segment the test data to avoid memory issues
        if n != n_segments - 1:
            img = df_test['image_path'].tolist()[divide*n : (n+1)*divide]
            df_test_t = df_test[divide*n:(n+1)*divide]
        else:
            img = df_test['image_path'].tolist()[divide*n :]
            df_test_t = df_test[divide*n:]

        # inference
        results = model(img)

        # convert results to a dataframe (Fill images with no predictions with GARBAGE class)
        start = n*divide
        results_df = pd.DataFrame()
        for i, pred in enumerate(results.pandas().xyxy, start=start):
            # if there is a prediction, then add the labels and bounding boxes
            if len(pred) > 0:
                pred['image_path'] = df_test_t['image_path'][i]
                results_df = results_df.append(pred)

            # if there is no prediction, then add GARBAGE class (This is done to avoid errors in the submission)
            else:
                no_predictions += 1
                results_df = results_df.append(
                    {'image_path' : df_test_t['image_path'][i],
                    'xmin' : 0,
                    'ymin' : 0,
                    'xmax' : 0,
                    'ymax' : 0,
                    'class' : 3,
                    'name' : 'GARBAGE'
                    }, ignore_index=True
                )

        # convert image_path to be the same as the test.csv for submission
        results_df['image_path'] = results_df['image_path'].apply(lambda x: x.split('/')[-1])    

        results_df = results_df[['class', 'image_path', 'name', 'xmax', 'xmin', 'ymax', 'ymin']]

        # append the segment's results to the total results
        total_results_df = total_results_df.append(results_df)    

    print("inference done")

    print("images with no prediction:", no_predictions)
    
    # create results folder if it does not exist
    os.makedirs('results', exist_ok=True)

    # save the results with the current time stamp
    dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    total_results_df.to_csv(f'results/{model_name}_{dt_string}.csv', index=False)
    print("results saved to:" + f'results/{model_name}_{dt_string}.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        default='pattrn',
        help="experiment name",
        type=str
    )

    parser.add_argument(
        '--segments',
        default='8',
        help="number of segments to divide the test data to avoid memory issues",
        type=int
    )

    # parse args
    args = parser.parse_args()
        
    # read path
    model_name = args.model_name
    n_segments = args.segments

    # run the script
    prepare_results(model_name, n_segments)