from argparse import ArgumentParser
import logging, os
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger()


def go(input):
    artifact_path = Path('components/artifacts/')
    output_dir = input['out_folder_name']
    processed_output_dir = os.path.join(output_dir, 'processed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(processed_output_dir)

    if len(os.listdir(processed_output_dir)) == 0:
        logger.info(f'Initiating pagination for {input["input_path"]}')
        logger.info("Reading data from input file...")

        fname = input['input_path'].split('/')[-1]
        data = (
            pd.read_csv(input['input_path'], sep = "\t")
            .drop_duplicates()
        )
        page_size = 100000
        groups = data.groupby(np.arange(len(data.index))//page_size)
        for (frame_no, frame) in groups:
            out_fname = fname.replace(".csv", "_{}.csv".format(frame_no))
            out_path  = os.path.join(output_dir, out_fname)
            frame.to_csv(out_path, sep = "\t")
    else:
        logger.info(f'Collating processed paged data')
        df_list = [] 
        for fname in os.listdir(processed_output_dir):
            df_list.append(pd.read_csv(os.path.join(processed_output_dir,fname), sep = "\t"))
        
        df_list = pd.concat(df_list, ignore_index=True)
        df_list.to_csv(input['output_path'], sep = "\t")