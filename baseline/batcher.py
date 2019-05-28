import os
import numpy as np
from lexical import parse_lines

DADB_DIR = 'data/dadb/'


def iterate_data(batch_size = 100,dir = DADB_DIR,shuffle=True):
    files_list=os.listdir(dir)
    for file in files_list:
        with open(DADB_DIR + file) as f:
            lines = f.readlines()
            sentences,labels,time_spans = parse_lines(lines)

        indx = np.arange(len(lines))[2:] #Not taking the first two sentences as they do not have context
        if shuffle:
            np.random.shuffle(indx)
        for start_idx in range(0, len(indx), batch_size):
            end_idx = min(start_idx + batch_size, len(indx))
            batch_sentences = np.concatenate((
                np.take(sentences,indx[start_idx: end_idx] - 2),
                np.take(sentences,indx[start_idx: end_idx] - 1),
                np.take(sentences,indx[start_idx: end_idx]))) # shape : [batch_size,context_length = 3,sent_len]
            batch_labels = np.take(labels,indx[start_idx: end_idx])
            batch_time_spans = np.take(time_spans,indx[start_idx: end_idx])
            yield batch_sentences,batch_labels,batch_time_spans

