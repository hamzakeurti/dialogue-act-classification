

def parse_line(line):
    line_split = line.split(',')
    begin_time,end_time = float(line_split[0]),float(line_split[1])
    label = line_split[-9]
    words = [x.split('+')[-1] for x in line_split[4].split('|')]
    return words,label,begin_time,end_time

def parse_lines(lines):
    sentences = []
    labels = []
    time_spans = []
    for line in lines:
        words,label,begin_time,end_time = parse_line(line)
        sentences.append(words)
        labels.append(label)
        time_spans.append((begin_time,end_time))
    return sentences,labels,time_spans



# def parse_dadb(filename):
#     with open(filename,'r') as file:
#         line = file.readline()
#         k=0
#         while line:
#             line_split = line.split(',')
#             begin_time,end_time = float(line_split[0]),float(line_split[1])
#             label = line_split[-9]
#             words = [x.split('+')[-1] for x in line_split[4].split('|')]


#             data.append(words)
#             labels.append(label)
#             frames.append(librosa.core.time_to_frames((begin_time,end_time)))

#             line = file.readline()
#     return data,labels,frames

