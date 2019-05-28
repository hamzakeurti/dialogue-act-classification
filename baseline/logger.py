from datetime import datetime
import logging

logs_dir = 'logs/'
log_file = ''

def INIT_LOG(log=''):
    global log_file
    if not log:
        return
    log_file = logs_dir + log
    logging.basicConfig(filename=log_file,level=logging.INFO)



def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    if not log_file:
        print('[' + display_now + ']' + ' ' + msg)
    else:
        logging.info('[' + display_now + ']' + ' ' + msg)

