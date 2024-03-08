import logging
import torch, time, gc

logger = logging.getLogger('irtorch')

START_TIME = None

def start_timer():
    """
    Start the timer for measuring execution time and memory usage.
    """
    global START_TIME
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    START_TIME = time.time()

def end_timer_and_print(local_msg):
    """
    End the timer and print the execution time and memory usage.
    """
    end_time = time.time()
    torch.cuda.synchronize()
    logger.info("\n%s", local_msg)
    logger.info("Total execution time = %s sec", end_time - START_TIME)
    logger.info("Max memory used by tensors = %s bytes", torch.cuda.max_memory_allocated())