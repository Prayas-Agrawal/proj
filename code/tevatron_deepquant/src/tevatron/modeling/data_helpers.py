
import string
import torch
import threading
import queue
import csv
import time
import atexit

def get_skiplist(tokenizer, mask_punctuation):
    skiplist = None
    if(tokenizer is None): return None
    if mask_punctuation:
            skiplist = {w: True for symbol in string.punctuation
                            for w in [symbol, tokenizer.encode(symbol, add_special_tokens=False)[0]]}
    return skiplist


def mask(input_ids, skiplist, pad_token):
    if(skiplist is None or len(skiplist) == 0): return None
    device = input_ids.get_device()
    mask = [[(x not in skiplist) and (x != pad_token) for x in d] for d in input_ids.cpu().tolist()]
    mask = torch.tensor(mask, device=device).unsqueeze(2).float()
    return mask



class AsyncDataSaver:
    """Asynchronously saves query data (queryid, docid, alpha, scores) to a CSV file."""
    
    def __init__(self, output_file="query_data.csv", flush_interval=2):
        """
        Initialize the async saver.
        
        Args:
            output_file (str): File path for saving data.
            flush_interval (int): Time in seconds between periodic writes.
        """
        self.output_file = output_file
        self.flush_interval = flush_interval
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Initialize CSV file with headers if it's a new file
        with open(self.output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["queryid", "docid", "alpha", "scores"])  # Column headers

        # Start background saving thread
        self.saver_thread = threading.Thread(target=self._async_saver, daemon=True)
        self.saver_thread.start()

        # Register stop function to be called when the program exits
        atexit.register(self.stop)

    def save_data(self, queryid, docid, alpha, scores):
        """
        Adds data to the queue for asynchronous saving.
        
        Args:
            queryid (torch.Tensor): Query IDs.
            docid (torch.Tensor): Document IDs.
            alpha (torch.Tensor): Alpha values.
            scores (torch.Tensor): Scores.
        """
        for q, d, a, s in zip(queryid.tolist(), docid.tolist(), alpha.tolist(), scores.tolist()):
            self.data_queue.put((q, d, a, s))

    def _async_saver(self):
        """Background thread function to save data from the queue to a CSV file."""
        while not self.stop_event.is_set():
            time.sleep(self.flush_interval)  # Control write frequency

            # Collect batch of data
            batch_data = []
            while not self.data_queue.empty():
                batch_data.append(self.data_queue.get())

            # Write batch to file
            if batch_data:
                with open(self.output_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(batch_data)

    def stop(self):
        """Ensures all data is saved and stops the background thread when the program exits."""
        if not self.stop_event.is_set():  # Prevent multiple calls
            print("Stopping AsyncDataSaver and saving remaining data...")
            self.stop_event.set()

            # Save remaining data in the queue
            batch_data = []
            while not self.data_queue.empty():
                batch_data.append(self.data_queue.get())

            if batch_data:
                with open(self.output_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(batch_data)

            self.saver_thread.join()  # Ensure thread finishes execution
            print("Data saver stopped successfully.")

