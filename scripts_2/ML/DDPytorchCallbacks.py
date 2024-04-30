import numpy as np
import torch
import time 

class EarlyStopping:
    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if val_loss < self.val_loss_min:
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if val_loss < self.val_loss_min:
            if self.verbose:
                self.trace_func(
                    f"Validation loss decreased ({self.val_loss_min:.6f} to {val_loss:.6f}). Saving model ..."
                )
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss

class TimedStopping:
    def __init__(self, max_seconds=None, verbose=1):
        self.max_seconds = max_seconds
        self.verbose = verbose
        self.start_time = time.time()

    def should_stop(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_seconds:
            if self.verbose:
                print(f"Stopping after {elapsed_time:.2f} seconds.")
            return True
        return False
