import threading
import time

class GraphExecutor(threading.Thread):
    def __init__(self,interval,update_callback ):
        threading.Thread.__init__(self)
        self.update_callback = update_callback
        self.running = False
        self.interval = interval


    def run(self):
        self.running = True

        while self.running:
            self.update_callback()
            time.sleep(self.interval)

    def stop(self):
        self.running = False
