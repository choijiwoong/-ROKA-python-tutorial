#[1. monitoring making&deleting file]
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class Target:
    watchDir=os.getcwd()#assign directory that will be observed

    def __init__(self):
        self.observer=Observer()#make Observer object

    def run(self):
        event_handler=Handler()#make handler object
        self.observer.schedule(event_handler, self.watchDir, recursive=True)#binding handler & watchdir, is include subdirectory
        self.observer.start()#make run

        try:
            while True:
                time.sleep(1)
        except:
            self.observer.stop()#wait observer
            print("Error")
            self.observer.join()#wait working of handler class
class Handler(FileSystemEventHandler):
    def on_moved(self, event):
        print(event)
    def on_created(self, event):
        print(event)
    def on_deleted(self, event):
        print(event)
    def on_modified(self, event):
        print(event)

if __name__=='__main__':
    print(os.getcwd())
    w=Target()
    w.run()
