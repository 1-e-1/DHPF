"""Refresh the local report while long training queues execute."""
import json
import os
import time
import psutil
from .protocol import RESULTS
from .run import write_json
from .report import report


if __name__=='__main__':
    while True:
        report()
        workers=[]
        for device in ['cpu','cuda']:
            path=RESULTS/f'worker_{device}.json'
            if path.exists():
                state=json.loads(path.read_text());state['process_alive']=psutil.pid_exists(state['pid']);workers.append(state)
        write_json(RESULTS/'monitor.json',dict(pid=os.getpid(),last_refresh_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),workers=workers))
        if len(workers)==2 and all(not w['process_alive'] for w in workers):break
        time.sleep(60)
