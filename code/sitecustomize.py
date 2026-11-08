"""Enforce Python offline mode when explicitly requested by the runner."""
import os
if os.environ.get('REPLICATION_NO_NETWORK')=='1':
 import socket
 def blocked(*args,**kwargs):raise RuntimeError('Network disabled for offline replication')
 socket.create_connection=blocked
 socket.socket.connect=blocked
 socket.socket.connect_ex=blocked
