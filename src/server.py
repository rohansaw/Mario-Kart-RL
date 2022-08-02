import mss
from pythonServer import PythonServer
import time
import sys
from http.server import HTTPServer
print("HSJ")
print("HHSKDHKJSHDKJS")
print("TEST")
print("HSD")
print("HSD")
print("HSD")
#from urllib.parse import urlparse, parse_qs
print("1")
print("2")
print("3")
print("4")
print("5")
#import numpy as np
HOST_NAME = "0.0.0.0"
PORT = 8070
print(sys.version)
print("HJDW SKJ ")


print("HSHDKJSHSK")
#mss_grabber = mss.mss()
print("SHDS")
time.sleep(2)
print("S .  ")


print("HA")
server = HTTPServer((HOST_NAME, PORT), PythonServer)
#print(f"Server started http://{HOST_NAME}:{PORT}")
try:
    print("Sdaw")
    server.serve_forever()
except KeyboardInterrupt:
    server.server_close()
    print("Server stopped successfully")
    sys.exit(0)
