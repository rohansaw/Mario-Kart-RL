from urllib.parse import urlparse, parse_qs
import mss
from pythonServer import PythonServer
from http.server import SimpleHTTPRequestHandler
import time
import sys
from http.server import HTTPServer
HOST_NAME = "0.0.0.0"
PORT = 8070
print(sys.version)
print("HJDW SKJ ")


print("HSHDKJSHSK")
mss_grabber = mss.mss()
print("SHDS")
time.sleep(2)
print("S .  ")


class PythonServer(SimpleHTTPRequestHandler):
    def do_GET(self):
        print("SS")
        top = None
        left = None
        width = None
        height = None
        if self.path == '/image':
            print("S")
            if '?' in self.path:
                path, tmp = self.path.split('?', 1)
                qs = parse_qs(urlparse.parse_qs(tmp).query)
                top = qs['top'][0]
                left = qs['left'][0]
                width = qs['width'][0]
                height = qs['height'][0]
            image_array = mss_grabber.grab()
            # {"top": top,
            # "left": left,
            #                                "width": width,
            #                                "height": height})
            #      dtype=np.uint8)
            # print(image_array)
            self.send_response(200, "OK")
            self.end_headers()
            self.wfile.write(bytes(image_array, "utf-8"))


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
