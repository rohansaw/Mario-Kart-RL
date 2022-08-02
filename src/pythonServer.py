from http.server import SimpleHTTPRequestHandler


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
                #qs = parse_qs(urlparse.parse_qs(tmp).query)
                #top = qs['top'][0]
                #left = qs['left'][0]
                #width = qs['width'][0]
                #height = qs['height'][0]
            # image_array = mss_grabber.grab({"top": top,
            #                               "left": left,
            #                                "width": width,
            #                                "height": height})
            #      dtype=np.uint8)
            # print(image_array)
            self.send_response(200, "OK")
            self.end_headers()
            self.wfile.write(bytes("image_array", "utf-8"))
