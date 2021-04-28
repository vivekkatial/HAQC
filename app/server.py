import http.server
import socketserver
from urllib.parse import urlparse
from urllib.parse import parse_qs


class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Send a '200 okay' response
        self.send_response(200)

        self.send_header("Content-type", "text/html")

        self.end_headers()

        query_components = parse_qs(urlparse(self.path).query)

        filename = "No file selected"

        ### CODE TO RETRIEVE FROM DATABASE STORED AT THIS SERVER###

        if "filename" in query_components:
            filename = query_components["filename"][0]
            with open(f"{filename}.json", "rb") as file:
                self.wfile.write(file.read())
        else:
            self.send_response(204)

        return


# Create object of class above
handler_object = MyHttpRequestHandler

PORT = 8000
my_server = socketserver.TCPServer(("", PORT), handler_object)

# Start the server
my_server.serve_forever()
