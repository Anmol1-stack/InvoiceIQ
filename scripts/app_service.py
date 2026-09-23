import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8000


class AppHandler(BaseHTTPRequestHandler):

    def do_POST(self):

        if self.path != "/ask":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        request_data = json.loads(body.decode("utf-8"))

        request = urllib.request.Request(
            "http://orchestrator:8003/ask",
            data=json.dumps(request_data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(request) as response:
                result = response.read()
                status = response.status
        except urllib.error.HTTPError as error:
            result = error.read()
            status = error.code
        except urllib.error.URLError as error:
            result = json.dumps({
                "error": "Application cannot reach the orchestrator service.",
                "detail": str(error)
            }).encode("utf-8")
            status = 503

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(result)))
        self.end_headers()

        self.wfile.write(result)

    def log_message(self, format, *args):
        print("Application Service:", format % args)


print(f"Application Service running on port {PORT}")

server = HTTPServer(("0.0.0.0", PORT), AppHandler)
server.serve_forever()
