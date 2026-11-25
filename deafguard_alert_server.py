import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "0.0.0.0"
PORT = 8765

latest_alert = {
    "event": "monitoring",
    "severity": "NONE",
    "message": "DeafGuard is waiting for a confirmed alert.",
    "symbol": "🛡️",
    "vibration": "none",
    "priority": 0,
}



class AlertHandler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "online"})

        elif self.path == "/latest-alert":
            self.send_json(latest_alert)

        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        global latest_alert

        if self.path != "/alert":
            self.send_json({"error": "not found"}, 404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            incoming_alert = json.loads(body.decode("utf-8"))

            # Sound alerts start the multimodal response loop.
            # ISL messages are communication responses and do not create a new prompt.
            if (
                incoming_alert.get("source") != "isl"
                and incoming_alert.get("severity") not in ("NONE", None)
            ):
                incoming_alert["prompt"] = (
                    "Do you need assistance? Show HELP or PAIN."
                )
                incoming_alert["response_mode"] = "WAITING_FOR_ISL_RESPONSE"

            latest_alert = incoming_alert

            print("Alert received:", latest_alert)
            self.send_json({"status": "alert_updated"})

        except Exception as error:
            self.send_json(
                {"error": f"invalid alert data: {error}"},
                400,
            )

    def log_message(self, format, *args):
        print(format % args)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), AlertHandler)

    print(f"DeafGuard alert server running on port {PORT}")
    print("GET  /health")
    print("GET  /latest-alert")
    print("POST /alert")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAlert server stopped.")
        server.server_close()
