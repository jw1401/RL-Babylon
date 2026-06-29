import http.server, functools

print("Dashboard läuft auf http://localhost:8000/.metrics/dashboard.html ... (Beenden mit Strg+C)")

try:
    # directory="." bedeutet: Nutze genau den Ordner, in dem dieses Skript liegt
    http.server.HTTPServer(("", 8000), functools.partial(http.server.SimpleHTTPRequestHandler, directory=".")).serve_forever()

except KeyboardInterrupt:
    print("\nServer sauber geschlossen.")