import { BSON } from './bson.mjs';

export class BsonSocketClient {

    constructor(wsUrl) {

        this._wsUrl = wsUrl;
        this._ws = null;
        this._listeners = {}; // Speicher für die Event-Callbacks
    }

    // Ermöglicht es BaseEnv, auf bestimmte Events zu hören
    on(eventType, callback) {

        this._listeners[eventType] = callback;
    }

    connect() {

        this._ws = new WebSocket(this._wsUrl);
        this._ws.binaryType = "arraybuffer";

        this._ws.onopen = () => console.log("WebSocket connected via BSON");
        this._ws.onclose = () => console.log("WebSocket closed");
        this._ws.onerror = (err) => console.error("WebSocket error: ", err);

        this._ws.onmessage = (msg) => {

            if (!(msg.data instanceof ArrayBuffer)) {

                console.warn("Received non-binary data over WebSocket.");
                return;
            }

            // BSON extrahieren
            const uint8View = new Uint8Array(msg.data);
            const data = BSON.deserialize(uint8View);

            // Events basierend auf dem Inhalt der Nachricht triggern
            if (data.init && this._listeners['init']) {

                this._listeners['init']();

            } else if (data.reset && this._listeners['reset']) {

                this._listeners['reset']();

            } else if (data.action !== undefined && this._listeners['action']) {

                this._listeners['action'](data.action);

            } else if (data.level !== undefined && this._listeners['curriculum_level']) {

                this._listeners['curriculum_level'](data.level);

            }
        };
    }

    send(obj) {

        if (this._ws && this._ws.readyState === WebSocket.OPEN) {

            const bsonPacket = BSON.serialize(obj);
            this._ws.send(bsonPacket);

        } else {
            
            console.error("Cannot send BSON: WebSocket is not open.");
        }
    }
}