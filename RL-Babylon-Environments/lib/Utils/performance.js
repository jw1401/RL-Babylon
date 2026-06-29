// Performance.js implements methods to measure performance or set time realted issues

export class Performance {

    constructor(){

        this.lastTime = performance.now();
        this.fps = 0;
        this.fpsSmoothing = 0.9;  // 0.9 = träge, 0.1 = sehr reaktiv
        this.fpsSamples = [];
        this.maxSamples = 30; // z.B. 30 Frames mitteln
    }

    computeFPS(){

        // Compute fps and display
        const now = performance.now();
        const currentFPS = 1000 / (now - this.lastTime)
        this.lastTime = now;

        // Gleitender Mittelwert (exponentiell geglättet)
        this.fps = this.fps * this.fpsSmoothing + currentFPS * (1 - this.fpsSmoothing);

        return this.fps.toFixed(0)
    }


    computeFPSWithSamples() {
        
        const now = performance.now();
        const fps = 1000 / (now - this.lastTime);
        this.lastTime = now;
    
        this.fpsSamples.push(fps);

        if (this.fpsSamples.length > this.maxSamples) {
            this.fpsSamples.shift(); // ältesten Wert entfernen
        }
    
        const avgFPS = this.fpsSamples.reduce((a, b) => a + b, 0) / this.fpsSamples.length;

        return avgFPS.toFixed(0)
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

   
}
