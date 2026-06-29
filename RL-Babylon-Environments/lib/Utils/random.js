// Helper for randomization in environment
export class Random {

    // Generate a random number between min and max
    static randomBetween(min, max) {
        return Math.random() * (max - min) + min;
    }

    // Generate a random integer between min and max (inclusive)
    static randomInt(min, max) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }

    // Example: random position vector for Babylon.js
    static randomPosition(min = -5, max = 5) {
        return {
            x: Random.randomBetween(min, max),
            y: Random.randomBetween(min, max),
            z: Random.randomBetween(min, max),
        };
    }
}



