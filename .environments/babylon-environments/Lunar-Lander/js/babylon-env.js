import { Random, BaseEnv } from '../../../lib/index.js'
import envConfig from './env-config.js';

export class BabylonEnv extends BaseEnv {

    constructor(wsUrl, canvasId) {
        super(wsUrl, canvasId, envConfig);

        this.size = 30.0;       // Map width
        this.gravity = -9.81;   // Downward gravity affecting Z-axis (treating Z as height)
        this.fixedDt = 0.02;    // 50 Hz frame rate

        // Physics variables for the Lander
        this.vx = 0.0;
        this.vz = 0.0;
        this.thrustPower = 15.0; // Acceleration power of the thrusters

        this.setupAll();
    }

    async setupScene() {

        // Camera and light (Positioned to give a 2D side-scrolling perspective)
        this.camera = new BABYLON.ArcRotateCamera("camera", Math.PI / 2, Math.PI / 1.1, 50, new BABYLON.Vector3(0, 0, 7.5), this.scene);
        this.camera.attachControl(this.canvas, true);
        new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), this.scene);

        // Lander (The Agent) - Box shape
        this.agent = BABYLON.MeshBuilder.CreateBox("lander", { width: 1, height: 1, depth: 1 }, this.scene);
        const agentMaterial = new BABYLON.StandardMaterial("landerMat", this.scene);
        agentMaterial.diffuseColor = new BABYLON.Color3(1, 1, 1);
        agentMaterial.diffuseTexture = new BABYLON.Texture("./js/texture/lander.jpg", this.scene);
        agentMaterial.specularColor = new BABYLON.Color3(0, 0, 0);
        agentMaterial.emissiveColor = agentMaterial.diffuseColor;
        this.agent.material = agentMaterial;

        // Landing Pad (The Target Zone)
        this.padSize = 3.0;
        this.target = BABYLON.MeshBuilder.CreateBox("landingPad", { width: this.padSize, height: 3, depth: 0.5 }, this.scene);
        this.target.position.z = 0.1; // Ground level height
        const targetMat = new BABYLON.StandardMaterial("padMat", this.scene);
        targetMat.diffuseColor = new BABYLON.Color3(0, 0, 1); // Blue pad
        this.target.material = targetMat;

        // Ground/Moon Surface
        this.ground = BABYLON.MeshBuilder.CreateGround("moonSurface", { width: this.size, height: 30 }, this.scene);
        this.ground.rotation.x = Math.PI / 2; // Orient flat against our 2D plane viewport
        const moonMaterial = new BABYLON.StandardMaterial("moonMat", this.scene);
        moonMaterial.diffuseTexture = new BABYLON.Texture("./js/texture/moon.jpg", this.scene);
        moonMaterial.diffuseTexture.uScale = 1.0;
        moonMaterial.diffuseTexture.vScale = 1.0;
        moonMaterial.specularColor = new BABYLON.Color3(0, 0, 0);
        this.ground.material = moonMaterial;
    }

    async setupUI() {
        this.UI.addSlider("Thrust", 5, 25, 1, this.thrustPower);
    }

    async resetEnv() {
        // Reset Lander position to the top area with random X variance
        this.agent.position.x = Random.randomBetween(-this.size / 3, this.size / 3);
        this.agent.position.z = 15.0; // Start high up
        this.agent.position.y = 0;   // Locked 2D axis

        // Reset velocities
        this.vx = Random.randomBetween(-2.0, 2.0);
        this.vz = 0.0;

        // Randomize Landing Pad position along the floor
        this.target.position.x = Random.randomBetween(-this.size / 4, this.size / 4);

        this.scene.render();

        // State vector: [x_pos, z_pos, x_vel, z_vel]
        const s = this.result.data.state;
        s[0] = this.agent.position.x;
        s[1] = this.agent.position.z;
        s[2] = this.vx;
        s[3] = this.vz;
        s[4] = this.agent.position.x - this.target.position.x; // <-- ADD THIS

        return this.result;
    }

    async stepEnv(action) {
        // 1. Apply Physics Engine Actions
        // action 0: Do Nothing (Gravity takes over)
        if (action === 1) this.vz += this.UI.THRUST * this.fixedDt;         // Main Thrust Up
        if (action === 2) this.vx -= (this.UI.THRUST * 0.5) * this.fixedDt; // Left Thruster
        if (action === 3) this.vx += (this.UI.THRUST * 0.5) * this.fixedDt; // Right Thruster

        // Apply constant gravity over time delta
        this.vz += this.gravity * this.fixedDt;

        // Update positions using discrete kinematic integrations
        this.agent.position.x += this.vx * this.fixedDt;
        this.agent.position.z += this.vz * this.fixedDt;

        // 2. Condition Evaluation (Rewards & Terminations)
        let reward = 0.0;
        let terminated = false;

        // Step penalty to encourage landing efficiently
        reward -= 0.1;

        // Distance reward (closer to the pad is better)
        const distanceToPad = Math.abs(this.agent.position.x - this.target.position.x);
        reward += 0.2 * (1 / (1 + distanceToPad));

        // Check Ground Touchdown (Height <= 0.5 due to half box sizing offset)
        if (this.agent.position.z <= 0.5) {
            terminated = true;
            this.agent.position.z = 0.5; // Prevent clipping through floor

            // Check alignment matching the pad's horizontal coordinates
            const landedOnPad = Math.abs(this.agent.position.x - this.target.position.x) < (this.padSize / 2);
            const safeVelocity = Math.abs(this.vz) < 4.0 && Math.abs(this.vx) < 2.0;

            if (landedOnPad && safeVelocity) {
                reward += 100.0; // Successful Soft Landing!
                console.log("🚀 Safe Landing!");
            } else if (landedOnPad && !safeVelocity) {
                reward -= 15.0;  // Hit pad too hard
                console.log("💥 Hard Crash on Pad!");
            } else {
                reward -= 50.0;  // Crashed into open terrain
                console.log("💀 Missed Pad/Crashed!");
            }
        }

        // Check Out of Bounds parameters
        if (Math.abs(this.agent.position.x) >= this.size / 2 || this.agent.position.z > 22.0) {
            terminated = true;
            reward = -100.0;
            console.log("❌ Flew out of bounds!");
        }

        const truncated = this.stepCount >= this.UI.MAX_STEPS;

        // 3. Environment Lifecycle Syncing
        this.scene.render();

        const data = this.result.data;
        const s = data.state;
        s[0] = this.agent.position.x;
        s[1] = this.agent.position.z;
        s[2] = this.vx;
        s[3] = this.vz;
        s[4] = this.agent.position.x - this.target.position.x;

        data.reward = reward;
        data.terminated = terminated;
        data.truncated = truncated;

        return this.result;
    }
}