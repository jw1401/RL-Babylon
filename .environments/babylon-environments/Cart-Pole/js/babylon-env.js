// BabylonEnv.js implements the RL Environment for Training.

import { BaseEnv } from "../../../lib/index.js";
import envConfig from './env-config.js';

export class BabylonEnv extends BaseEnv {

    constructor(wsUrl, canvasId) {
        super(wsUrl, canvasId, envConfig);

        // Physics parameters for physics simulation
        this.x = 0;                                             // cart position
        this.x_dot = 0;                                         // cart velocity
        this.theta = 0.05;                                      // pole angle (radians)
        this.theta_dot = 0;                                     // pole angular velocity
        this.gravity = 9.8;
        this.mass_cart = 1.0;
        this.mass_pole = 0.1;
        this.length = 1.0;                                      // half pole length
        this.force_mag = 10.0;
        this.tau = 0.02;                                        // time step for 50 FPS
        this.total_mass = this.mass_cart + this.mass_pole;
        this.polemass_length = this.mass_pole * this.length;

        // Variables important for training
        this.fixedDt = this.tau                                 // tick time = 20 ms = 0.02 sec --> 50 HZ framerate 
        this.setupAll();
    }

    // Setup the RL Scene based on individula design
    setupScene() {

        // Camera and lighting
        const camera = new BABYLON.ArcRotateCamera("camera", Math.PI / 2, Math.PI / 3, 10, new BABYLON.Vector3(0, 1, 0), this.scene);
        camera.attachControl(this.canvas, true);
        new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), this.scene);

        // Ground plane
        const ground = BABYLON.MeshBuilder.CreateGround("ground", { width: 5.6, height: 0.2 }, this.scene);
        const groundMat = new BABYLON.StandardMaterial("groundMat", this.scene);
        groundMat.diffuseColor = new BABYLON.Color3(0.1, 0.1, 0.1); // dark gray
        ground.material = groundMat;

        // Cart (a box)
        const cartWidth = 0.8
        this.cart = BABYLON.MeshBuilder.CreateBox("cart", { width: cartWidth, height: 0.4, depth: 0.5 }, this.scene);
        this.cart.position.y = 0.2;

        // ---- Cart color ----
        const cartMat = new BABYLON.StandardMaterial("cartMat", this.scene);
        cartMat.diffuseColor = new BABYLON.Color3(0.1, 0.1, 0.1); // dark gray
        this.cart.material = cartMat;

        // Pole creation with pivot at bottom (joint) 
        const poleHeight = 2.0;
        this.pole = BABYLON.MeshBuilder.CreateBox("pole", { width: 0.1, height: poleHeight, depth: 0.1 }, this.scene);
        this.pole.position.y = (poleHeight / 2) + 0.2; // pole's center is poleHeight/2 above its bottom

        // ---- Pole color ----
        const poleMat = new BABYLON.StandardMaterial("poleMat", this.scene);
        poleMat.diffuseColor = new BABYLON.Color3(1, 1, 0); // yellow
        this.pole.material = poleMat;

        // Set the pivot of the pole to its bottom so rotations happen around that bottom point.
        const pivotToBottom = BABYLON.Matrix.Translation(0, poleHeight / 2, 0);
        this.pole.setPivotMatrix(pivotToBottom);

        this.pole.parent = this.cart;

        // Pole hinge
        const tip = BABYLON.MeshBuilder.CreateSphere("tip", { diameter: 0.15 }, this.scene);
        tip.position.y = -poleHeight / 2; // top of the pole
        tip.parent = this.pole;
        const tipMat = new BABYLON.StandardMaterial("tipMat", this.scene);
        tipMat.diffuseColor = new BABYLON.Color3(200 / 255, 162 / 255, 200 / 255);
        tip.material = tipMat;

        // ---- Draw termination boundaries ----
        const limit = 2.4 + cartWidth / 2; // same as Gym CartPole
        const boundaryMat = new BABYLON.StandardMaterial("boundaryMat", this.scene);
        boundaryMat.emissiveColor = new BABYLON.Color3(1, 0, 0);

        // Left boundary
        const leftBoundary = BABYLON.MeshBuilder.CreateBox("leftBoundary", { width: 0.05, height: 0.5, depth: 0.2 }, this.scene);
        leftBoundary.position.x = -limit;
        leftBoundary.position.y = 0.25;
        leftBoundary.material = boundaryMat;

        // Right boundary
        const rightBoundary = leftBoundary.clone("rightBoundary");
        rightBoundary.position.x = limit;
        rightBoundary.material = boundaryMat;
        // --------------------------------------
    }

    // Setup individual UI
    setupUI() {
    }

    // Reset logic for RL Scene
    async resetEnv() {

        this.x = 0;
        this.x_dot = 0;
        this.theta = (Math.random() - 0.5) * 0.1;
        this.theta_dot = 0;
        this.cart.position.x = 0;
        this.pole.rotation.z = -this.theta;

        // Render scene
        this.scene.render()

        const s = this.result.data.state;
        s[0] = this.x;
        s[1] = this.x_dot;
        s[2] = this.theta;
        s[3] = this.theta_dot;

        // Return state
        return this.result
    }

    // Step logic for RL Scene
    async stepEnv(action) {

        // Apply force: 0 = left, 1 = right
        const force = (action === 1 ? this.force_mag : -this.force_mag);

        // Equations of motion (simplified physics)
        const costheta = Math.cos(this.theta);
        const sintheta = Math.sin(this.theta);

        const temp = (force + this.polemass_length * this.theta_dot * this.theta_dot * sintheta) / this.total_mass;
        const thetaacc = (this.gravity * sintheta - costheta * temp) / (this.length * (4.0 / 3.0 - this.mass_pole * costheta * costheta / this.total_mass));
        const xacc = temp - this.polemass_length * thetaacc * costheta / this.total_mass;

        // Integrate the state
        this.x += this.tau * this.x_dot;
        this.x_dot += this.tau * xacc;
        this.theta += this.tau * this.theta_dot;
        this.theta_dot += this.tau * thetaacc;

        // Update visuals
        this.cart.position.x = this.x;
        this.pole.rotation.z = -this.theta;

        // Check termination
        const terminated = this.x < -2.4 || this.x > 2.4 || this.theta < -0.2 || this.theta > 0.2;
        const reward = terminated ? -1.0 : 1.0;

        // if max steps reached then episode is truncated == Time Horizon
        const truncated = this.stepCount >= this.UI.MAX_STEPS;

        // Render Scene
        this.scene.render()

        const data = this.result.data;
        const s = data.state;
        s[0] = this.x;
        s[1] = this.x_dot;
        s[2] = this.theta;
        s[3] = this.theta_dot;

        data.reward = reward;
        data.terminated = terminated;
        data.truncated = truncated;

        // Return state, reward, done [truncated or terminated]
        return this.result
    }
}

